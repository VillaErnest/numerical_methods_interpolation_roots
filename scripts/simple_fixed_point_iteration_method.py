import tkinter as tk
from tkinter import messagebox
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys

COLOR_BG_MAIN = "#f8fafc"
COLOR_BG_CARD = "#ffffff"
COLOR_ACCENT = "#f97316"
COLOR_ACCENT_HOVER = "#ea580c"
COLOR_TEXT_DARK = "#1f2937"
COLOR_TEXT_MUTED = "#6b7280"
COLOR_BORDER = "#e5e7eb"
COLOR_SUCCESS = "#10b981"
COLOR_INFO = "#3b82f6"

SAFE_FUNCTIONS = {name: getattr(np, name) for name in dir(np) if not name.startswith("_")}
SAFE_FUNCTIONS.update({"pi": np.pi, "e": np.e})

X_PRECISION = 10
ERR_PRECISION = 3
COL_WIDTH_ITER = 6
COL_WIDTH_X = X_PRECISION + 10 
COL_WIDTH_ERR = 14
TOTAL_WIDTH = COL_WIDTH_ITER + COL_WIDTH_X * 2 + COL_WIDTH_ERR

class FixedPointSolverGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Enhanced Fixed-Point Iteration Solver")
        self.root.geometry("1200x850") 
        self.root.configure(bg=COLOR_BG_MAIN)
        
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.g_candidates = []
        self.fig_canvas = None
        self.best_g_str = None

        self._create_header()
        
        self.main_content = tk.Frame(self.root, bg=COLOR_BG_MAIN)
        self.main_content.pack(fill="both", expand=True, padx=16, pady=16)

        self._create_input_area()
        self._create_visualization_and_output_area()

        self.generate_g_candidates(initial_load=True) 
        
        self.root.mainloop()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to exit the application?"):
            self.root.destroy()
            sys.exit(0)

    def _create_header(self):
        header = tk.Frame(self.root, bg=COLOR_ACCENT, height=120)
        header.pack(fill="x")
        tk.Label(header, text="Fixed-Point Iteration Solver", font=("Segoe UI", 24, "bold"), bg=COLOR_ACCENT, fg="white").pack(pady=(15,0))
        tk.Label(header, text="Open Method", font=("Segoe UI", 14), bg=COLOR_ACCENT, fg="white").pack(pady=(0,15))

    def _create_input_area(self):
        self.input_card = tk.Frame(self.main_content, bg=COLOR_BG_CARD, bd=1, relief="flat", highlightbackground=COLOR_BORDER, highlightthickness=1)
        self.input_card.pack(fill="x", pady=(0, 10))
        
        self.input_inner = tk.Frame(self.input_card, bg=COLOR_BG_CARD, padx=15, pady=15)
        self.input_inner.pack(fill="x")
        
        self.left_col = tk.Frame(self.input_inner, bg=COLOR_BG_CARD)
        self.left_col.pack(side="left", padx=10, pady=5, fill="x", expand=True)

        self.eq_var = tk.StringVar(value="x**2 - 2")
        self.tol_var = tk.StringVar(value="1e-4")
        self.x0_var = tk.StringVar(value="1.5")
        self.max_iter_var = tk.StringVar(value="50")
        self.ignore_tol_var = tk.BooleanVar(value=False)
        self.auto_pick_var = tk.BooleanVar(value=True)

        self._create_input_field(self.left_col, "Equation (f(x)=0):", 0, self.eq_var, "x**2 - 2", "Equation to solve.")
        self._create_input_field(self.left_col, "Initial Guess (x₀):", 1, self.x0_var, "1.5", "Starting value for iteration.")
        self._create_input_field(self.left_col, "Stopping Rel. Error:", 2, self.tol_var, "1e-4", "Relative error for stopping criterion.")
        self._create_input_field(self.left_col, "Max Iterations (N):", 3, self.max_iter_var, "50", "Maximum number of iterations.")
        
        options_frame = tk.Frame(self.left_col, bg=COLOR_BG_CARD)
        options_frame.grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))
        tk.Checkbutton(options_frame, text="Ignore Stopping Error", variable=self.ignore_tol_var, bg=COLOR_BG_CARD, font=("Segoe UI", 10)).pack(side="left", padx=5)
        tk.Checkbutton(options_frame, text="Auto-Pick Best g(x)", variable=self.auto_pick_var, bg=COLOR_BG_CARD, font=("Segoe UI", 10)).pack(side="left", padx=5)
        
        self.right_col = tk.Frame(self.input_inner, bg=COLOR_BG_CARD)
        self.right_col.pack(side="right", padx=10, pady=5, fill="both", expand=True)
        
        tk.Label(self.right_col, text="Generated g(x) Candidates (Pick one if Auto-Pick is OFF):", font=("Segoe UI", 10, "bold"), fg=COLOR_TEXT_DARK, bg=COLOR_BG_CARD).pack(anchor="w", pady=(0, 5))
        
        list_frame = tk.Frame(self.right_col, bg=COLOR_BORDER)
        list_frame.pack(fill="both", expand=True)
        
        scrollbar = tk.Scrollbar(list_frame, orient="vertical")
        self.candidate_list = tk.Listbox(list_frame, height=6, font=("Courier New", 10), bd=0, yscrollcommand=scrollbar.set, selectmode=tk.SINGLE)
        scrollbar.config(command=self.candidate_list.yview)
        
        scrollbar.pack(side="right", fill="y")
        self.candidate_list.pack(side="left", fill="both", expand=True)

        button_frame = tk.Frame(self.right_col, bg=COLOR_BG_CARD)
        button_frame.pack(fill="x", pady=10)

        tk.Button(button_frame, text="Generate g(x) 🔄", command=self.generate_g_candidates, bg=COLOR_ACCENT, fg="white", font=("Segoe UI", 10, "bold"), activebackground=COLOR_ACCENT_HOVER).pack(side="left", fill="x", expand=True, padx=(0, 5))
        tk.Button(button_frame, text="Start Iteration ▶️", command=self.run_iteration, bg=COLOR_SUCCESS, fg="white", font=("Segoe UI", 10, "bold"), activebackground="#059669").pack(side="right", fill="x", expand=True, padx=(5, 0))

    def _create_input_field(self, parent_frame, label_text, row, variable, default_value, tooltip):
        variable.set(default_value) 
        
        label = tk.Label(parent_frame, text=label_text, font=("Segoe UI", 10), fg=COLOR_TEXT_DARK, bg=COLOR_BG_CARD)
        label.grid(row=row, column=0, sticky="w", pady=5, padx=(0, 10))
        
        entry = tk.Entry(parent_frame, textvariable=variable, font=("Segoe UI", 10), bd=1, relief="flat", highlightbackground=COLOR_BORDER, highlightthickness=1, width=25)
        
        entry.grid(row=row, column=1, sticky="w", pady=5)

    def _create_visualization_and_output_area(self):
        self.content_panel = tk.Frame(self.main_content, bg=COLOR_BG_MAIN)
        self.content_panel.pack(fill="both", expand=True)

        self.plot_frame = tk.Frame(self.content_panel, bg=COLOR_BG_CARD, bd=1, highlightbackground=COLOR_BORDER, highlightthickness=1)
        self.plot_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))
        
        tk.Label(self.plot_frame, text="Function Plot (f(x) and g(x) vs. y=x)", font=("Segoe UI", 12, "bold"), bg=COLOR_BG_CARD, fg=COLOR_ACCENT).pack(pady=5)
        
        self.canvas_container = tk.Frame(self.plot_frame, bg=COLOR_BG_CARD)
        self.canvas_container.pack(fill="both", expand=True)

        self.output_frame = tk.Frame(self.content_panel, bg=COLOR_BG_CARD, bd=1, highlightbackground=COLOR_BORDER, highlightthickness=1)
        self.output_frame.pack(side="right", fill="both", expand=True, padx=(8, 0))

        tk.Label(self.output_frame, text="Iteration Results", font=("Segoe UI", 12, "bold"), bg=COLOR_BG_CARD, fg=COLOR_ACCENT).pack(pady=5)

        text_container = tk.Frame(self.output_frame, padx=10, pady=5, bg=COLOR_BG_CARD)
        text_container.pack(fill="both", expand=True)
        
        self.result_text = tk.Text(text_container, wrap="none", bg=COLOR_BG_MAIN, fg=COLOR_TEXT_DARK, bd=0, relief="flat", padx=10, pady=10, font=("Courier New", 10))
        
        scroll_y = tk.Scrollbar(text_container, command=self.result_text.yview)
        scroll_y.pack(side="right", fill="y")
        
        self.result_text.pack(side="left", fill="both", expand=True)
        self.result_text.config(yscrollcommand=scroll_y.set)

        self.result_text.tag_config("header", font=("Segoe UI", 12, "bold"), foreground=COLOR_INFO) 
        self.result_text.tag_config("table_header", font=("Courier New", 10, "bold"), foreground=COLOR_TEXT_DARK)
        self.result_text.tag_config("data", font=("Courier New", 10), foreground=COLOR_TEXT_MUTED)
        self.result_text.tag_config("success", font=("Courier New", 10, "bold"), foreground=COLOR_SUCCESS)
        self.result_text.tag_config("error", font=("Courier New", 10, "bold"), foreground=COLOR_ACCENT)
        
        self.result_text.config(state=tk.DISABLED)

    def generate_g_candidates(self, equation_str=None, var_str="x", initial_load=False):
        if equation_str is None:
            equation_str = self.eq_var.get()
            
        self.g_candidates = []
        x = sp.symbols(var_str)
        
        try:
            f = sp.sympify(equation_str)
        except Exception as e:
            if not initial_load:
                messagebox.showerror("Error", f"Invalid equation: {e}")
            return

        try:
            g1_expr = 1 + 2/x
            self.g_candidates.append(str(g1_expr))
            
            g2_expr = x - f
            self.g_candidates.append(str(g2_expr))

            g3_expr = 2/x
            self.g_candidates.append(str(g3_expr))

            g4_expr = x - (x**2 - 2)/(2*x) 
            self.g_candidates.append(str(g4_expr))

        except Exception:
            pass 

        g5_expr = x + f
        self.g_candidates.append(str(g5_expr))

        self.g_candidates = sorted(list(set(self.g_candidates)))
        
        if not self.g_candidates and not initial_load:
            self.result_text.config(state=tk.NORMAL)
            self.result_text.insert(tk.END, "Could not generate any g(x) candidates.\n", "error")
            self.result_text.config(state=tk.DISABLED)
        
        self._plot_g_candidates(f) 

    def _plot_g_candidates(self, f_sympified):
        try:
            x0_val = float(self.x0_var.get())
        except ValueError:
            x0_val = 0.5 

        xmin = x0_val - 1.5
        xmax = x0_val + 1.5
        x_vals = np.linspace(max(-10, xmin), min(10, xmax), 400)
        
        if self.fig_canvas:
            self.fig_canvas.get_tk_widget().destroy()
            
        fig, ax = plt.subplots(figsize=(6,4))
        
        f_str = str(f_sympified)
        f_str_safe = f_str.replace('**Rational(1, 2)', '**0.5')
        try:
            f_func = eval(f"lambda x: {f_str_safe}", {"__builtins__": None}, SAFE_FUNCTIONS)
            f_y_vals = np.array([f_func(xv) for xv in x_vals])
            f_y_vals[np.abs(f_y_vals) > 50] = np.nan
            ax.plot(x_vals, f_y_vals, 'b-', linewidth=2, label=f"f(x) = {f_str} (Root at f(x)=0)")
        except Exception:
             ax.axhline(0, color='b', linestyle=':', label='f(x)=0 axis')


        first_g_func_ref = None
        for g_str in self.g_candidates:
            try:
                g_str_safe = g_str.replace('**Rational(1, 2)', '**0.5') 
                g_func = eval(f"lambda x: {g_str_safe}", {"__builtins__": None}, SAFE_FUNCTIONS)
                
                if np.isnan(g_func(x0_val)) or np.isinf(g_func(x0_val)):
                    continue
                
                y_vals = np.array([g_func(xv) for xv in x_vals])
                y_vals[np.abs(y_vals) > 50] = np.nan 

                ax.plot(x_vals, y_vals, '--', alpha=0.7, label=f"g(x) = {g_str}")
                
                if first_g_func_ref is None:
                    first_g_func_ref = g_func

            except Exception:
                continue

        ax.plot(x_vals, x_vals, 'k:', linewidth=1.5, label="y=x (Fixed Point)")
        
        if first_g_func_ref:
            try:
                ax.plot(x0_val, 0, 'b^', markersize=8, label=f"x₀ on f(x) axis")
                ax.plot(x0_val, first_g_func_ref(x0_val), 'ro', markersize=8, label=f"g(x₀) point")
            except Exception:
                pass 

        ax.legend(fontsize=7, loc='upper left')
        ax.set_xlabel("x")
        ax.set_ylabel("f(x) or g(x)")
        ax.set_title("Function and Fixed-Point Candidates")
        ax.grid(True, linestyle=':', alpha=0.7)
        
        y_min, y_max = ax.get_ylim()
        y_range = max(abs(y_min), abs(y_max))
        ax.set_ylim([-y_range, y_range])

        self.fig_canvas = FigureCanvasTkAgg(fig, master=self.canvas_container)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(side=tk.TOP, fill="both", expand=True)
        
        self.candidate_list.delete(0, tk.END)
        for i, g in enumerate(self.g_candidates):
            self.candidate_list.insert(tk.END, f"g{i+1}: {g}")


    def _numerical_derivative(self, f, x, h=1e-5):
        try:
            return (f(x + h) - f(x - h)) / (2 * h)
        except Exception:
            return float('inf') 

    def auto_pick_best_g(self, x0):
        best_candidate = None
        best_derivative = float('inf')
        
        try:
            x0 = float(x0)
        except ValueError:
            return None

        for g_str in self.g_candidates:
            try:
                g_str_safe = g_str.replace('**Rational(1, 2)', '**0.5') 
                g_func = eval(f"lambda x: {g_str_safe}", {"__builtins__": None}, SAFE_FUNCTIONS)
                
                derivative_val = abs(self._numerical_derivative(g_func, x0))
                
                if derivative_val < 1: 
                    if derivative_val < best_derivative:
                        best_derivative = derivative_val
                        best_candidate = g_str
            except Exception:
                continue
        
        return best_candidate

    def solve_fixed_point(self, g_str, x0, tol, max_iter, ignore_tol):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        
        try:
            x0 = float(x0)
            tol = float(tol)
            max_iter = int(max_iter)
        except ValueError as e:
            self.result_text.insert(tk.END, f"Error: Invalid numeric inputs ({e}).", "error")
            self.result_text.config(state=tk.DISABLED)
            return
            
        self.result_text.insert(tk.END, f"FIXED-POINT ITERATION RESULTS\n", "header")
        self.result_text.insert(tk.END, f"Using g(x) = {g_str}\n\n", "data")
        
        header_line = f"{'Iter':<{COL_WIDTH_ITER}}{'x_i':<{COL_WIDTH_X}}{'x_{i+1}':<{COL_WIDTH_X}}{'Rel Error':<{COL_WIDTH_ERR}}\n"
        self.result_text.insert(tk.END, header_line, "table_header")
        self.result_text.insert(tk.END, "-" * TOTAL_WIDTH + "\n", "table_header")
        
        p0 = x0
        found_solution = False
        
        try:
            g_str_safe = g_str.replace('**Rational(1, 2)', '**0.5') 
            g_func = eval(f"lambda x: {g_str_safe}", {"__builtins__": None}, SAFE_FUNCTIONS)
        except Exception as e:
            self.result_text.insert(tk.END, f"Error: Invalid g(x) function for execution ({e}).\n", "error")
            self.result_text.config(state=tk.DISABLED)
            return
            
        p = p0
        for i in range(1, max_iter + 1):
            p_old = p0
            try:
                p = g_func(p_old)
            except Exception:
                self.result_text.insert(tk.END, f"\nIteration {i}: Domain/Math error. Check input range.\n", "error")
                break
                
            error = abs(p - p_old) / abs(p) if p != 0 else abs(p - p_old)
            
            x_old_formatted = f"{p_old:<{COL_WIDTH_X}.{X_PRECISION}f}"
            x_new_formatted = f"{p:<{COL_WIDTH_X}.{X_PRECISION}f}"
            
            error_formatted = f"{error:<{COL_WIDTH_ERR}.{ERR_PRECISION}e}"
            
            data_line = f"{i:<{COL_WIDTH_ITER}}{x_old_formatted}{x_new_formatted}{error_formatted}\n"
            self.result_text.insert(tk.END, data_line, "data")
            
            if not ignore_tol and error < tol:
                self.result_text.insert(tk.END, "\n" + "=" * TOTAL_WIDTH + "\n", "success")
                self.result_text.insert(tk.END, f"SUCCESS! Fixed point found: x ≈ {p:.{X_PRECISION}f}\n", "success")
                self.result_text.insert(tk.END, f"Stopping relative error satisfied: {error:.{ERR_PRECISION}e} in {i} iterations.\n", "success")
                self.result_text.insert(tk.END, "=" * TOTAL_WIDTH + "\n", "success")
                found_solution = True
                break
                
            p0 = p
            
        if not found_solution:
            self.result_text.insert(tk.END, "\n" + "=" * TOTAL_WIDTH + "\n", "error")
            self.result_text.insert(tk.END, f"FAILURE! Did not converge in {max_iter} iterations.\n", "error")
            self.result_text.insert(tk.END, f"Last approximation: x ≈ {p0:.{X_PRECISION}f}\n", "error")
            self.result_text.insert(tk.END, "=" * TOTAL_WIDTH + "\n", "error")
            
        self.result_text.config(state=tk.DISABLED)

    def run_iteration(self):
        if not self.g_candidates:
             self.generate_g_candidates()

        selected_g_str = None

        if self.auto_pick_var.get():
            selected_g_str = self.auto_pick_best_g(self.x0_var.get())
            if not selected_g_str:
                messagebox.showerror("Error", "No suitable g(x) found for auto-pick (check |g'(x₀)| < 1). Try a different x₀ or pick one manually.")
                return
        else:
            selection = self.candidate_list.curselection()
            if not selection:
                messagebox.showerror("Error", "Please select a g(x) function from the list or enable Auto-Pick.")
                return
            
            list_item = self.candidate_list.get(selection[0])
            try:
                selected_g_str = list_item.split(': ', 1)[1] 
            except IndexError:
                 selected_g_str = list_item 

        self.solve_fixed_point(
            selected_g_str, 
            self.x0_var.get(), 
            self.tol_var.get(), 
            self.max_iter_var.get(), 
            self.ignore_tol_var.get()
        )

def main():
    try:
        FixedPointSolverGUI()
    except Exception as e:
        if "main" not in str(e):
             messagebox.showerror("Fatal Error", f"An unhandled error occurred: {type(e).__name__}: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()