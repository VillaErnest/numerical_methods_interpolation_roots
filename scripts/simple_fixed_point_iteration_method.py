import tkinter as tk
from tkinter import messagebox
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import sys
from io import BytesIO
from PIL import Image, ImageTk
import math

# ----------------- Config / UI Colors (unchanged) -----------------
COLOR_BG_MAIN = "#f8fafc"
COLOR_BG_CARD = "#ffffff"
COLOR_ACCENT = "#f97316"
COLOR_ACCENT_HOVER = "#ea580c"
COLOR_TEXT_DARK = "#1f2937"
COLOR_TEXT_MUTED = "#6b7280"
COLOR_BORDER = "#e5e7eb"
COLOR_SUCCESS = "#10b981"
COLOR_INFO = "#3b82f6"

# Safe functions for eval
SAFE_FUNCTIONS = {name: getattr(np, name) for name in dir(np) if not name.startswith("_")}
SAFE_FUNCTIONS.update({"pi": np.pi, "e": np.e})

# Formatting constants (unchanged)
X_PRECISION = 10
ERR_PRECISION = 3
COL_WIDTH_ITER = 6
COL_WIDTH_X = X_PRECISION + 10
COL_WIDTH_ERR = 14
TOTAL_WIDTH = COL_WIDTH_ITER + COL_WIDTH_X * 2 + COL_WIDTH_ERR

# Cobweb animation delay (seconds)
COBWEB_STEP_DELAY = 0.5  # seconds per iteration step

# ----------------- Computation Core -----------------
class FixedPointSolverCore:
    """
    Pure computational core for:
     - generating g(x) candidates from f(x)=0 (using sympy)
     - picking best g(x) using numerical derivative at x0
     - running fixed-point iterations numerically
    """

    def __init__(self):
        self.g_candidates = []

    def generate_g_candidates(self, equation_str, var_str="x"):
        """Generate g(x) candidates by manipulating f(x)=0. Returns list[str] of python-style expressions."""
        self.g_candidates = []
        x = sp.symbols(var_str)
        try:
            f = sp.sympify(equation_str)
        except Exception as e:
            raise ValueError(f"Invalid equation: {e}")

        # 1) try solving for x
        try:
            solutions = sp.solve(f, x)
            for sol in solutions:
                # Accept expressions that contain x (iterative forms)
                try:
                    sol_simplified = sp.simplify(sol)
                    if sol_simplified.has(x):
                        self.g_candidates.append(str(sol_simplified))
                except Exception:
                    continue
        except Exception:
            pass

        # 2) x = x - f(x)
        try:
            g2 = sp.simplify(x - f)
            self.g_candidates.append(str(g2))
        except Exception:
            pass

        # 3) x = x + f(x)
        try:
            g3 = sp.simplify(x + f)
            self.g_candidates.append(str(g3))
        except Exception:
            pass

        # 4) polynomial rearrangements (quadratic / cubic attempts)
        try:
            if f.is_polynomial(x):
                degree = sp.degree(f, x)
                coeffs = sp.Poly(f, x).all_coeffs()
                if degree == 2 and len(coeffs) == 3:
                    a, b, c = coeffs
                    if a != 0:
                        g_quad = sp.sqrt((-b * x - c) / a)
                        self.g_candidates.append(str(sp.simplify(g_quad)))
                        # alternate rational rearrangement
                        if x != 0:
                            g_quad2 = sp.simplify((-b * x - c) / (a * x))
                            self.g_candidates.append(str(g_quad2))
                elif degree >= 3:
                    leading = coeffs[0]
                    remaining = sum(c * x ** (degree - i - 1) for i, c in enumerate(coeffs[1:], 1))
                    g_cubic = sp.simplify((-remaining / leading) ** (sp.Rational(1, degree)))
                    self.g_candidates.append(str(g_cubic))
        except Exception:
            pass

        # 5) exponential isolation
        try:
            if f.has(sp.exp):
                sols = sp.solve(f, sp.exp(x))
                for s in sols:
                    try:
                        log_form = sp.log(s)
                        if not log_form.has(sp.I):
                            self.g_candidates.append(str(sp.simplify(log_form)))
                    except Exception:
                        continue
        except Exception:
            pass

        # 6) trig extraction
        try:
            if any(f.has(fn) for fn in [sp.cos, sp.sin, sp.tan]):
                for fn in [sp.cos, sp.sin, sp.tan]:
                    trig_terms = [arg for arg in f.atoms(fn)]
                    if trig_terms:
                        self.g_candidates.append(str(trig_terms[0]))
        except Exception:
            pass

        # Deduplicate preserving order
        seen = set()
        unique = []
        for s in self.g_candidates:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        self.g_candidates = unique
        return self.g_candidates

    def _make_safe_callable(self, expr_str):
        """Return a python-callable lambda x: ... from expr_str using SAFE_FUNCTIONS."""
        # Replace sympy style to numpy where reasonable (basic)
        replacements = {
            'sqrt(': 'np.sqrt(',
            'exp(': 'np.exp(',
            'log(': 'np.log(',
            'sin(': 'np.sin(',
            'cos(': 'np.cos(',
            'tan(': 'np.tan(',
        }
        safe_expr = expr_str
        for k, v in replacements.items():
            safe_expr = safe_expr.replace(k, v)
        # Build callable
        try:
            func = eval(f"lambda x: {safe_expr}", {"__builtins__": None, "np": np}, SAFE_FUNCTIONS)
            # Test small value
            _ = func(0.1)
            return func
        except Exception as e:
            # raise or return None to indicate failure
            return None

    def _numerical_derivative(self, func, x, h=1e-6):
        try:
            return (func(x + h) - func(x - h)) / (2 * h)
        except Exception:
            return float('inf')

    def auto_pick_best_g(self, g_candidates, x0):
        """Pick candidate with smallest |g'(x0)| < 1. Returns the string or None."""
        try:
            x0_v = float(x0)
        except Exception:
            return None
        best = None
        best_val = float('inf')
        for g in g_candidates:
            func = self._make_safe_callable(g)
            if func is None:
                continue
            deriv = abs(self._numerical_derivative(func, x0_v))
            if deriv < 1 and deriv < best_val:
                best = g
                best_val = deriv
        return best

    def solve_fixed_point(self, g_str, x0, tol, max_iter, ignore_tol=False):
        """
        Execute iteration numerically and return:
          (iterations_list, last_approx, converged)
        iterations_list: list of tuples (i, x_old, x_new, rel_error)
        """
        # parse inputs
        try:
            x0_v = float(x0)
            tol_v = float(tol)
            max_it = int(max_iter)
        except Exception as e:
            raise ValueError(f"Invalid numeric input: {e}")

        func = self._make_safe_callable(g_str)
        if func is None:
            raise ValueError("g(x) cannot be evaluated (invalid expression)")

        iterations = []
        p = x0_v
        converged = False
        for i in range(1, max_it + 1):
            try:
                p_new = func(p)
                if np.isnan(p_new) or np.isinf(p_new):
                    raise ValueError("NaN or Inf encountered")
            except Exception as e:
                raise ValueError(f"Math error during iteration {i}: {e}")

            err = abs(p_new - p) / abs(p_new) if p_new != 0 else abs(p_new - p)
            iterations.append((i, p, p_new, err))
            if not ignore_tol and err < tol_v:
                converged = True
                return iterations, p_new, True
            p = p_new

        return iterations, p, False

# ----------------- GUI (keeps structure & layout unchanged) -----------------
class FixedPointSolverGUI:
    def __init__(self):
        self.core = FixedPointSolverCore()

        self.root = tk.Tk()
        self.root.title("Enhanced Fixed-Point Iteration Solver")
        self.root.geometry("1200x850")
        self.root.configure(bg=COLOR_BG_MAIN)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # retained variables/widgets (unchanged names where possible)
        self.g_candidates = []
        self.fig_canvas = None
        self.anim = None

        self._create_header()

        self.main_content = tk.Frame(self.root, bg=COLOR_BG_MAIN)
        self.main_content.pack(fill="both", expand=True, padx=16, pady=16)

        self._create_input_area()
        self._create_visualization_and_output_area()

        # generate initial candidates (silently)
        try:
            self.generate_g_candidates(initial_load=True)
        except Exception:
            pass

        self.root.mainloop()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to exit the application?"):
            self.root.destroy()
            sys.exit(0)

    def _create_header(self):
        header = tk.Frame(self.root, bg=COLOR_ACCENT, height=120)
        header.pack(fill="x")
        tk.Label(header, text="Fixed-Point Iteration Solver", font=("Segoe UI", 24, "bold"),
                 bg=COLOR_ACCENT, fg="white").pack(pady=(15, 0))
        tk.Label(header, text="Open Method", font=("Segoe UI", 14), bg=COLOR_ACCENT, fg="white").pack(pady=(0, 15))

    def _create_input_area(self):
        self.input_card = tk.Frame(self.main_content, bg=COLOR_BG_CARD, bd=1, relief="flat",
                                   highlightbackground=COLOR_BORDER, highlightthickness=1)
        self.input_card.pack(fill="x", pady=(0, 10))

        self.input_inner = tk.Frame(self.input_card, bg=COLOR_BG_CARD, padx=15, pady=15)
        self.input_inner.pack(fill="x")

        self.left_col = tk.Frame(self.input_inner, bg=COLOR_BG_CARD)
        self.left_col.pack(side="left", padx=10, pady=5, fill="x", expand=True)

        # input variables
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
        tk.Checkbutton(options_frame, text="Ignore Stopping Error", variable=self.ignore_tol_var, bg=COLOR_BG_CARD,
                       font=("Segoe UI", 10)).pack(side="left", padx=5)
        tk.Checkbutton(options_frame, text="Auto-Pick Best g(x)", variable=self.auto_pick_var, bg=COLOR_BG_CARD,
                       font=("Segoe UI", 10)).pack(side="left", padx=5)

        self.right_col = tk.Frame(self.input_inner, bg=COLOR_BG_CARD)
        self.right_col.pack(side="right", padx=10, pady=5, fill="both", expand=True)

        tk.Label(self.right_col, text="Generated g(x) Candidates (Pick one if Auto-Pick is OFF):",
                 font=("Segoe UI", 10, "bold"), fg=COLOR_TEXT_DARK, bg=COLOR_BG_CARD).pack(anchor="w", pady=(0, 5))

        list_frame = tk.Frame(self.right_col, bg=COLOR_BORDER)
        list_frame.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(list_frame, orient="vertical")
        self.candidate_list = tk.Listbox(list_frame, height=6, font=("Courier New", 10), bd=0, yscrollcommand=scrollbar.set,
                                         selectmode=tk.SINGLE)
        scrollbar.config(command=self.candidate_list.yview)

        scrollbar.pack(side="right", fill="y")
        self.candidate_list.pack(side="left", fill="both", expand=True)

        button_frame = tk.Frame(self.right_col, bg=COLOR_BG_CARD)
        button_frame.pack(fill="x", pady=10)

        tk.Button(button_frame, text="Generate g(x) 🔄", command=self.generate_g_candidates, bg=COLOR_ACCENT, fg="white",
                  font=("Segoe UI", 10, "bold"), activebackground=COLOR_ACCENT_HOVER).pack(side="left", fill="x", expand=True, padx=(0, 5))
        tk.Button(button_frame, text="Start Iteration ▶️", command=self.run_iteration, bg=COLOR_SUCCESS, fg="white",
                  font=("Segoe UI", 10, "bold"), activebackground="#059669").pack(side="right", fill="x", expand=True, padx=(5, 0))

    def _create_input_field(self, parent_frame, label_text, row, variable, default_value, tooltip):
        variable.set(default_value)

        label = tk.Label(parent_frame, text=label_text, font=("Segoe UI", 10), fg=COLOR_TEXT_DARK, bg=COLOR_BG_CARD)
        label.grid(row=row, column=0, sticky="w", pady=5, padx=(0, 10))

        entry = tk.Entry(parent_frame, textvariable=variable, font=("Segoe UI", 10), bd=1, relief="flat",
                         highlightbackground=COLOR_BORDER, highlightthickness=1, width=25)
        entry.grid(row=row, column=1, sticky="w", pady=5)

    def _create_visualization_and_output_area(self):
        self.content_panel = tk.Frame(self.main_content, bg=COLOR_BG_MAIN)
        self.content_panel.pack(fill="both", expand=True)

        # Left: plotting area
        self.plot_frame = tk.Frame(self.content_panel, bg=COLOR_BG_CARD, bd=1, highlightbackground=COLOR_BORDER, highlightthickness=1)
        self.plot_frame.pack(side="left", fill="both", expand=True, padx=(0, 8))

        tk.Label(self.plot_frame, text="Function Plot (f(x) and g(x) vs. y=x)", font=("Segoe UI", 12, "bold"),
                 bg=COLOR_BG_CARD, fg=COLOR_ACCENT).pack(pady=5)

        # Right: output / results area
        self.output_frame = tk.Frame(self.content_panel, bg=COLOR_BG_CARD, bd=1, highlightbackground=COLOR_BORDER, highlightthickness=1)
        self.output_frame.pack(side="right", fill="both", expand=True, padx=(8, 0))

        # Keep the label but show plain text (no LaTeX)
        self.latex_display = tk.Label(self.output_frame, bg=COLOR_BG_CARD, text="", justify="left", anchor="w", font=("Segoe UI", 10))
        self.latex_display.pack(pady=(5, 0), fill="x", padx=8)

        tk.Label(self.output_frame, text="Iteration Results", font=("Segoe UI", 12, "bold"), bg=COLOR_BG_CARD, fg=COLOR_ACCENT).pack(pady=(5, 5))

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

        # Canvas container inside plot_frame
        self.canvas_container = tk.Frame(self.plot_frame, bg=COLOR_BG_CARD)
        self.canvas_container.pack(fill="both", expand=True)

    # ----------------- Plot helpers -----------------
    def _clear_plot_canvas(self):
        if self.fig_canvas:
            try:
                self.fig_canvas.get_tk_widget().destroy()
            except Exception:
                pass
            self.fig_canvas = None

    def _plot_all_g_candidates(self, f_sympy):
        """Plot all generated g(x) curves and y=x (and faint f(x) for reference)."""
        try:
            x0_val = float(self.x0_var.get())
        except Exception:
            x0_val = 0.5

        xmin = x0_val - 1.5
        xmax = x0_val + 1.5
        x_vals = np.linspace(max(-10, xmin), min(10, xmax), 400)

        self._clear_plot_canvas()
        fig, ax = plt.subplots(figsize=(6, 4))

        # f(x) (faint reference)
        try:
            f_str = str(sp.sympify(self.eq_var.get()))
            f_str_safe = f_str.replace('**', '**').replace('sqrt', 'np.sqrt').replace('exp', 'np.exp').replace('log', 'np.log').replace('sin', 'np.sin').replace('cos', 'np.cos').replace('tan', 'np.tan')
            f_func = eval(f"lambda x: {f_str_safe}", {"__builtins__": None, "np": np}, SAFE_FUNCTIONS)
            f_y = np.array([f_func(xx) if not (np.isnan(xx) or np.isinf(xx)) else np.nan for xx in x_vals])
            f_y[np.abs(f_y) > 50] = np.nan
            ax.plot(x_vals, f_y, color='lightgray', linestyle='-', linewidth=1, label="f(x) (reference)")
        except Exception:
            ax.axhline(0, color='lightgray', linestyle=':', label='f(x) (ref)')

        # Plot each g candidate
        colors = ['red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive']
        for i, g in enumerate(self.g_candidates[:8]):
            try:
                g_safe = g.replace('sqrt', 'np.sqrt').replace('exp', 'np.exp').replace('log', 'np.log').replace('sin', 'np.sin').replace('cos', 'np.cos').replace('tan', 'np.tan')
                g_func = eval(f"lambda x: {g_safe}", {"__builtins__": None, "np": np}, SAFE_FUNCTIONS)
                y_vals = np.array([g_func(xx) if not (np.isnan(xx) or np.isinf(xx)) else np.nan for xx in x_vals])
                y_vals[np.abs(y_vals) > 50] = np.nan
                ax.plot(x_vals, y_vals, '--', alpha=0.8, color=colors[i % len(colors)], label=f"g{i+1}(x)")
            except Exception:
                continue

        # y = x
        ax.plot(x_vals, x_vals, 'k:', linewidth=1.5, label='y = x')

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("g(x) candidates and y = x")
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(fontsize=8, loc='best')

        self.fig_canvas = FigureCanvasTkAgg(fig, master=self.canvas_container)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(side=tk.TOP, fill="both", expand=True)

    def _plot_selected_and_animate(self, f_sympy, g_str, iterations):
        """
        Plot f(x), selected g(x), y=x and animate cobweb from initial x through the iterations.
        iterations: list of tuples (i, x_old, x_new, err)
        """
        try:
            x0_val = float(self.x0_var.get())
        except Exception:
            x0_val = 0.5

        # range: center around x0 and iteration points
        xs = [x0_val] + [it[1] for it in iterations] + [it[2] for it in iterations]
        finite_xs = [x for x in xs if (not math.isnan(x) and not math.isinf(x))]
        if finite_xs:
            xmin = min(finite_xs) - 1.0
            xmax = max(finite_xs) + 1.0
        else:
            xmin = x0_val - 1.5
            xmax = x0_val + 1.5

        x_vals = np.linspace(max(-10, xmin), min(10, xmax), 600)

        self._clear_plot_canvas()
        fig, ax = plt.subplots(figsize=(6, 4))

        # f(x)
        try:
            f_str = str(sp.sympify(self.eq_var.get()))
            f_str_safe = f_str.replace('sqrt', 'np.sqrt').replace('exp', 'np.exp').replace('log', 'np.log').replace('sin', 'np.sin').replace('cos', 'np.cos').replace('tan', 'np.tan')
            f_func = eval(f"lambda x: {f_str_safe}", {"__builtins__": None, "np": np}, SAFE_FUNCTIONS)
            f_y = np.array([f_func(xx) if not (np.isnan(xx) or np.isinf(xx)) else np.nan for xx in x_vals])
            f_y[np.abs(f_y) > 50] = np.nan
            ax.plot(x_vals, f_y, color='lightgray', linestyle='-', linewidth=1, label="f(x) (reference)")
        except Exception:
            ax.axhline(0, color='lightgray', linestyle=':', label='f(x) (ref)')

        # selected g(x)
        try:
            g_safe = g_str.replace('sqrt', 'np.sqrt').replace('exp', 'np.exp').replace('log', 'np.log').replace('sin', 'np.sin').replace('cos', 'np.cos').replace('tan', 'np.tan')
            g_func = eval(f"lambda x: {g_safe}", {"__builtins__": None, "np": np}, SAFE_FUNCTIONS)
            g_y = np.array([g_func(xx) if not (np.isnan(xx) or np.isinf(xx)) else np.nan for xx in x_vals])
            g_y[np.abs(g_y) > 50] = np.nan
            g_line, = ax.plot(x_vals, g_y, '--', linewidth=1.8, label="selected g(x)")
        except Exception:
            g_line = None

        # y = x
        identity_line, = ax.plot(x_vals, x_vals, 'k:', linewidth=1.5, label='y = x')

        # prepare cobweb line object (step segments)
        cobweb_line, = ax.plot([], [], '-', linewidth=1.5, color='blue')

        ax.set_xlim([max(-10, xmin), min(10, xmax)])
        # set y limits similar to x-range for clarity
        y_min = min(ax.get_xlim())
        y_max = max(ax.get_xlim())
        ax.set_ylim([y_min, y_max])

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Cobweb (iteration) visualization")
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(fontsize=8, loc='best')

        # Build the cobweb points as a sequence of coordinates to draw step-by-step
        # For each iteration (x_n -> g(x_n) -> x_{n+1}) we make two segments:
        #   vertical: (x_n, x_n) -> (x_n, g(x_n))
        #   horizontal: (x_n, g(x_n)) -> (g(x_n), g(x_n))
        cobweb_coords_x = []
        cobweb_coords_y = []
        # start at x0
        if iterations:
            # include initial point on y=x
            first_x = iterations[0][1]
            cobweb_coords_x.append(first_x)
            cobweb_coords_y.append(first_x)
            for (_, x_old, x_new, _) in iterations:
                # vertical to g(x_old)
                cobweb_coords_x.append(x_old)
                cobweb_coords_y.append(x_new)
                # horizontal to (x_new, x_new)
                cobweb_coords_x.append(x_new)
                cobweb_coords_y.append(x_new)

        # Convert COBWEB_STEP_DELAY to interval (ms) per frame.
        # We'll show one coordinate per frame (so 2 coords per iteration -> fit)
        interval_ms = int(COBWEB_STEP_DELAY * 1000)

        # Animation update function
        def init():
            cobweb_line.set_data([], [])
            return cobweb_line,

        def update(frame):
            # frame is index into cobweb_coords
            xs = cobweb_coords_x[:frame + 1]
            ys = cobweb_coords_y[:frame + 1]
            cobweb_line.set_data(xs, ys)
            return cobweb_line,

        # If no steps, simply show static plot
        if not cobweb_coords_x:
            self.fig_canvas = FigureCanvasTkAgg(fig, master=self.canvas_container)
            self.fig_canvas.draw()
            self.fig_canvas.get_tk_widget().pack(side=tk.TOP, fill="both", expand=True)
            return

        # Create animation: frames = number of cobweb points
        frames = len(cobweb_coords_x)
        # Use blit for better performance where possible
        try:
            anim = FuncAnimation(fig, update, frames=frames, init_func=init, interval=interval_ms, blit=True, repeat=False)
        except Exception:
            anim = FuncAnimation(fig, update, frames=frames, init_func=init, interval=interval_ms, repeat=False)

        # Store anim to prevent garbage collection
        self.anim = anim

        # Embed in Tk
        self.fig_canvas = FigureCanvasTkAgg(fig, master=self.canvas_container)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(side=tk.TOP, fill="both", expand=True)

        # Note: FuncAnimation will run in the background and will update the canvas.
        # To ensure updates appear in Tkinter, we'll use the fig_canvas.draw() after each interval.
        # Matplotlib's animation typically runs its own timer—embedding with TkAgg will make it visible.

    # ----------------- Core GUI actions -----------------
    def generate_g_candidates(self, equation_str=None, var_str="x", initial_load=False):
        """Generate candidates using the core and plot them (unchanged GUI)."""
        if equation_str is None:
            equation_str = self.eq_var.get()

        try:
            candidates = self.core.generate_g_candidates(equation_str, var_str=var_str)
            self.g_candidates = candidates
        except Exception as e:
            if not initial_load:
                messagebox.showerror("Error", f"Invalid equation: {e}")
            self.g_candidates = []
            return

        # Update listbox (show plain text)
        self.candidate_list.delete(0, tk.END)
        for i, g in enumerate(self.g_candidates):
            display_text = f"g{i+1}: {g}"
            self.candidate_list.insert(tk.END, display_text)

        # Plot all g candidates (and y=x + faint f(x))
        try:
            f_sym = sp.sympify(self.eq_var.get())
        except Exception:
            f_sym = None
        self._plot_all_g_candidates(f_sym)

        # Update text label showing f and count of g's (replacing LaTeX display)
        show_text = f"f(x) = {self.eq_var.get()}\nGenerated {len(self.g_candidates)} g(x) candidate(s)."
        self.latex_display.config(text=show_text)

    def solve_fixed_point(self, g_str, x0, tol, max_iter, ignore_tol):
        """Use core to execute iterations and update result_text (GUI display)."""
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)

        try:
            iterations, last_x, converged = self.core.solve_fixed_point(g_str, x0, tol, max_iter, ignore_tol)
        except Exception as e:
            self.result_text.insert(tk.END, f"Error: {e}\n", "error")
            self.result_text.config(state=tk.DISABLED)
            return

        # Display header and info (plain text)
        self.result_text.insert(tk.END, f"FIXED-POINT ITERATION RESULTS\n", "header")
        self.result_text.insert(tk.END, f"{'=' * TOTAL_WIDTH}\n\n", "header")

        # Show original equation (plain)
        try:
            f_expr = sp.sympify(self.eq_var.get())
            self.result_text.insert(tk.END, f"Original Equation: f(x) = 0\n", "data")
            self.result_text.insert(tk.END, f"  Python: {str(f_expr)} = 0\n\n", "data")
        except Exception:
            pass

        # Show the g used
        self.result_text.insert(tk.END, f"Using g(x) for iteration:\n", "data")
        self.result_text.insert(tk.END, f"  Python: g(x) = {g_str}\n\n", "data")

        self.result_text.insert(tk.END, f"Initial guess: x₀ = {x0}\n", "data")
        self.result_text.insert(tk.END, f"Stopping tolerance: ε = {tol}\n", "data")
        self.result_text.insert(tk.END, f"Max iterations: N = {max_iter}\n\n", "data")

        header_line = f"{'Iter':<{COL_WIDTH_ITER}}{'x_i':<{COL_WIDTH_X}}{'x_{i+1}':<{COL_WIDTH_X}}{'Rel Error':<{COL_WIDTH_ERR}}\n"
        self.result_text.insert(tk.END, header_line, "table_header")
        self.result_text.insert(tk.END, "-" * TOTAL_WIDTH + "\n", "table_header")

        for i, x_old, x_new, err in iterations:
            x_old_formatted = f"{x_old:<{COL_WIDTH_X}.{X_PRECISION}f}"
            x_new_formatted = f"{x_new:<{COL_WIDTH_X}.{X_PRECISION}f}"
            error_formatted = f"{err:<{COL_WIDTH_ERR}.{ERR_PRECISION}e}"
            data_line = f"{i:<{COL_WIDTH_ITER}}{x_old_formatted}{x_new_formatted}{error_formatted}\n"
            self.result_text.insert(tk.END, data_line, "data")

        if converged:
            self.result_text.insert(tk.END, "\n" + "=" * TOTAL_WIDTH + "\n", "success")
            self.result_text.insert(tk.END, f"SUCCESS! Fixed point found: x* ≈ {last_x:.{X_PRECISION}f}\n", "success")
            self.result_text.insert(tk.END, "=" * TOTAL_WIDTH + "\n", "success")
        else:
            self.result_text.insert(tk.END, "\n" + "=" * TOTAL_WIDTH + "\n", "error")
            self.result_text.insert(tk.END, f"FAILURE! Did not converge in {max_iter} iterations.\n", "error")
            self.result_text.insert(tk.END, f"Last approximation: x ≈ {last_x:.{X_PRECISION}f}\n", "error")
            self.result_text.insert(tk.END, "=" * TOTAL_WIDTH + "\n", "error")

        self.result_text.config(state=tk.DISABLED)
        return iterations, last_x, converged

    def run_iteration(self):
        # Ensure candidates exist
        if not self.g_candidates:
            self.generate_g_candidates()

        selected_g_str = None

        if self.auto_pick_var.get():
            selected_g_str = self.core.auto_pick_best_g(self.g_candidates, self.x0_var.get())
            if not selected_g_str:
                messagebox.showerror("Error", "No suitable g(x) found for auto-pick (check |g'(x₀)| < 1). Try a different x₀ or pick one manually.")
                return
        else:
            selection = self.candidate_list.curselection()
            if not selection:
                messagebox.showerror("Error", "Please select a g(x) function from the list or enable Auto-Pick.")
                return
            idx = selection[0]
            selected_g_str = self.g_candidates[idx]

        # Run the iteration (core) and show results
        try:
            iterations, last_x, converged = self.solve_fixed_point(
                selected_g_str,
                self.x0_var.get(),
                self.tol_var.get(),
                self.max_iter_var.get(),
                self.ignore_tol_var.get()
            )
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        # Update the top label text area with plain info (replace previous LaTeX use)
        self.latex_display.config(text=f"f(x) = {self.eq_var.get()}\ng(x) = {selected_g_str}")

        # Plot selected g and animate cobweb (and show f(x) for reference)
        try:
            f_sym = sp.sympify(self.eq_var.get())
        except Exception:
            f_sym = None

        # The animation will automatically play and finish frozen at last frame
        self._plot_selected_and_animate(f_sym, selected_g_str, iterations)

# ----------------- Main -----------------
def main():
    try:
        FixedPointSolverGUI()
    except Exception as e:
        # Only show fatal messagebox if GUI exists
        try:
            messagebox.showerror("Fatal Error", f"An unhandled error occurred: {type(e).__name__}: {e}")
        except Exception:
            print("Fatal Error:", e)
        sys.exit(1)

if __name__ == "__main__":
    main()
