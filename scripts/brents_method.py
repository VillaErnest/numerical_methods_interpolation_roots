import tkinter as tk
from tkinter import messagebox, scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import re
import traceback

# Constants for the application
FONT = ('Inter', 10)
TITLE_FONT = ('Inter', 14, 'bold')
BG_COLOR = '#f0f4f8'
PRIMARY_COLOR = '#4a90e2'
TEXT_COLOR = '#333333'


class BrentMethodApp:
    """
    Tkinter GUI application for finding roots using Brent's method.
    """

    def __init__(self):
        self.master = tk.Tk()
        self.master.title("Brent's Method Root Finder")
        self.master.config(bg=BG_COLOR)
        self.master.geometry("1200x800")
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.history = []

        # --- Variables ---
        self.func_var = tk.StringVar(value='x**3 - 0.165*x**2 + 3.993e-4')
        self.a_var = tk.StringVar(value='0')
        self.b_var = tk.StringVar(value='0.11')
        self.tol_var = tk.StringVar(value='1e-6')
        self.max_iter_var = tk.StringVar(value='100')

        # --- Setup UI Components ---
        self.setup_input_frame()
        self.setup_result_frame()
        self.setup_plot_frame()

    def on_closing(self):
        """Handle window close event."""
        self.master.quit()
        self.master.destroy()

    def run(self):
        """Start the Tkinter main loop."""
        self.master.mainloop()

    def setup_input_frame(self):
        """Creates and organizes the input widgets."""
        input_frame = tk.Frame(self.master, padx=15, pady=15, bg=BG_COLOR)
        input_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Title
        tk.Label(input_frame, text="Brent's Method Parameters", font=TITLE_FONT, bg=BG_COLOR, fg=PRIMARY_COLOR).grid(
            row=0, column=0, columnspan=2, pady=(0, 15), sticky='w')

        # Labels and Entry fields
        fields = [
            ("Function f(x):", self.func_var, 1),
            ("Left End (a):", self.a_var, 2),
            ("Right End (b):", self.b_var, 3),
            ("Tolerance:", self.tol_var, 4),
            ("Max Iterations:", self.max_iter_var, 5)
        ]

        for text, var, row in fields:
            tk.Label(input_frame, text=text, font=FONT, bg=BG_COLOR, fg=TEXT_COLOR).grid(row=row, column=0, sticky='w',
                                                                                         pady=5)
            entry = tk.Entry(input_frame, textvariable=var, font=FONT, width=25, relief=tk.FLAT, bd=2)
            entry.grid(row=row, column=1, sticky='ew', padx=5, pady=5)

        # Calculate Button
        calc_button = tk.Button(input_frame, text="Calculate Root", command=self.calculate_root,
                                font=('Inter', 12, 'bold'), bg=PRIMARY_COLOR, fg='white',
                                activebackground='#5b9ade', activeforeground='white',
                                relief=tk.FLAT, bd=0, padx=10, pady=5)
        calc_button.grid(row=6, column=0, columnspan=2, pady=20, sticky='ew')

    def setup_result_frame(self):
        """Creates the text area for logging results and steps."""
        result_frame = tk.Frame(self.master, padx=10, pady=10, bg=BG_COLOR)
        result_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        tk.Label(result_frame, text="Calculation Log", font=('Inter', 12, 'bold'), bg=BG_COLOR, fg=TEXT_COLOR).pack(
            anchor='w', pady=(0, 5))

        self.log_text = scrolledtext.ScrolledText(result_frame, wrap=tk.WORD, font=('Courier', 9), height=15,
                                                  relief=tk.SUNKEN, bd=1, bg='white', fg=TEXT_COLOR)
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def setup_plot_frame(self):
        """Creates a frame to hold the Matplotlib plot."""
        self.plot_container = tk.Frame(self.master, bg='white', relief=tk.GROOVE, bd=1)
        self.plot_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Placeholder figure setup
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.ax.set_title("Brent's Method Plot")
        self.ax.grid(alpha=0.3)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_container)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Matplotlib toolbar
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_container)
        self.toolbar.update()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # --- Core Logic Functions ---

    def log(self, message):
        """Appends a message to the calculation log."""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)

    def parse_function(self, func_str):
        """Parse user input string into a callable function f(x) using NumPy."""
        func_str = func_str.replace('^', '**')

        # Insert multiplication where omitted (e.g., 2x, x(5), )x)
        func_str = re.sub(r'(\d)(x)', r'\1*\2', func_str)
        func_str = re.sub(r'(x)(\d)', r'\1*\2', func_str)
        func_str = re.sub(r'(x)(\()', r'\1*\2', func_str)
        func_str = re.sub(r'(\))(x)', r'\1*\2', func_str)

        # Allowed functions/constants from numpy
        allowed = {
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan, 'asin': np.arcsin, 'acos': np.arccos, 'atan': np.arctan,
            'exp': np.exp, 'log': np.log, 'ln': np.log, 'log10': np.log10, 'sqrt': np.sqrt,
            'pi': np.pi, 'e': np.e, 'abs': np.abs
        }

        try:
            # Safely evaluate the lambda function
            return eval(f"lambda x: {func_str}", {"__builtins__": None}, allowed)
        except Exception as e:
            raise ValueError(f"Invalid function string or syntax: {e}")

    def brent_method(self, f, a, b, tol, max_iter):
        """Brent's method implementation for root finding."""
        try:
            fa, fb = f(a), f(b)
        except Exception as e:
            return None, f"Error evaluating function at endpoints: {e}"
        
        # Check for NaN or infinite values
        if not (np.isfinite(fa) and np.isfinite(fb)):
            return None, "Function returns NaN or infinite values at endpoints"
        
        if fa * fb >= 0:
            return None, "f(a) and f(b) must have opposite signs. Try a different interval [a, b]."

        # Ensure |f(a)| >= |f(b)| for the initial state
        if abs(fa) < abs(fb): 
            a, b, fa, fb = b, a, fb, fa

        c, fc = a, fa
        d = 0
        mflag = True
        self.history = []

        self.log(f"|Iter|    a    |    b    |    s    |  f(s)  |  error ")
        self.log("-" * 60)

        for i in range(1, max_iter + 1):
            if abs(fb) < tol: 
                return b, "Exact root found"

            # Check if inverse quadratic interpolation (IQI) is possible
            if fa != fc and fb != fc:
                # IQI
                s = (a * fb * fc) / ((fa - fb) * (fa - fc)) \
                    + (b * fa * fc) / ((fb - fa) * (fb - fc)) \
                    + (c * fa * fb) / ((fc - fa) * (fc - fb))
            else:
                # Secant method - check for division by zero
                if abs(fb - fa) < 1e-15:
                    s = (a + b) / 2  # Fall back to bisection
                else:
                    s = b - fb * (b - a) / (fb - fa)

            # Check conditions for falling back to bisection
            tmp2 = (3 * a + b) / 4
            
            # Condition 1: s not between (3a+b)/4 and b
            if not ((min(tmp2, b) <= s <= max(tmp2, b)) or (min(b, tmp2) <= s <= max(b, tmp2))):
                cond1 = True
            else:
                cond1 = False
            
            # Condition 2: Check against previous steps
            if mflag:
                cond2 = abs(s - b) >= abs(b - c) / 2
            else:
                cond2 = abs(s - b) >= abs(c - d) / 2
            
            # Condition 3: Check if steps are too small
            if mflag:
                cond3 = abs(b - c) < tol
            else:
                cond3 = abs(c - d) < tol

            # Use bisection if any condition is true
            if cond1 or cond2 or cond3:
                s = (a + b) / 2
                mflag = True
            else:
                mflag = False

            try:
                fs = f(s)
            except Exception as e:
                return None, f"Error evaluating function at s={s}: {e}"
            
            if not np.isfinite(fs):
                return None, f"Function returns NaN or infinite value at s={s}"
            
            self.history.append({'iteration': i, 'a': a, 'b': b, 's': s, 'f(s)': fs, 'error': abs(b - a)})

            self.log(f"{i:4d}|{a:8.6f}|{b:8.6f}|{s:8.6f}|{fs:8.2e}|{abs(b - a):8.2e}")

            # Update d, c for the next iteration
            d = c
            c, fc = b, fb
            
            if fa * fs < 0:
                b, fb = s, fs
            else:
                a, fa = s, fs

            # Maintain the condition |f(a)| >= |f(b)|
            if abs(fa) < abs(fb): 
                a, b, fa, fb = b, a, fb, fa

            # Check for convergence
            if abs(b - a) < tol: 
                return b, "Converged successfully within tolerance"

        return b, "Max iterations reached"

    def plot_results(self, f, a_init, b_init, root):
        """Plots the function and the convergence path in the Tkinter window."""
        self.ax.clear()

        # Safely determine plot bounds.
        if a_init == b_init:
            plot_a = a_init - 0.1
            plot_b = b_init + 0.1
        else:
            # Use max/min of current bounds and root
            min_val = min(a_init, b_init, root)
            max_val = max(a_init, b_init, root)

            # Add a buffer for visualization
            range_val = max_val - min_val if max_val != min_val else 0.1

            plot_a = min_val - range_val * 0.1
            plot_b = max_val + range_val * 0.1

        x = np.linspace(plot_a, plot_b, 1000)

        # Check if f(x) evaluation fails on the array before plotting
        try:
            y = f(x)
        except Exception as e:
            self.log(f"❌ Plotting Error: Function evaluation failed on plot domain: {e}")
            self.ax.set_title("Plotting Failed (Check function definition)")
            self.canvas.draw()
            return

        # Plot the function
        self.ax.plot(x, y, color=PRIMARY_COLOR, label='f(x)')
        self.ax.axhline(0, color='k', ls='--', alpha=0.5, label='$y=0$')

        if self.history:
            s_vals = [d['s'] for d in self.history]
            f_s_vals = [d['f(s)'] for d in self.history]

            # Plot the root and iterations
            self.ax.plot(root, f(root), 'o', color='red', markersize=8, label=f'Root $\\approx$ {root:.6f}')
            self.ax.plot(s_vals, f_s_vals, 'x', color='orange', markersize=5, alpha=0.7, label="Iterations")

            # Connect iterations to the x-axis for visual aid
            for s, fs in zip(s_vals, f_s_vals):
                self.ax.plot([s, s], [0, fs], color='orange', linestyle=':', alpha=0.3)

        self.ax.set_title(f"Brent's Method Convergence (Root $\\approx$ {root:.6f})", fontsize=10)
        self.ax.set_xlabel('$x$', fontsize=10)
        self.ax.set_ylabel('$f(x)$', fontsize=10)
        self.ax.legend(loc='upper right', fontsize=8)
        self.ax.grid(alpha=0.3)

        # Redraw the canvas
        self.canvas.draw()

    def calculate_root(self):
        """Main method called when the Calculate button is pressed."""
        self.log_text.delete(1.0, tk.END)
        self.log("--- Starting Brent's Method Calculation ---")

        try:
            # 1. Parse and validate inputs
            func_str = self.func_var.get()
            f = self.parse_function(func_str)
            a = float(self.a_var.get())
            b = float(self.b_var.get())
            tol = float(self.tol_var.get())
            max_iter = int(self.max_iter_var.get())

            if a >= b:
                raise ValueError("Left endpoint 'a' must be less than right endpoint 'b'.")
            
            if tol <= 0:
                raise ValueError("Tolerance must be positive.")
            
            if max_iter <= 0:
                raise ValueError("Max iterations must be positive.")

            # 2. Run Brent's Method
            self.log(f"Function: f(x) = {func_str}")
            self.log(f"Interval: [{a}, {b}], Tolerance: {tol:.2e}, Max Iter: {max_iter}")

            root, msg = self.brent_method(f, a, b, tol, max_iter)

            # 3. Display Results and Plot
            if root is not None:
                final_f_root = f(root)
                self.log("-" * 60)
                self.log(f"| FINAL RESULT |")
                self.log(f"🎯 Root found at x ≈ {root:.10f}")
                self.log(f"f(root) ≈ {final_f_root:.2e}")
                self.log(f"Status: {msg}")
                self.log("-" * 60)
                self.plot_results(f, a, b, root)
            else:
                self.log("-" * 60)
                self.log(f"❌ Calculation Failed: {msg}")
                self.log("-" * 60)
                self.ax.clear()
                self.ax.set_title("Calculation Failed")
                self.canvas.draw()

        except ValueError as e:
            error_msg = f"Input Error: {e}"
            self.log(f"❌ {error_msg}")
            messagebox.showerror("Input Error", error_msg)
        except Exception as e:
            # Log the full traceback for better debugging
            full_trace = traceback.format_exc()
            error_msg = f"An unexpected error occurred: {e}"
            self.log("-" * 60)
            self.log(f"❌ {error_msg}")
            self.log("--- FULL TRACEBACK ---")
            self.log(full_trace)
            self.log("-" * 60)
            messagebox.showerror("Runtime Error", f"{error_msg}. Check Calculation Log for details.")


if __name__ == '__main__':
    # Initialize and run the application
    app = BrentMethodApp()
    app.run()