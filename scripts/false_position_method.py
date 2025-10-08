import tkinter as tk
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import math
import numpy as np

# ------------------------------
# False Position Method
# ------------------------------
def false_position_method(f, x_lower, x_upper, tolerance, max_iterations):
    if f(x_lower) * f(x_upper) >= 0:
        raise ValueError("The function must have opposite signs at the interval endpoints.")
    
    iteration = 0
    x_root = x_upper
    error = 100.0  # start with large error %
    history = []

    while error > tolerance and iteration < max_iterations:
        x_root_new = x_upper - (f(x_upper) * (x_lower - x_upper)) / (f(x_lower) - f(x_upper))

        # Compute percent relative error
        if iteration > 0:
            error = abs((x_root_new - x_root) / x_root_new) * 100

        x_root = x_root_new
        f_root = f(x_root)

        if f(x_lower) * f_root < 0:
            x_upper = x_root
        else:
            x_lower = x_root

        history.append({
            'iteration': iteration,
            'x_lower': x_lower,
            'x_upper': x_upper,
            'x_root': x_root,
            'f_root': f_root,
            'error': error if iteration > 0 else float('inf')
        })
        iteration += 1
    
    return x_root, history

# ------------------------------
# Polynomial / Function Parser
# ------------------------------
def parse_polynomial(coefficients):
    def poly_func(x):
        result = 0
        for i, coef in enumerate(coefficients):
            power = len(coefficients) - i - 1
            result += coef * (x ** power)
        return result
    return poly_func

def parse_function(func_str):
    func_str = func_str.replace('^', '**')
    safe_dict = {
        'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
        'exp': math.exp, 'log': math.log, 'log10': math.log10,
        'sqrt': math.sqrt, 'pi': math.pi, 'e': math.e,
        'abs': abs
    }
    def f(x):
        return eval(func_str, {"__builtins__": None}, {**safe_dict, 'x': x})
    return f

# ------------------------------
# Tkinter GUI App
# ------------------------------
class FalsePositionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("False Position Method (Regula Falsi)")
        self.root.geometry("1000x750")

        # Input Type
        self.input_mode = tk.StringVar(value="function")
        mode_frame = ttk.LabelFrame(root, text="Input Type")
        mode_frame.pack(fill="x", padx=10, pady=5)
        ttk.Radiobutton(mode_frame, text="Enter Function f(x)", variable=self.input_mode, value="function").pack(side="left", padx=10)
        ttk.Radiobutton(mode_frame, text="Enter Polynomial Coefficients", variable=self.input_mode, value="polynomial").pack(side="left")

        # Function / Polynomial Input
        func_frame = ttk.LabelFrame(root, text="Function / Polynomial Input")
        func_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(func_frame, text="f(x) =").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.func_entry = ttk.Entry(func_frame, width=50)
        self.func_entry.insert(0, "x**3 - 2*x - 5")
        self.func_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(func_frame, text="Coefficients (comma separated):").grid(row=1, column=0, sticky="w", padx=5)
        self.poly_entry = ttk.Entry(func_frame, width=50)
        self.poly_entry.insert(0, "1, 0, -2, -5")
        self.poly_entry.grid(row=1, column=1, padx=5, pady=5)

        # Interval and Settings
        interval_frame = ttk.LabelFrame(root, text="Interval & Settings")
        interval_frame.pack(fill="x", padx=10, pady=5)
        ttk.Label(interval_frame, text="Lower Bound (a):").grid(row=0, column=0, padx=5, pady=5)
        self.lower_entry = ttk.Entry(interval_frame, width=10)
        self.lower_entry.insert(0, "2")
        self.lower_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Label(interval_frame, text="Upper Bound (b):").grid(row=0, column=2, padx=5, pady=5)
        self.upper_entry = ttk.Entry(interval_frame, width=10)
        self.upper_entry.insert(0, "3")
        self.upper_entry.grid(row=0, column=3, padx=5, pady=5)
        ttk.Label(interval_frame, text="Tolerance (%):").grid(row=1, column=0, padx=5, pady=5)
        self.tol_entry = ttk.Entry(interval_frame, width=10)
        self.tol_entry.insert(0, "0.01")
        self.tol_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Label(interval_frame, text="Max Iterations:").grid(row=1, column=2, padx=5, pady=5)
        self.iter_entry = ttk.Entry(interval_frame, width=10)
        self.iter_entry.insert(0, "50")
        self.iter_entry.grid(row=1, column=3, padx=5, pady=5)

        # Run Button
        ttk.Button(root, text="Run False Position Method", command=self.run_method).pack(pady=10)

        # Plot Frame
        self.plot_frame = ttk.LabelFrame(root, text="Convergence Plot")
        self.plot_frame.pack(fill="both", expand=True, padx=10, pady=5)

    # --------------------------
    # Run Method
    # --------------------------
    def run_method(self):
        try:
            if self.input_mode.get() == "function":
                func_str = self.func_entry.get()
                f = parse_function(func_str)
                display_func = f"f(x) = {func_str}"
            else:
                coef_str = self.poly_entry.get()
                coefficients = [float(x.strip()) for x in coef_str.split(',')]
                f = parse_polynomial(coefficients)
                display_func = "Polynomial with coefficients: " + str(coefficients)

            x_lower = float(self.lower_entry.get())
            x_upper = float(self.upper_entry.get())
            tol = float(self.tol_entry.get())
            max_iter = int(self.iter_entry.get())

            root_val, history = false_position_method(f, x_lower, x_upper, tol, max_iter)

            # Display iterations in popup
            self.display_results_popup(display_func, history)

            # Plot charts
            self.plot_convergence_tk(f, history, x_lower, x_upper)

        except Exception as e:
            messagebox.showerror("Error", str(e))

    # --------------------------
    # Display Iteration Results in Popup
    # --------------------------
    def display_results_popup(self, func, history):
        popup = tk.Toplevel(self.root)
        popup.title("Iteration Details")
        popup.geometry("750x400")

        text = tk.Text(popup, wrap="word")
        text.pack(fill="both", expand=True, padx=5, pady=5)

        text.insert(tk.END, f"{func}\n\n")
        text.insert(tk.END, f"{'Iter':<6}{'x_lower':<12}{'x_upper':<12}{'x_root':<12}{'f(x_root)':<12}{'Error (%)':<12}\n")
        text.insert(tk.END, "-"*75 + "\n")
        for h in history:
            error = h['error'] if h['iteration'] > 0 else float('inf')
            text.insert(tk.END, f"{h['iteration']:<6}{h['x_lower']:<12.6f}{h['x_upper']:<12.6f}{h['x_root']:<12.6f}{h['f_root']:<12.6f}{error:<12.6f}\n")
        final_root = history[-1]['x_root']
        text.insert(tk.END, f"\nFinal Root: {final_root:.10f}\n")

    # --------------------------
    # Plot Convergence
    # --------------------------
    def plot_convergence_tk(self, f, history, x_lower_init, x_upper_init):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
        x_min = min(x_lower_init, x_upper_init, min([h['x_root'] for h in history]))
        x_max = max(x_lower_init, x_upper_init, max([h['x_root'] for h in history]))
        x_vals = np.linspace(x_min - 0.5, x_max + 0.5, 400)
        y_vals = [f(x) for x in x_vals]

        ax1.plot(x_vals, y_vals, 'b-', label='f(x)')
        ax1.axhline(0, color='black', linestyle='--')
        roots = [h['x_root'] for h in history]
        ax1.plot(roots, [f(x) for x in roots], 'ro--')
        ax1.set_title("Root Approximation")
        ax1.set_xlabel("x")
        ax1.set_ylabel("f(x)")
        ax1.grid(True)
        ax1.legend()

        if len(history) > 1:
            iters = [h['iteration'] for h in history if h['iteration'] > 0]
            errors = [h['error'] for h in history if h['iteration'] > 0]
            ax2.semilogy(iters, errors, 'bo-')
            ax2.set_title("Error Convergence (%)")
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Percent Error (log scale)")
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, "Not enough iterations", ha="center")

        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

# ------------------------------
# Run App
# ------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = FalsePositionApp(root)
    root.mainloop()
