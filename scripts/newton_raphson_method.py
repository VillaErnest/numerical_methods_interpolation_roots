"""
NEWTON-RAPHSON METHOD (OPEN METHOD) - GUI VERSION
Modified: Now supports full symbolic math (e.g., 2x+3, ln(x), e^(2x))
"""

# =========================================================
# 1️⃣  IMPORTS & SETUP
# =========================================================
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Optional symbolic math support
try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False


# =========================================================
# 2️⃣  FUNCTION CREATION
# =========================================================
def create_function(func_str):
    """Create f(x) and f'(x) callables, symbolic if possible, else numeric."""
    if SYMPY_AVAILABLE:
        x = sp.symbols("x")
        f = sp.sympify(func_str)
        f_prime = sp.diff(f, x)
        f_l = sp.lambdify(x, f, "numpy")
        f_p = sp.lambdify(x, f_prime, "numpy")
    else:
        def f_l(x):
            # Add all numpy functions (sin, cos, exp, log, etc.)
            safe_dict = {"x": x, "__builtins__": {}}
            safe_dict.update(vars(np))
            return eval(func_str, safe_dict)

        def f_p(x):
            h = 1e-6
            return (f_l(x + h) - f_l(x - h)) / (2 * h)
    return f_l, f_p



# =========================================================
# 3️⃣  DERIVATIVE CALCULATION
# =========================================================
def compute_derivative(f, x, h=1e-6):
    """Compute numeric derivative using central difference."""
    return (f(x + h) - f(x - h)) / (2 * h)


# =========================================================
# 4️⃣  NEWTON–RAPHSON LOOP
# =========================================================
def newton_raphson(f, df, x0, tol=1e-6, max_iter=50):
    iterations = []
    x_curr = x0

    for i in range(1, max_iter + 1):
        f_val = f(x_curr)
        df_val = df(x_curr)

        if df_val == 0:
            raise ZeroDivisionError(f"Derivative is zero at iteration {i}")

        x_new = x_curr - f_val / df_val
        dx = abs(x_new - x_curr)
        iterations.append((i, x_curr, f_val, df_val, x_new, dx))

        if dx < tol:
            return x_new, iterations, True

        x_curr = x_new

    return x_curr, iterations, False


# =========================================================
# 5️⃣  MAIN PROGRAM & RESULTS (GUI IMPLEMENTATION)
# =========================================================
class NewtonRaphsonApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Newton–Raphson Method (Open Method)")
        self.geometry("1180x720")
        self.configure(bg="#f5f6fa")
        self.create_widgets()

    def create_widgets(self):
        frm = ttk.Frame(self, padding=15)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            frm,
            text="Newton–Raphson Method (Open Method)",
            font=("Segoe UI", 15, "bold"),
            foreground="#2b5797",
        ).grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")

        left_col = ttk.Frame(frm)
        left_col.grid(row=1, column=0, sticky="nsew", padx=8, pady=6)
        frm.columnconfigure(0, weight=1)
        frm.columnconfigure(1, weight=2)

        inp = ttk.LabelFrame(left_col, text=" User Inputs ", padding=12)
        inp.pack(fill=tk.X, anchor="n")

        ttk.Label(inp, text="Function f(x):").grid(row=0, column=0, sticky="w", pady=2)
        self.func_entry = ttk.Entry(inp, width=40, font=("Consolas", 10))
        self.func_entry.grid(row=0, column=1, columnspan=3, sticky="w", padx=6)
        self.func_entry.insert(0, "x**3 - x - 2")

        ttk.Label(inp, text="Initial guess x₀:").grid(row=1, column=0, sticky="w", pady=2)
        self.x0_entry = ttk.Entry(inp, width=12, font=("Consolas", 10))
        self.x0_entry.grid(row=1, column=1, sticky="w", padx=(0, 8))
        self.x0_entry.insert(0, "1.5")

        ttk.Label(inp, text="Tolerance:").grid(row=1, column=2, sticky="w", pady=2)
        self.tol_entry = ttk.Entry(inp, width=12, font=("Consolas", 10))
        self.tol_entry.grid(row=1, column=3, sticky="w")
        self.tol_entry.insert(0, "1e-6")

        ttk.Label(inp, text="Max Iterations:").grid(row=2, column=0, sticky="w", pady=2)
        self.maxit_entry = ttk.Entry(inp, width=12, font=("Consolas", 10))
        self.maxit_entry.grid(row=2, column=1, sticky="w")
        self.maxit_entry.insert(0, "50")

        btn_frame = ttk.Frame(inp)
        btn_frame.grid(row=3, column=0, columnspan=4, pady=(10, 0))
        ttk.Button(btn_frame, text="▶ Run Newton-Raphson", command=self.on_run).grid(row=0, column=0, padx=4)
        ttk.Button(btn_frame, text="🧹 Clear Table", command=self.clear_table).grid(row=0, column=1, padx=4)

        sym_msg = "✓ sympy available" if SYMPY_AVAILABLE else "⚠ sympy not available — using numeric derivative"
        ttk.Label(inp, text=sym_msg, foreground="gray").grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 0))

        plot_frame = ttk.LabelFrame(left_col, text=" Function Plot ", padding=8)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.figure = Figure(figsize=(4.2, 2.8), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("f(x) and Convergence Point", fontsize=9)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        tbl = ttk.LabelFrame(frm, text=" Iteration Results ", padding=8)
        tbl.grid(row=1, column=1, sticky="nsew", padx=8, pady=6)
        frm.rowconfigure(1, weight=1)

        cols = ("Iter", "x", "f(x)", "f'(x)", "x_new", "|dx|")
        self.tree = ttk.Treeview(tbl, columns=cols, show="headings", height=20)
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120 if c != "Iter" else 50, anchor="center")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(tbl, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        res = ttk.LabelFrame(frm, text=" Computed Result ", padding=10)
        res.grid(row=2, column=0, columnspan=2, sticky="we", padx=8, pady=(6, 10))
        self.result_var = tk.StringVar()
        ttk.Label(res, textvariable=self.result_var, font=("Segoe UI", 11, "bold"), foreground="#333").pack(anchor="w")

        ttk.Label(
            frm,
            text="Developed by Group 5 – Numerical Methods Project",
            font=("Segoe UI", 9, "italic"),
            foreground="gray",
        ).grid(row=3, column=0, columnspan=2, pady=(10, 0), sticky="e")

    def clear_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.result_var.set("")
        self.ax.clear()
        self.ax.set_title("f(x) and Convergence Point", fontsize=9)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")
        self.canvas.draw()

    def on_run(self):
        func_str = self.func_entry.get()
        try:
            x0 = float(self.x0_entry.get())
            tol = float(self.tol_entry.get())
            max_iter = int(self.maxit_entry.get())
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter numeric values.")
            return

        f_l, f_p = create_function(func_str)
        self.clear_table()

        try:
            root, iterations, converged = newton_raphson(f_l, f_p, x0, tol, max_iter)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        for (i, x_val, f_val, df_val, x_new, dx_abs) in iterations:
            self.tree.insert(
                "", "end",
                values=(i, f"{x_val:.6f}", f"{f_val:.6f}", f"{df_val:.6f}", f"{x_new:.6f}", f"{dx_abs:.2e}")
            )

        if converged:
            self.result_var.set(f"Converged to x = {root:.6f} after {len(iterations)} iterations.")
        else:
            self.result_var.set(f"Did not converge after {max_iter} iterations.")
        self.plot_function(f_l, root, func_str)

    def plot_function(self, f, root, func_str):
        self.ax.clear()
        self.ax.set_title("f(x) and Convergence Point", fontsize=9)
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("f(x)")

        x_vals = np.linspace(root - 5, root + 5, 400)
        try:
            y_vals = f(x_vals)
        except Exception:
            # fallback symbolic eval if numpy fails
            if SYMPY_AVAILABLE:
                x = sp.symbols("x")
                f_expr = sp.sympify(func_str.replace("^", "**"))
                y_vals = [float(f_expr.subs(x, val)) for val in x_vals]
            else:
                y_vals = [f(val) for val in x_vals]

        self.ax.axhline(0, color="gray", linewidth=0.8)
        self.ax.plot(x_vals, y_vals, label="f(x)", color="#2b5797")
        self.ax.scatter(root, f(root), color="red", label=f"Root ≈ {root:.4f}")
        self.ax.legend(fontsize=8)
        self.canvas.draw()


# =========================================================
# PROGRAM ENTRY POINT
# =========================================================
if __name__ == "__main__":
    app = NewtonRaphsonApp()
    app.mainloop()
