import tkinter as tk
from tkinter import ttk, messagebox
import math

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except Exception:
    SYMPY_AVAILABLE = False


# ======================================
#  FUNCTION CREATION & SAFE EVALUATION
# ======================================
def safe_eval_func(expr_str):

    if SYMPY_AVAILABLE:
        x = sp.symbols('x')
        try:
            expr = sp.sympify(expr_str)
        except Exception as e:
            raise ValueError(f"Invalid function expression: {e}")
        f_lambda = sp.lambdify(x, expr, modules=["math"])
        return f_lambda, expr
    else:
        allowed_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("__")}
        allowed_names['x'] = 0

        def f(x_val):
            allowed_names['x'] = x_val
            try:
                return eval(expr_str, {"__builtins__": {}}, allowed_names)
            except Exception as e:
                raise ValueError(f"Error evaluating function: {e}")

        return f, None


# ======================================
#   DERIVATIVE (NUMERIC & SYMBOLIC)
# ======================================
def numeric_derivative(f, x, h=1e-6):

    return (f(x + h) - f(x - h)) / (2 * h)


# ======================================
#   NEWTON–RAPHSON ITERATION PROCESS
# ======================================
def newton_raphson(f, df, x0, tol=1e-6, max_iter=50):

    iterations = []
    x = x0
    for i in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)

        if dfx == 0:
            raise ZeroDivisionError(f"Zero derivative encountered at iteration {i}, x={x}")

        x_new = x - fx / dfx
        iterations.append((i, x, fx, dfx, x_new, abs(x_new - x)))

        if abs(x_new - x) < tol:
            return x_new, iterations, True

        x = x_new

    return x, iterations, False


# ======================================
# GUI DESIGN, USER INTERACTION & OUTPUT DISPLAY
# ======================================
class NewtonRaphsonGUI(tk.Tk):
    """Graphical User Interface for the Newton–Raphson Method."""

    def __init__(self):
        super().__init__()
        self.title("Newton-Raphson (Open Method)")
        self.geometry("1180x720")
        self.resizable(True, True)

        # ---- Theme & Style ----
        style = ttk.Style(self)
        style.theme_use("clam")

        # Colors
        bg_main = "#f4f6f9"
        accent = "#2b5797"
        accent_light = "#c6d4f0"

        self.configure(bg=bg_main)
        style.configure("TFrame", background=bg_main)
        style.configure("TLabel", background=bg_main, font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=6)
        style.map("TButton", background=[("active", accent_light)])
        style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), background=accent, foreground="white")
        style.configure("Treeview", font=("Consolas", 9), rowheight=22)

        self.create_widgets()

    def create_widgets(self):
        """Create all GUI widgets and layout."""
        frm = ttk.Frame(self, padding=15)
        frm.pack(fill=tk.BOTH, expand=True)

        title_lbl = ttk.Label(frm, text="Newton–Raphson Method (Open Method)", font=("Segoe UI", 15, "bold"), foreground="#2b5797")
        title_lbl.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky="w")

        # Input Section
        inp = ttk.LabelFrame(frm, text=" User Inputs ", padding=12)
        inp.grid(row=1, column=0, sticky="nw", padx=8, pady=6)

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

        # Buttons Row
        btn_frame = ttk.Frame(inp)
        btn_frame.grid(row=3, column=0, columnspan=4, pady=(10, 0))
        run_btn = ttk.Button(btn_frame, text="▶ Run Newton-Raphson", command=self.on_run)
        run_btn.grid(row=0, column=0, padx=4)
        clear_btn = ttk.Button(btn_frame, text="🧹 Clear Table", command=self.clear_table)
        clear_btn.grid(row=0, column=1, padx=4)

        # sympy info label
        sympy_msg = "✓ sympy available" if SYMPY_AVAILABLE else "⚠ sympy not available — using numeric derivative"
        sym_lbl = ttk.Label(inp, text=sympy_msg, foreground="gray")
        sym_lbl.grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 0))

        # Iteration Table
        tbl = ttk.LabelFrame(frm, text=" Iteration Results ", padding=8)
        tbl.grid(row=1, column=1, sticky="ne", padx=8, pady=6)

        cols = ("Iter", "x", "f(x)", "f'(x)", "x_new", "|dx|")
        self.tree = ttk.Treeview(tbl, columns=cols, show="headings", height=20)

        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=120 if c != "Iter" else 50, anchor="center")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH)

        scrollbar = ttk.Scrollbar(tbl, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Result Section
        res = ttk.LabelFrame(frm, text=" Computed Result ", padding=10)
        res.grid(row=2, column=0, columnspan=2, sticky="we", padx=8, pady=(6, 10))

        self.result_var = tk.StringVar()
        ttk.Label(res, textvariable=self.result_var, font=("Segoe UI", 11, "bold"), foreground="#333").pack(anchor="w")

        # Footer Label
        footer = ttk.Label(frm, text="Developed by Group 5 – Numerical Methods Project", font=("Segoe UI", 9, "italic"), foreground="gray")
        footer.grid(row=3, column=0, columnspan=2, pady=(10, 0), sticky="e")

    def clear_table(self):
        """Clears all table data and result output."""
        for i in self.tree.get_children():
            self.tree.delete(i)
        self.result_var.set("")

    def on_run(self):
        """Runs the Newton–Raphson algorithm when the button is pressed."""
        self.clear_table()

        func_str = self.func_entry.get().strip()
        x0_str = self.x0_entry.get().strip()
        tol_str = self.tol_entry.get().strip()
        maxit_str = self.maxit_entry.get().strip()

        try:
            x0 = float(x0_str)
            tol = float(tol_str)
            maxit = int(maxit_str)
        except Exception:
            messagebox.showerror("Input Error", "Invalid numeric input.")
            return

        # Function & derivative setup
        try:
            f_callable, sym_expr = safe_eval_func(func_str)
        except Exception as e:
            messagebox.showerror("Function Error", str(e))
            return

        if SYMPY_AVAILABLE and sym_expr is not None:
            x = sp.symbols('x')
            try:
                d_sym = sp.diff(sym_expr, x)
                d_callable = sp.lambdify(x, d_sym, modules=["math"])
            except Exception as e:
                messagebox.showwarning("Derivative Warning",
                                       f"Symbolic derivative failed: {e}\nUsing numeric derivative instead.")
                d_callable = lambda xx: numeric_derivative(f_callable, xx)
        else:
            d_callable = lambda xx: numeric_derivative(f_callable, xx)

        # Run the Newton-Raphson computation
        try:
            root, iters, converged = newton_raphson(f_callable, d_callable, x0, tol=tol, max_iter=maxit)
        except Exception as e:
            messagebox.showerror("Computation Error", str(e))
            return

        # Display results in table
        for (i, x_val, fx, dfx, x_new, dx_abs) in iters:
            self.tree.insert("", tk.END,
                             values=(i, f"{x_val:.10g}", f"{fx:.5g}", f"{dfx:.5g}", f"{x_new:.10g}", f"{dx_abs:.5g}"))

        if converged:
            self.result_var.set(f"✅ Converged: root ≈ {root:.10g} after {len(iters)} iterations. f(root) = {f_callable(root):.5g}")
        else:
            self.result_var.set(f"⚠ Stopped: reached max iterations. Last x ≈ {root:.10g}. f(x) = {f_callable(root):.5g}")

# ======================================
# Program Entry Point
# ======================================
if __name__ == "__main__":
    app = NewtonRaphsonGUI()
    app.mainloop()
