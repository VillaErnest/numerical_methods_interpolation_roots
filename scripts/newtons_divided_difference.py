import tkinter as tk
from tkinter import messagebox, scrolledtext, filedialog
import matplotlib.pyplot as plt
import numpy as np
import math

# -------------------------------
# Core Numerical Functions
# -------------------------------

def parse_input(text):
    """Convert comma-separated string into list of floats."""
    try:
        return [float(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError:
        messagebox.showerror("Input Error", "Enter valid numbers separated by commas.")
        return []

def divided_difference_table(x, y):
    """Generate divided difference table."""
    n = len(x)
    table = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        table[i][0] = y[i]
    for j in range(1, n):
        for i in range(n - j):
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (x[i+j] - x[i])
    return table

def newton_interpolation(x, table, value):
    """Evaluate Newton’s interpolation at a specific value."""
    n = len(x)
    result = table[0][0]
    term = 1.0
    for i in range(1, n):
        term *= (value - x[i-1])
        result += table[0][i] * term
    return result

def polynomial_to_string(x, table):
    """Return polynomial string."""
    n = len(x)
    poly = f"{table[0][0]:.4f}"
    for i in range(1, n):
        poly += f" + ({table[0][i]:.4f})"
        for j in range(i):
            poly += f"*(x - {x[j]:.2f})"
    return poly

def show_steps(x, table, value):
    """Show step-by-step evaluation of each polynomial term."""
    steps = []
    result = table[0][0]
    term = 1.0
    steps.append(f"Term 0: {table[0][0]:.6f}")
    for i in range(1, len(x)):
        term *= (value - x[i-1])
        contrib = table[0][i] * term
        result += contrib
        steps.append(f"Term {i}: ({table[0][i]:.6f}) * product({[f'(x - {x[j]})' for j in range(i)]}) = {contrib:.6f}")
    steps.append(f"Final f({value}) = {result:.6f}")
    return "\n".join(steps)

def compute_percent_error(true_value, approx_value):
    """Compute percent error."""
    try:
        return abs((true_value - approx_value) / true_value) * 100
    except ZeroDivisionError:
        return float('inf')

# -------------------------------
# GUI Functions
# -------------------------------

def compute():
    x_vals = parse_input(x_entry.get())
    y_vals = parse_input(y_entry.get())
    if not x_vals or not y_vals:
        return
    if len(x_vals) != len(y_vals):
        messagebox.showerror("Input Error", "X and Y must have the same number of values.")
        return
    try:
        x_target = float(x_value_entry.get())
    except ValueError:
        messagebox.showerror("Input Error", "Enter a valid number for x to estimate.")
        return

    table = divided_difference_table(x_vals, y_vals)
    fx_approx = newton_interpolation(x_vals, table, x_target)
    poly = polynomial_to_string(x_vals, table)
    steps = show_steps(x_vals, table, x_target)

    # Calculate true value and percent error (using ln(x) as true function)
    try:
        fx_true = math.log(x_target)
        percent_error = compute_percent_error(fx_true, fx_approx)
    except ValueError:
        fx_true = None
        percent_error = None

    # Display results in text box
    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, "Divided Difference Table:\n")
    for i in range(len(x_vals)):
        row = "".join(f"{table[i][j]:10.4f}\t" for j in range(len(x_vals) - i))
        result_text.insert(tk.END, row + "\n")
    result_text.insert(tk.END, f"\nNewton Polynomial:\n{poly}\n")
    result_text.insert(tk.END, f"\nEstimated f({x_target}) = {fx_approx:.6f}\n")

    if fx_true is not None:
        result_text.insert(tk.END, f"True Value (ln({x_target})) = {fx_true:.6f}\n")
        result_text.insert(tk.END, f"Percent Error = {percent_error:.4f}%\n\n")

        # 💬 Pop-up summary
        messagebox.showinfo(
            "Computation Result",
            f"✅ Computation Complete!\n\n"
            f"f({x_target}) ≈ {fx_approx:.6f}\n"
            f"True Value: {fx_true:.6f}\n"
            f"Percent Error: {percent_error:.4f}%"
        )

    result_text.insert(tk.END, "Evaluation Steps:\n" + steps)

def plot_graph():
    x_vals = parse_input(x_entry.get())
    y_vals = parse_input(y_entry.get())
    if not x_vals or not y_vals or len(x_vals) != len(y_vals):
        messagebox.showerror("Error", "Enter valid and matching X, Y data first.")
        return

    # True curve (ln(x))
    x_range = np.linspace(min(x_vals), max(x_vals), 200)
    y_true = np.log(x_range)

    plt.figure(figsize=(7, 5))
    plt.plot(x_range, y_true, color='black', linewidth=2.5, label='f(x) = ln(x)')

    # Linear approximations between points
    for i in range(len(x_vals) - 1):
        plt.plot([x_vals[i], x_vals[i+1]], [y_vals[i], y_vals[i+1]], color='deepskyblue', linewidth=2)

    # Data points
    plt.scatter(x_vals, y_vals, facecolors='white', edgecolors='black', s=80, zorder=5)
    plt.scatter(x_vals[1], y_vals[1], color='black', s=60, label='True value', zorder=6)

    plt.title("f(x) = ln(x) with Linear Estimates", fontsize=13)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("f(x)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def save_to_file():
    output = result_text.get(1.0, tk.END).strip()
    if not output:
        messagebox.showinfo("No Data", "Compute results first before saving.")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
    if file_path:
        with open(file_path, "w") as f:
            f.write(output)
        messagebox.showinfo("Saved", f"Results saved to:\n{file_path}")

def clear_all():
    x_entry.delete(0, tk.END)
    y_entry.delete(0, tk.END)
    x_value_entry.delete(0, tk.END)
    result_text.delete(1.0, tk.END)

# -------------------------------
# GUI Setup
# -------------------------------

root = tk.Tk()
root.title("Newton's Divided Difference Interpolation")
root.geometry("750x680")
root.configure(bg="#F8F9FA")

tk.Label(root, text="Newton’s Divided Difference Interpolation",
         font=("Arial", 18, "bold"), bg="#F8F9FA", fg="#212529").pack(pady=10)

frame = tk.Frame(root, bg="#F8F9FA")
frame.pack(pady=10)

tk.Label(frame, text="X values (comma-separated):", bg="#F8F9FA", font=("Arial", 11)).grid(row=0, column=0, sticky="e")
x_entry = tk.Entry(frame, width=55)
x_entry.grid(row=0, column=1, padx=10, pady=5)

tk.Label(frame, text="Y values (comma-separated):", bg="#F8F9FA", font=("Arial", 11)).grid(row=1, column=0, sticky="e")
y_entry = tk.Entry(frame, width=55)
y_entry.grid(row=1, column=1, padx=10, pady=5)

tk.Label(frame, text="Estimate at x =", bg="#F8F9FA", font=("Arial", 11)).grid(row=2, column=0, sticky="e")
x_value_entry = tk.Entry(frame, width=15)
x_value_entry.grid(row=2, column=1, sticky="w", padx=10, pady=5)

# Buttons
button_frame = tk.Frame(root, bg="#F8F9FA")
button_frame.pack(pady=10)

tk.Button(button_frame, text="Compute", width=12, bg="#198754", fg="white",
          font=("Arial", 11, "bold"), command=compute).grid(row=0, column=0, padx=5)
tk.Button(button_frame, text="Plot Graph", width=12, bg="#0D6EFD", fg="white",
          font=("Arial", 11, "bold"), command=plot_graph).grid(row=0, column=1, padx=5)
tk.Button(button_frame, text="Save Result", width=12, bg="#FFC107", fg="black",
          font=("Arial", 11, "bold"), command=save_to_file).grid(row=0, column=2, padx=5)
tk.Button(button_frame, text="Clear All", width=12, bg="#DC3545", fg="white",
          font=("Arial", 11, "bold"), command=clear_all).grid(row=0, column=3, padx=5)

# Results
tk.Label(root, text="Results:", bg="#F8F9FA", font=("Arial", 13, "bold")).pack()
result_text = scrolledtext.ScrolledText(root, width=85, height=22, font=("Consolas", 10))
result_text.pack(pady=10)

root.mainloop()
