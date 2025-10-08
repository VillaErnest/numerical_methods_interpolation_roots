import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import math

# -----------------------------
# Modified Secant Method Function
# -----------------------------
def modified_secant_method(func, x0, delta, tol, max_iter):
    iteration_data = []
    for i in range(max_iter):
        f_x0 = func(x0)
        f_x0_delta = func(x0 + delta * x0)
        if f_x0_delta - f_x0 == 0:
            messagebox.showerror("Math Error", "Division by zero encountered.")
            return None, iteration_data

        x1 = x0 - (f_x0 * delta * x0) / (f_x0_delta - f_x0)
        iteration_data.append((i + 1, x0, f_x0, x1))
        if abs(x1 - x0) < tol:
            return x1, iteration_data
        x0 = x1
    return x1, iteration_data

# -----------------------------
# Function to Evaluate Input Expression
# -----------------------------
def evaluate_function(expr):
    def f(x):
        return eval(expr, {"x": x, "math": math})
    return f

# -----------------------------
# Functionality: Solve Button
# -----------------------------
def solve_equation():
    try:
        expr = entry_function.get()
        x0 = float(entry_x0.get())
        delta = float(entry_delta.get())
        tol = float(entry_tol.get())
        max_iter = int(entry_iter.get())

        func = evaluate_function(expr)
        root_val, iterations = modified_secant_method(func, x0, delta, tol, max_iter)

        if root_val is not None:
            result_text = f"Approximate Root: {root_val:.6f}\n\nIteration Data:\n"
            result_text += "Iter\t\tx0\t\tf(x0)\t\tx1\n"
            result_text += "-" * 55 + "\n"
            for i, x_old, fx, x_new in iterations:
                result_text += f"{i}\t{x_old:.6f}\t{fx:.6f}\t{x_new:.6f}\n"

            text_result.config(state=tk.NORMAL)
            text_result.delete(1.0, tk.END)
            text_result.insert(tk.END, result_text)
            text_result.config(state=tk.DISABLED)
        else:
            text_result.config(state=tk.NORMAL)
            text_result.delete(1.0, tk.END)
            text_result.insert(tk.END, "Computation failed.")
            text_result.config(state=tk.DISABLED)

    except Exception as e:
        messagebox.showerror("Input Error", f"Invalid input or expression.\n\nDetails: {e}")

# -----------------------------
# GUI Setup
# -----------------------------
root = tk.Tk()
root.title("USTP Numerical Methods - Modified Secant Method")
root.geometry("850x650")
root.resizable(False, False)
root.configure(bg="#1A1446")  # Deep USTP blue

# Center window on screen
window_width = 850
window_height = 650
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_cordinate = int((screen_width/2) - (window_width/2))
y_cordinate = int((screen_height/2) - (window_height/2))
root.geometry(f"{window_width}x{window_height}+{x_cordinate}+{y_cordinate}")

# -----------------------------
# USTP Logo and Title
# -----------------------------
try:
    logo = Image.open("ustp_logo.png")
    logo = logo.resize((100, 100), Image.LANCZOS)
    logo_img = ImageTk.PhotoImage(logo)
    logo_label = tk.Label(root, image=logo_img, bg="#1A1446")
    logo_label.image = logo_img
    logo_label.pack(pady=(15, 5))
except Exception as e:
    print("Logo not found or failed to load:", e)

tk.Label(
    root,
    text="UNIVERSITY OF SCIENCE AND TECHNOLOGY OF SOUTHERN PHILIPPINES",
    font=("Segoe UI", 12, "bold"),
    fg="white",
    bg="#1A1446"
).pack()

tk.Label(
    root,
    text="Modified Secant Method Solver",
    font=("Segoe UI", 18, "bold"),
    fg="#F9B233",  # USTP Gold
    bg="#1A1446",
    pady=10
).pack()

# -----------------------------
# Input Frame
# -----------------------------
input_frame = tk.Frame(root, bg="white", padx=20, pady=20, bd=0, relief="flat")
input_frame.pack(pady=10)

def create_labeled_entry(frame, label_text, default_val, row):
    tk.Label(frame, text=label_text, bg="white", font=("Segoe UI", 11)).grid(row=row, column=0, sticky="w", pady=5)
    entry = tk.Entry(frame, width=30, font=("Consolas", 11))
    entry.grid(row=row, column=1, padx=10, pady=5)
    entry.insert(0, default_val)
    return entry

# Function entry fields
tk.Label(input_frame, text="f(x) =", bg="white", font=("Segoe UI", 11, "bold")).grid(row=0, column=0, sticky="w", pady=5)
entry_function = tk.Entry(input_frame, width=35, font=("Consolas", 11))
entry_function.grid(row=0, column=1, padx=10, pady=5)
entry_function.insert(0, "math.exp(-x) - x")

entry_x0 = create_labeled_entry(input_frame, "Initial guess (x₀):", "0.5", 1)
entry_delta = create_labeled_entry(input_frame, "Perturbation (δ):", "0.01", 2)
entry_tol = create_labeled_entry(input_frame, "Tolerance:", "1e-6", 3)
entry_iter = create_labeled_entry(input_frame, "Max iterations:", "20", 4)

# Solve Button
solve_btn = tk.Button(
    root,
    text="Compute Root",
    command=solve_equation,
    bg="#F9B233",
    fg="#1A1446",
    font=("Segoe UI", 12, "bold"),
    relief="flat",
    padx=25,
    pady=8
)
solve_btn.pack(pady=10)

# -----------------------------
# Output Frame
# -----------------------------
output_frame = tk.Frame(root, bg="white", padx=10, pady=10)
output_frame.pack(pady=10, fill="both", expand=True)

tk.Label(output_frame, text="Results:", bg="white", font=("Segoe UI", 12, "bold")).pack(anchor="w")

text_result = tk.Text(output_frame, height=12, width=90, font=("Consolas", 10), wrap="none", bg="#F8F8F8")
text_result.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
scrollbar = ttk.Scrollbar(output_frame, orient="vertical", command=text_result.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
text_result.config(yscrollcommand=scrollbar.set, state=tk.DISABLED)

# Footer
tk.Label(
    root,
    text="Developed for Numerical Methods PIT | USTP - 2025",
    bg="#1A1446",
    fg="white",
    font=("Segoe UI", 9)
).pack(side="bottom", pady=5)

root.mainloop()
