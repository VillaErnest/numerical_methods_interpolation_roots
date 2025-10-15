import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import time
import math

def secant_method(func_str, x0, x1, tol=1e-5, max_iter=100):
    """Secant Method with enhanced output formatting"""
    try:
        # Import math functions into namespace for evaluation
        safe_namespace = {
            "x": 0,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "exp": math.exp,
            "log": math.log,
            "log10": math.log10,
            "sqrt": math.sqrt,
            "pi": math.pi,
            "e": math.e,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            "sinh": math.sinh,
            "cosh": math.cosh,
            "tanh": math.tanh,
            "__builtins__": {}
        }

        func = lambda x: eval(func_str, {**safe_namespace, "x": x})
        results = []
        start_time = time.time()

        # Store iteration data for table
        iteration_data = []

        for iteration in range(1, max_iter + 1):
            f_x0 = func(x0)
            f_x1 = func(x1)

            if abs(f_x1 - f_x0) < 1e-14:
                return None, "Division by zero", iteration_data, 0, 0

            x_new = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
            f_new = func(x_new)
            error = abs(x_new - x1)

            # Store data for table
            iteration_data.append({
                'iter': iteration,
                'x_n': x_new,
                'f_x': f_new,
                'error': error
            })

            if error < tol:
                elapsed = time.time() - start_time
                return x_new, "success", iteration_data, iteration, elapsed

            x0, x1 = x1, x_new

        elapsed = time.time() - start_time
        return x1, "Max iterations", iteration_data, max_iter, elapsed

    except Exception as e:
        return None, f"Error: {str(e)}", [], 0, 0

class SecantMethodGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Secant Method Calculator - ECE 311")
        self.root.geometry("900x750")
        self.root.configure(bg='#f0f0f0')
        self.setup_ui()

    def setup_ui(self):
        # Header
        header = tk.Frame(self.root, bg='#34495e', height=80)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        tk.Label(header, text="SECANT METHOD ROOT FINDER",
                font=('Arial', 20, 'bold'),
                bg='#34495e', fg='white').pack(pady=25)

        # Input Parameters Frame
        input_frame = tk.LabelFrame(self.root, text="Input Parameters",
                                   font=('Arial', 11, 'bold'),
                                   bg='#f0f0f0',
                                   padx=25, pady=20)
        input_frame.pack(padx=15, pady=15, fill=tk.X)

        # Function f(x)
        tk.Label(input_frame, text="Function f(x):",
                font=('Arial', 10, 'bold'),
                bg='#f0f0f0').grid(row=0, column=0, sticky='w', pady=8)

        func_container = tk.Frame(input_frame, bg='#f0f0f0')
        func_container.grid(row=0, column=1, columnspan=2, sticky='w', pady=8)

        self.func_entry = tk.Entry(func_container, width=40,
                                  font=('Courier', 10),
                                  relief='solid', bd=1)
        self.func_entry.pack(side=tk.LEFT)

        tk.Label(func_container,
                text=" (e.g., cos(x)-x, x**2-3*x+1, exp(x)-2)",
                font=('Arial', 8), fg='gray',
                bg='#f0f0f0').pack(side=tk.LEFT, padx=5)

        # Initial x0
        tk.Label(input_frame, text="Initial x₀:",
                font=('Arial', 10, 'bold'),
                bg='#f0f0f0').grid(row=1, column=0, sticky='w', pady=8)

        self.x0_entry = tk.Entry(input_frame, width=20,
                               font=('Arial', 10),
                               relief='solid', bd=1)
        self.x0_entry.grid(row=1, column=1, sticky='w', pady=8)

        # Initial x1
        tk.Label(input_frame, text="Initial x₁:",
                font=('Arial', 10, 'bold'),
                bg='#f0f0f0').grid(row=2, column=0, sticky='w', pady=8)

        self.x1_entry = tk.Entry(input_frame, width=20,
                               font=('Arial', 10),
                               relief='solid', bd=1)
        self.x1_entry.grid(row=2, column=1, sticky='w', pady=8)

        # Tolerance
        tk.Label(input_frame, text="Tolerance:",
                font=('Arial', 10, 'bold'),
                bg='#f0f0f0').grid(row=3, column=0, sticky='w', pady=8)

        self.tol_entry = tk.Entry(input_frame, width=20,
                                font=('Arial', 10),
                                relief='solid', bd=1)
        self.tol_entry.insert(0, "0.00001")
        self.tol_entry.grid(row=3, column=1, sticky='w', pady=8)

        # Max Iterations
        tk.Label(input_frame, text="Max Iterations:",
                font=('Arial', 10, 'bold'),
                bg='#f0f0f0').grid(row=4, column=0, sticky='w', pady=8)

        self.max_iter_entry = tk.Entry(input_frame, width=20,
                                     font=('Arial', 10),
                                     relief='solid', bd=1)
        self.max_iter_entry.insert(0, "100")
        self.max_iter_entry.grid(row=4, column=1, sticky='w', pady=8)

        # Buttons Frame
        btn_frame = tk.Frame(self.root, bg='#f0f0f0')
        btn_frame.pack(pady=10)

        self.calc_btn = tk.Button(btn_frame, text="Calculate",
                                 bg='#27ae60', fg='white',
                                 font=('Arial', 11, 'bold'),
                                 width=15, height=2,
                                 relief='raised', bd=2,
                                 cursor='hand2',
                                 command=self.calculate)
        self.calc_btn.pack(side=tk.LEFT, padx=5)

        self.clear_btn = tk.Button(btn_frame, text="Clear",
                                  bg='#e74c3c', fg='white',
                                  font=('Arial', 11, 'bold'),
                                  width=15, height=2,
                                  relief='raised', bd=2,
                                  cursor='hand2',
                                  command=self.clear_all)
        self.clear_btn.pack(side=tk.LEFT, padx=5)

        self.example_btn = tk.Button(btn_frame, text="Load Example",
                                    bg='#3498db', fg='white',
                                    font=('Arial', 11, 'bold'),
                                    width=15, height=2,
                                    relief='raised', bd=2,
                                    cursor='hand2',
                                    command=self.load_example)
        self.example_btn.pack(side=tk.LEFT, padx=5)

        # Results Frame with ENHANCED TABLE
        results_frame = tk.LabelFrame(self.root, text="Iteration Results",
                                    font=('Arial', 11, 'bold'),
                                    bg='#f0f0f0',
                                    padx=15, pady=15)
        results_frame.pack(padx=15, pady=10, fill=tk.BOTH, expand=True)

        # Create custom text widget with tags for formatting
        self.results_text = tk.Text(results_frame,
                                   font=('Courier New', 9),
                                   bg='white',
                                   fg='#2c3e50',
                                   relief='solid',
                                   bd=1,
                                   wrap=tk.NONE,
                                   padx=10,
                                   pady=10)

        # Add scrollbars
        y_scrollbar = tk.Scrollbar(results_frame, command=self.results_text.yview)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        x_scrollbar = tk.Scrollbar(results_frame, orient=tk.HORIZONTAL,
                                 command=self.results_text.xview)
        x_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)

        self.results_text.config(yscrollcommand=y_scrollbar.set,
                               xscrollcommand=x_scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Configure text tags for colored/styled output
        self.results_text.tag_config('header', font=('Courier New', 10, 'bold'),
                                    foreground='#2c3e50')
        self.results_text.tag_config('subheader', font=('Courier New', 9, 'bold'),
                                    foreground='#34495e')
        self.results_text.tag_config('table_header', font=('Courier New', 9, 'bold'),
                                    foreground='white', background='#34495e')
        self.results_text.tag_config('data_row', font=('Courier New', 9),
                                    foreground='#2c3e50')
        self.results_text.tag_config('converged', font=('Courier New', 9),
                                    foreground='#27ae60', background='#d5f4e6')
        self.results_text.tag_config('summary', font=('Courier New', 10, 'bold'),
                                    foreground='#27ae60')
        self.results_text.tag_config('error_small', font=('Courier New', 9),
                                    foreground='#f39c12')

        # Footer
        footer = tk.Label(self.root,
                         text="ECE 311 - Numerical Methods | Secant Method Implementation",
                         font=('Arial', 9),
                         bg='#34495e', fg='white',
                         pady=12)
        footer.pack(side=tk.BOTTOM, fill=tk.X)

    def format_iteration_table(self, iteration_data, x0_initial, x1_initial):
        """Format iteration data into an enhanced table"""
        self.results_text.delete(1.0, tk.END)

        # Initial values section
        self.results_text.insert(tk.END, "Initial Values: ", 'subheader')
        self.results_text.insert(tk.END, f"x₀ = {x0_initial}, x₁ = {x1_initial}\n\n", 'data_row')

        # Table header with proper alignment
        header_line = f"{'Iter':<8} {'x_n':>20} {'f(x_n)':>22} {'Error':>22}\n"
        self.results_text.insert(tk.END, header_line, 'table_header')

        # Separator line
        separator = "─" * 75 + "\n"
        self.results_text.insert(tk.END, separator, 'subheader')

        # Data rows with proper formatting
        for data in iteration_data:
            iter_num = data['iter']
            x_n = data['x_n']
            f_x = data['f_x']
            error = data['error']

            # Format each value with proper width and precision
            row = f"{iter_num:<8} {x_n:>20.10f} {f_x:>22.12e} {error:>22.12e}\n"

            # Color code based on error magnitude
            if error < 1e-8:
                self.results_text.insert(tk.END, row, 'converged')
            elif error < 1e-4:
                self.results_text.insert(tk.END, row, 'error_small')
            else:
                self.results_text.insert(tk.END, row, 'data_row')

    def calculate(self):
        """Execute secant method with enhanced display"""
        try:
            func = self.func_entry.get()
            x0 = float(self.x0_entry.get())
            x1 = float(self.x1_entry.get())
            tol = float(self.tol_entry.get())
            max_iter = int(self.max_iter_entry.get())

            if not func:
                messagebox.showerror("Input Error", "Please enter a function!")
                return

            # Run calculation
            root, status, iteration_data, iterations, elapsed = secant_method(
                func, x0, x1, tol, max_iter
            )

            # Display formatted table
            self.format_iteration_table(iteration_data, x0, x1)

            # Add convergence message
            if status == "success":
                self.results_text.insert(tk.END, "\n", 'data_row')
                self.results_text.insert(tk.END, "Converged!\n", 'summary')
                self.results_text.insert(tk.END, f"Root found at x = {root:.10f}\n", 'summary')
                self.results_text.insert(tk.END, f"f(x) = {iteration_data[-1]['f_x']:.12e}\n", 'data_row')
                self.results_text.insert(tk.END, f"Iterations: {iterations}\n", 'data_row')
                messagebox.showinfo("Success", f"Root found: {root:.6f}")
            else:
                self.results_text.insert(tk.END, f"\n{status}\n", 'data_row')
                messagebox.showwarning("Warning", status)

        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numbers!")
        except Exception as e:
            messagebox.showerror("Error", f"Calculation error: {str(e)}")

    def load_example(self):
        """Load example problem with transcendental function"""
        self.func_entry.delete(0, tk.END)
        self.func_entry.insert(0, "cos(x) - x")
        self.x0_entry.delete(0, tk.END)
        self.x0_entry.insert(0, "0")
        self.x1_entry.delete(0, tk.END)
        self.x1_entry.insert(0, "1")
        messagebox.showinfo("Example Loaded",
                          "Example: f(x) = cos(x) - x\nExpected root ≈ 0.739085")

    def clear_all(self):
        """Clear all inputs and outputs"""
        self.func_entry.delete(0, tk.END)
        self.x0_entry.delete(0, tk.END)
        self.x1_entry.delete(0, tk.END)
        self.tol_entry.delete(0, tk.END)
        self.tol_entry.insert(0, "0.00001")
        self.max_iter_entry.delete(0, tk.END)
        self.max_iter_entry.insert(0, "100")
        self.results_text.delete(1.0, tk.END)

if __name__ == "__main__":
    root = tk.Tk()
    app = SecantMethodGUI(root)
    root.mainloop()
