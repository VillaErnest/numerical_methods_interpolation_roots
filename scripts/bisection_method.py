import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import re

class BisectionCalculatorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bisection Method Calculator")
        self.root.geometry("700x600")
        self.root.configure(bg='#f0f0f0')
        
        self.calculator = BisectionCalculator()

        self.create_widgets()
        
    def create_widgets(self):
        title_label = tk.Label(self.root, text="Bisection Method Calculator", 
                              font=('Arial', 18, 'bold'), bg='#f0f0f0')
        title_label.pack(pady=10)

        desc_text = "This calculator finds roots of functions using the bisection method.\n" \
                   "You can enter mathematical functions using Python syntax with math module functions."
        desc_label = tk.Label(self.root, text=desc_text, 
                             font=('Arial', 10), bg='#f0f0f0', justify=tk.LEFT)
        desc_label.pack(pady=5)
  
        input_frame = tk.Frame(self.root, bg='#f0f0f0')
        input_frame.pack(pady=10, fill=tk.X, padx=20)

        func_label = tk.Label(input_frame, text="Function f(x):", 
                             font=('Arial', 10, 'bold'), bg='#f0f0f0')
        func_label.grid(row=0, column=0, sticky=tk.W, pady=5)
        
        self.func_entry = tk.Entry(input_frame, width=50, font=('Arial', 10))
        self.func_entry.grid(row=0, column=1, columnspan=2, sticky=tk.W+tk.E, pady=5, padx=5)

        a_label = tk.Label(input_frame, text="Left endpoint (a):", 
                          font=('Arial', 10, 'bold'), bg='#f0f0f0')
        a_label.grid(row=1, column=0, sticky=tk.W, pady=5)
        
        self.a_entry = tk.Entry(input_frame, width=15, font=('Arial', 10))
        self.a_entry.grid(row=1, column=1, sticky=tk.W, pady=5, padx=5)
        
        b_label = tk.Label(input_frame, text="Right endpoint (b):", 
                          font=('Arial', 10, 'bold'), bg='#f0f0f0')
        b_label.grid(row=1, column=2, sticky=tk.W, pady=5)
        
        self.b_entry = tk.Entry(input_frame, width=15, font=('Arial', 10))
        self.b_entry.grid(row=1, column=3, sticky=tk.W, pady=5, padx=5)
        
        tol_label = tk.Label(input_frame, text="Tolerance:", 
                            font=('Arial', 10, 'bold'), bg='#f0f0f0')
        tol_label.grid(row=2, column=0, sticky=tk.W, pady=5)
        
        self.tol_entry = tk.Entry(input_frame, width=15, font=('Arial', 10))
        self.tol_entry.grid(row=2, column=1, sticky=tk.W, pady=5, padx=5)
        
        iter_label = tk.Label(input_frame, text="Max iterations:", 
                             font=('Arial', 10, 'bold'), bg='#f0f0f0')
        iter_label.grid(row=2, column=2, sticky=tk.W, pady=5)
        
        self.iter_entry = tk.Entry(input_frame, width=15, font=('Arial', 10))
        self.iter_entry.grid(row=2, column=3, sticky=tk.W, pady=5, padx=5)
        
        input_frame.columnconfigure(1, weight=1)

        examples_frame = tk.LabelFrame(self.root, text="Example Functions", 
                                      font=('Arial', 10, 'bold'), bg='#f0f0f0')
        examples_frame.pack(pady=10, fill=tk.X, padx=20)
        
        examples = [
            "x**2 - 4",
            "x**3 - 2*x - 5",
            "math.sin(x) - 0.5*x",
            "math.exp(x) - 3*x",
            "math.log(x) - 1"
        ]
        
        for i, example in enumerate(examples):
            example_btn = tk.Button(examples_frame, text=example, 
                                   font=('Arial', 9),
                                   command=lambda ex=example: self.func_entry.delete(0, tk.END) or self.func_entry.insert(0, ex))
            example_btn.pack(side=tk.LEFT, padx=5, pady=5)

        button_frame = tk.Frame(self.root, bg='#f0f0f0')
        button_frame.pack(pady=10)
        
        self.calc_button = tk.Button(button_frame, text="Calculate Root", 
                                    font=('Arial', 12, 'bold'),
                                    command=self.calculate_root,
                                    bg='#4CAF50', fg='white', width=15)
        self.calc_button.pack(side=tk.LEFT, padx=10)
        
        self.clear_button = tk.Button(button_frame, text="Clear", 
                                     font=('Arial', 12),
                                     command=self.clear_all,
                                     bg='#f44336', fg='white', width=10)
        self.clear_button.pack(side=tk.LEFT, padx=10)

        results_frame = tk.Frame(self.root, bg='#f0f0f0')
        results_frame.pack(pady=10, fill=tk.BOTH, expand=True, padx=20)

        notebook = ttk.Notebook(results_frame)
        notebook.pack(fill=tk.BOTH, expand=True)

        results_tab = ttk.Frame(notebook)
        notebook.add(results_tab, text="Results")

        self.results_text = scrolledtext.ScrolledText(results_tab, height=15, 
                                                     font=('Consolas', 10))
        self.results_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        error_tab = ttk.Frame(notebook)
        notebook.add(error_tab, text="Error Analysis")

        self.error_text = scrolledtext.ScrolledText(error_tab, height=15, 
                                                   font=('Consolas', 10))
        self.error_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        plot_tab = ttk.Frame(notebook)
        notebook.add(plot_tab, text="Plot")

        self.plot_frame = tk.Frame(plot_tab, bg='white')
        self.plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
    def calculate_root(self):
        try:
            func_str = self.func_entry.get().strip()
            if not func_str:
                messagebox.showerror("Input Error", "Please enter a function")
                return
                
            if not self.a_entry.get():
                messagebox.showerror("Input Error", "Please enter left endpoint (a)")
                return
            a = float(self.a_entry.get())
            
            if not self.b_entry.get():
                messagebox.showerror("Input Error", "Please enter right endpoint (b)")
                return
            b = float(self.b_entry.get())
            
            if not self.tol_entry.get():
                messagebox.showerror("Input Error", "Please enter tolerance")
                return
            tol = float(self.tol_entry.get())
            
            if not self.iter_entry.get():
                messagebox.showerror("Input Error", "Please enter maximum iterations")
                return
            max_iter = int(self.iter_entry.get())

            f = self.calculator.parse_function(func_str)

            root, message = self.calculator.bisection_method(f, a, b, tol, max_iter)

            self.display_results(root, message, f, a, b)

            self.display_error_analysis()

            self.plot_results(f, a, b, root)
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {e}")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")
    
    def display_results(self, root, message, f, a, b):
        self.results_text.delete(1.0, tk.END)
        
        if root is not None:
            self.results_text.insert(tk.END, f"🎯 RESULT: {message}\n")
            self.results_text.insert(tk.END, f"Root ≈ {root:.10f}\n")
            self.results_text.insert(tk.END, f"f({root:.6f}) = {f(root):.2e}\n\n")

            self.results_text.insert(tk.END, "ITERATION HISTORY:\n")
            self.results_text.insert(tk.END, "-" * 80 + "\n")

            header = f"{'Iter':<5} {'a':<12} {'b':<12} {'c':<12} {'f(c)':<15} {'Error':<15}\n"
            self.results_text.insert(tk.END, header)
            self.results_text.insert(tk.END, "-" * 80 + "\n")

            for data in self.calculator.history:
                row = f"{data['iteration']:<5} {data['a']:<12.6f} {data['b']:<12.6f} " \
                      f"{data['c']:<12.6f} {data['f(c)']:<15.6e} {data['error']:<15.6e}\n"
                self.results_text.insert(tk.END, row)
        else:
            self.results_text.insert(tk.END, f"❌ {message}\n")
    
    def display_error_analysis(self):
        self.error_text.delete(1.0, tk.END)
        
        if len(self.calculator.history) < 2:
            self.error_text.insert(tk.END, "Not enough data for error analysis.\n")
            return
        
        errors = [data['error'] for data in self.calculator.history]
        convergence_rates = []
        
        self.error_text.insert(tk.END, "ERROR ANALYSIS\n")
        self.error_text.insert(tk.END, "=" * 60 + "\n")
        
        header = f"{'Iteration':<10} {'Error':<15} {'Ratio':<15} {'Convergence Rate':<15}\n"
        self.error_text.insert(tk.END, header)
        self.error_text.insert(tk.END, "-" * 60 + "\n")
        
        for i in range(1, len(errors)):
            if errors[i-1] > 0:
                ratio = errors[i] / errors[i-1]
                convergence_rates.append(ratio)
                rate_type = 'Linear' if 0.4 <= ratio <= 0.6 else 'Atypical'
                row = f"{i:<10} {errors[i]:<15.6e} {ratio:<15.6f} {rate_type:<15}\n"
                self.error_text.insert(tk.END, row)
        
        if convergence_rates:
            avg_rate = np.mean(convergence_rates)
            self.error_text.insert(tk.END, f"\nAverage convergence rate: {avg_rate:.6f}\n")
            if 0.4 <= avg_rate <= 0.6:
                self.error_text.insert(tk.END, "✓ Excellent convergence (typical for bisection method)\n")
            else:
                self.error_text.insert(tk.END, "⚠ Atypical convergence behavior\n")
    
    def plot_results(self, f, a, b, root):
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        if not self.calculator.history:
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        x_vals = np.linspace(a, b, 1000)
        y_vals = [f(x) for x in x_vals]
        
        ax.plot(x_vals, y_vals, 'b-', linewidth=2, label='f(x)')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax.axvline(x=root, color='r', linestyle='--', alpha=0.7, label=f'Root: {root:.6f}')

        ax.axvspan(self.calculator.history[0]['a'], self.calculator.history[0]['b'], 
                  alpha=0.2, color='green', label='Initial interval')

        iterations = [data['iteration'] for data in self.calculator.history]
        midpoints = [data['c'] for data in self.calculator.history]
        
        ax.plot(midpoints, [f(c) for c in midpoints], 'ro-', linewidth=1.5, 
                markersize=6, label='Bisection convergence')

        for i, (iter_num, c) in enumerate(zip(iterations, midpoints)):
            ax.plot(c, f(c), 'ro', markersize=6)
            if i % max(1, len(iterations)//5) == 0 or i == len(iterations)-1:
                ax.annotate(f'Iter {iter_num}', (c, f(c)), 
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=8, alpha=0.7)
        
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('Bisection Method: Function and Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasTkAgg(fig, self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def clear_all(self):
        self.func_entry.delete(0, tk.END)
        self.a_entry.delete(0, tk.END)
        self.b_entry.delete(0, tk.END)
        self.tol_entry.delete(0, tk.END)
        self.iter_entry.delete(0, tk.END)

        self.results_text.delete(1.0, tk.END)
        self.error_text.delete(1.0, tk.END)

        for widget in self.plot_frame.winfo_children():
            widget.destroy()


class BisectionCalculator:
    def __init__(self):
        self.history = []
        
    def parse_function(self, func_str):
        """Parse string function into a callable function"""
        func_str = func_str.replace('^', '**')
        func_str = re.sub(r'(\d)(x)', r'\1*\2', func_str)
        func_str = re.sub(r'(x)(\d)', r'\1*\2', func_str)
        func_str = re.sub(r'(x)(\()', r'\1*\2', func_str)
        func_str = re.sub(r'(\))(x)', r'\1*\2', func_str)
        
        try:
            code = f"lambda x: {func_str}"
            func = eval(code, {'__builtins__': None}, 
                       {'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
                        'exp': np.exp, 'log': np.log, 'log10': np.log10,
                        'sqrt': np.sqrt, 'pi': np.pi, 'e': np.e,
                        'math': __import__('math')})
            return func
        except Exception as e:
            raise ValueError(f"Error parsing function: {e}")
    
    def bisection_method(self, f, a, b, tol, max_iter):
        """Implement bisection method exactly as per the logic diagram"""
        self.history = []
        i = 0
        fa, fb = f(a), f(b)
        if fa * fb > 0:
            return None, "Error: f(a) and f(b) must have opposite signs. No root guaranteed in interval."
        
        while i <= max_iter:
            c = (a + b) / 2
            fc = f(c)

            iter_data = {
                'iteration': i,
                'a': a, 'b': b, 'c': c,
                'f(a)': fa, 'f(b)': fb, 'f(c)': fc,
                'error': abs(b - a) / 2
            }
            self.history.append(iter_data)

            if abs(fc) < 1e-15:
                return c, "Exact root found"

            if abs(fc) <= tol and i <= max_iter:
                return c, "Root found within tolerance"

            if np.sign(fc) == np.sign(fa):
                a = c
                fa = fc
            else:
                b = c
                fb = fc
            
            i += 1

        c = (a + b) / 2
        return c, "Maximum iterations reached"


if __name__ == "__main__":
    root = tk.Tk()
    app = BisectionCalculatorGUI(root)
    root.mainloop()