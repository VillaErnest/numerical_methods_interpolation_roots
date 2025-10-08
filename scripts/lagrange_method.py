import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy.interpolate import lagrange

class LagrangeInterpolationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Lagrange Interpolation Calculator")
        self.root.geometry("950x600")
        self.root.resizable(True, True)

        # Windows 11 color scheme
        self.bg_color = "#F3F3F3"
        self.panel_color = "#FFFFFF"
        self.accent_color = "#0078D4"
        self.text_color = "#1F1F1F"
        self.border_color = "#E0E0E0"
        self.hover_color = "#005A9E"

        self.root.configure(bg=self.bg_color)

        # Configure ttk style for Windows 11 look
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Configure custom styles
        self.style.configure('Card.TFrame', background=self.panel_color, relief='flat')
        self.style.configure('Main.TFrame', background=self.bg_color)
        self.style.configure('Title.TLabel', background=self.panel_color, foreground=self.text_color, 
                           font=('Segoe UI', 16, 'bold'))
        self.style.configure('Heading.TLabel', background=self.panel_color, foreground=self.text_color,
                           font=('Segoe UI', 12, 'bold'))
        self.style.configure('Normal.TLabel', background=self.panel_color, foreground=self.text_color,
                           font=('Segoe UI', 10))
        self.style.configure('Custom.TEntry', fieldbackground='white', font=('Segoe UI', 10))

        # Main container
        main_frame = ttk.Frame(root, style='Main.TFrame', padding=15)
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weights
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Left Panel - INPUT
        left_panel = tk.Frame(main_frame, bg=self.panel_color, relief='flat', bd=0)
        left_panel.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 8))
        self._add_shadow(left_panel)

        # Right Panel - RESULTS
        right_panel = tk.Frame(main_frame, bg=self.panel_color, relief='flat', bd=0)
        right_panel.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(8, 0))
        self._add_shadow(right_panel)

        # LEFT PANEL CONTENT
        left_content = ttk.Frame(left_panel, style='Card.TFrame', padding=20)
        left_content.pack(fill='both', expand=True)

        # Title
        title = ttk.Label(left_content, text="INPUT PANEL", style='Title.TLabel')
        title.pack(pady=(0, 20))

        # Instructions
        instructions_text = (
            "Enter data points and interpolation value.\n"
            "Use comma-separated values for coordinates."
        )
        instructions = tk.Label(left_content, text=instructions_text, bg=self.panel_color,
                               fg="#666666", font=('Segoe UI', 9), justify=tk.LEFT, wraplength=350)
        instructions.pack(pady=(0, 20), anchor='w')

        # Input fields
        input_frame = ttk.Frame(left_content, style='Card.TFrame')
        input_frame.pack(fill='x', pady=5)

        # X Points
        x_label = ttk.Label(input_frame, text="X Coordinates:", style='Normal.TLabel')
        x_label.pack(anchor='w', pady=(10, 5))
        self.x_entry = tk.Entry(input_frame, font=('Segoe UI', 10), relief='solid', bd=1,
                               bg='white', fg=self.text_color, insertbackground=self.accent_color)
        self.x_entry.pack(fill='x', ipady=6, pady=(0, 5))
        self.x_entry.insert(0, "0, 1, 2, 3, 4")
        self._style_entry(self.x_entry)

        x_hint = tk.Label(input_frame, text="Example: 0, 1, 2, 3, 4", bg=self.panel_color,
                         fg="#999999", font=('Segoe UI', 8))
        x_hint.pack(anchor='w')

        # Y Points
        y_label = ttk.Label(input_frame, text="Y Coordinates:", style='Normal.TLabel')
        y_label.pack(anchor='w', pady=(15, 5))
        self.y_entry = tk.Entry(input_frame, font=('Segoe UI', 10), relief='solid', bd=1,
                               bg='white', fg=self.text_color, insertbackground=self.accent_color)
        self.y_entry.pack(fill='x', ipady=6, pady=(0, 5))
        self.y_entry.insert(0, "1, 3, 2, 5, 4")
        self._style_entry(self.y_entry)

        y_hint = tk.Label(input_frame, text="Example: 1, 3, 2, 5, 4", bg=self.panel_color,
                         fg="#999999", font=('Segoe UI', 8))
        y_hint.pack(anchor='w')

        # X Value to interpolate
        interp_label = ttk.Label(input_frame, text="X Value to Interpolate:", style='Normal.TLabel')
        interp_label.pack(anchor='w', pady=(15, 5))
        self.interp_entry = tk.Entry(input_frame, font=('Segoe UI', 10), relief='solid', bd=1,
                                     bg='white', fg=self.text_color, insertbackground=self.accent_color)
        self.interp_entry.pack(fill='x', ipady=6, pady=(0, 5))
        self.interp_entry.insert(0, "2.5")
        self._style_entry(self.interp_entry)

        interp_hint = tk.Label(input_frame, text="Example: 2.5", bg=self.panel_color,
                              fg="#999999", font=('Segoe UI', 8))
        interp_hint.pack(anchor='w')

        # Buttons
        button_frame = ttk.Frame(left_content, style='Card.TFrame')
        button_frame.pack(fill='x', pady=(25, 10))

        self.calc_btn = tk.Button(button_frame, text="Calculate", command=self.calculate,
                                  bg=self.accent_color, fg='white', font=('Segoe UI', 11, 'bold'),
                                  relief='flat', cursor='hand2', padx=20, pady=8, bd=0)
        self.calc_btn.pack(side='left', padx=(0, 10))
        self._style_button(self.calc_btn, self.accent_color)

        self.clear_btn = tk.Button(button_frame, text="Clear", command=self.clear_all,
                                   bg='#6C757D', fg='white', font=('Segoe UI', 11, 'bold'),
                                   relief='flat', cursor='hand2', padx=20, pady=8, bd=0)
        self.clear_btn.pack(side='left', padx=(0, 10))
        self._style_button(self.clear_btn, '#6C757D')

        self.example_btn = tk.Button(button_frame, text="Example", command=self.load_example,
                                     bg='#28A745', fg='white', font=('Segoe UI', 11, 'bold'),
                                     relief='flat', cursor='hand2', padx=20, pady=8, bd=0)
        self.example_btn.pack(side='left')
        self._style_button(self.example_btn, '#28A745')

        # RIGHT PANEL CONTENT
        right_content = ttk.Frame(right_panel, style='Card.TFrame', padding=20)
        right_content.pack(fill='both', expand=True)

        # Results Title
        results_title = ttk.Label(right_content, text="RESULTS PANEL", style='Title.TLabel')
        results_title.pack(pady=(0, 20))

        # Results Text Area with Windows 11 style
        text_frame = tk.Frame(right_content, bg=self.panel_color)
        text_frame.pack(fill='both', expand=True)

        # Custom scrollbar
        scrollbar = tk.Scrollbar(text_frame, bg=self.bg_color, troughcolor=self.panel_color,
                                activebackground=self.accent_color, bd=0, width=12)
        scrollbar.pack(side='right', fill='y')

        self.result_text = tk.Text(text_frame, font=('Consolas', 9), wrap=tk.WORD,
                                   relief='solid', bd=1, bg='#FAFAFA', fg=self.text_color,
                                   padx=15, pady=15, yscrollcommand=scrollbar.set)
        self.result_text.pack(fill='both', expand=True)
        scrollbar.config(command=self.result_text.yview)

        # Initial message
        initial_msg = (
            "═══════════════════════════════════════════════════\n"
            "       LAGRANGE INTERPOLATION CALCULATOR\n"
            "═══════════════════════════════════════════════════\n\n"
            "Welcome! Enter your data points and click Calculate\n"
            "to perform Lagrange interpolation.\n\n"
            "Results will appear here."
        )
        self.result_text.insert(1.0, initial_msg)
        self.result_text.config(state='disabled')

    def _add_shadow(self, widget):
        """Add subtle shadow effect to panels"""
        widget.configure(highlightbackground='#D0D0D0', highlightthickness=1)

    def _style_entry(self, entry):
        """Add focus effects to entry widgets"""
        def on_focus_in(e):
            entry.config(relief='solid', bd=2, highlightbackground=self.accent_color,
                        highlightthickness=1)

        def on_focus_out(e):
            entry.config(relief='solid', bd=1, highlightthickness=0)

        entry.bind('<FocusIn>', on_focus_in)
        entry.bind('<FocusOut>', on_focus_out)

    def _style_button(self, button, bg_color):
        """Add hover effects to buttons"""
        def on_enter(e):
            button.config(bg=self._darken_color(bg_color))

        def on_leave(e):
            button.config(bg=bg_color)

        button.bind('<Enter>', on_enter)
        button.bind('<Leave>', on_leave)

    def _darken_color(self, hex_color):
        """Darken a hex color for hover effect"""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        r, g, b = max(0, r-30), max(0, g-30), max(0, b-30)
        return f'#{r:02x}{g:02x}{b:02x}'

    def lagrange_interpolation(self, x_points, y_points, x_value):
        """Perform Lagrange interpolation using scipy"""
        poly = lagrange(x_points, y_points)
        result = poly(x_value)
        return result, poly

    def lagrange_from_scratch(self, x_points, y_points, x_value):
        """Perform Lagrange interpolation from scratch"""
        n = len(x_points)
        result = 0
        basis_values = []
        for i in range(n):
            L = 1.0
            for j in range(n):
                if i != j:
                    L *= (x_value - x_points[j]) / (x_points[i] - x_points[j])
            basis_values.append((i, L))
            result += y_points[i] * L
        return result, basis_values

    def calculate(self):
        """Main calculation function"""
        try:
            x_str = self.x_entry.get().strip()
            y_str = self.y_entry.get().strip()
            x_val_str = self.interp_entry.get().strip()

            if not x_str or not y_str or not x_val_str:
                messagebox.showwarning("Input Required", "Please fill in all fields!",
                                      parent=self.root)
                return

            x_points = [float(x.strip()) for x in x_str.split(',')]
            y_points = [float(y.strip()) for y in y_str.split(',')]
            x_val = float(x_val_str)

            if len(x_points) != len(y_points):
                messagebox.showerror("Error", "Number of X and Y points must match!",
                                   parent=self.root)
                return

            if len(x_points) < 2:
                messagebox.showerror("Error", "Need at least 2 data points!",
                                   parent=self.root)
                return

            if len(x_points) != len(set(x_points)):
                messagebox.showerror("Error", "X values must be unique!",
                                   parent=self.root)
                return

            result_scipy, poly = self.lagrange_interpolation(x_points, y_points, x_val)
            result_scratch, basis = self.lagrange_from_scratch(x_points, y_points, x_val)

            self.result_text.config(state='normal')
            self.result_text.delete(1.0, tk.END)

            output = "═══════════════════════════════════════════════════\n"
            output += "         LAGRANGE INTERPOLATION RESULTS\n"
            output += "═══════════════════════════════════════════════════\n\n"

            output += "INPUT DATA POINTS\n"
            output += "─────────────────────────────────────────────────────\n"
            for i in range(len(x_points)):
                output += f"  Point {i+1}: ({x_points[i]}, {y_points[i]})\n"

            output += "\n═══════════════════════════════════════════════════\n"
            output += f"INTERPOLATION AT x = {x_val}\n"
            output += "═══════════════════════════════════════════════════\n\n"

            output += f"  Interpolated Value: y = {result_scipy:.8f}\n\n"

            output += "POLYNOMIAL EQUATION\n"
            output += "─────────────────────────────────────────────────────\n"
            coeffs = poly.coef
            terms = []
            for i, coef in enumerate(coeffs):
                deg = len(coeffs) - 1 - i
                if abs(coef) > 1e-10:
                    if deg == 0:
                        terms.append(f"{coef:.4f}")
                    elif deg == 1:
                        terms.append(f"{coef:.4f}x")
                    else:
                        terms.append(f"{coef:.4f}x^{deg}")
            output += "P(x) = " + " + ".join(terms) + "\n\n"

            output += "POLYNOMIAL COEFFICIENTS\n"
            output += "─────────────────────────────────────────────────────\n"
            for i, coef in enumerate(coeffs):
                deg = len(coeffs) - 1 - i
                output += f"  x^{deg}: {coef:.10f}\n"

            output += "\n═══════════════════════════════════════════════════\n"
            output += f"BASIS POLYNOMIALS at x = {x_val}\n"
            output += "═══════════════════════════════════════════════════\n"
            for idx, L_val in basis:
                output += f"  L_{idx}({x_val}) = {L_val:.10f}\n"

            output += "\n═══════════════════════════════════════════════════\n"
            output += "VERIFICATION\n"
            output += "─────────────────────────────────────────────────────\n"
            output += f"  SciPy Method:   {result_scipy:.10f}\n"
            output += f"  Manual Method:  {result_scratch:.10f}\n"
            output += f"  Difference:     {abs(result_scipy - result_scratch):.2e}\n"

            output += "\n═══════════════════════════════════════════════════\n"
            output += "✓ Calculation completed successfully!\n"
            output += "═══════════════════════════════════════════════════\n"

            self.result_text.insert(1.0, output)
            self.result_text.config(state='disabled')

            messagebox.showinfo("Success", "Interpolation completed successfully!",
                              parent=self.root)

        except ValueError:
            messagebox.showerror("Input Error", 
                               "Invalid input. Please enter numbers only.",
                               parent=self.root)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n\n{str(e)}",
                               parent=self.root)

    def clear_all(self):
        """Clear all input and output fields"""
        self.x_entry.delete(0, tk.END)
        self.y_entry.delete(0, tk.END)
        self.interp_entry.delete(0, tk.END)
        self.result_text.config(state='normal')
        self.result_text.delete(1.0, tk.END)
        initial_msg = (
            "═══════════════════════════════════════════════════\n"
            "       LAGRANGE INTERPOLATION CALCULATOR\n"
            "═══════════════════════════════════════════════════\n\n"
            "Welcome! Enter your data points and click Calculate\n"
            "to perform Lagrange interpolation.\n\n"
            "Results will appear here."
        )
        self.result_text.insert(1.0, initial_msg)
        self.result_text.config(state='disabled')

    def load_example(self):
        """Load example data"""
        self.x_entry.delete(0, tk.END)
        self.y_entry.delete(0, tk.END)
        self.interp_entry.delete(0, tk.END)
        self.x_entry.insert(0, "0, 1, 2, 3, 4")
        self.y_entry.insert(0, "1, 3, 2, 5, 4")
        self.interp_entry.insert(0, "2.5")

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = LagrangeInterpolationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
