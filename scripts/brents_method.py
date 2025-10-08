import numpy as np
import matplotlib.pyplot as plt
import re

class BrentsCalculator:
    def __init__(self):
        self.history = []

    def parse_function(self, func_str):
        """Parse user input string into a callable function f(x)."""
        # Replace ^ with **
        func_str = func_str.replace('^', '**')

        # Insert multiplication where omitted
        func_str = re.sub(r'(\d)(x)', r'\1*\2', func_str)
        func_str = re.sub(r'(x)(\d)', r'\1*\2', func_str)
        func_str = re.sub(r'(x)(\()', r'\1*\2', func_str)
        func_str = re.sub(r'(\))(x)', r'\1*\2', func_str)

        # Allowed functions/constants
        allowed = {
            'sin': np.sin, 'cos': np.cos, 'tan': np.tan,
            'exp': np.exp, 'log': np.log, 'ln': np.log,
            'log10': np.log10, 'sqrt': np.sqrt,
            'pi': np.pi, 'e': np.e, 'abs': np.abs
        }

        try:
            return eval(f"lambda x: {func_str}", allowed, allowed)
        except Exception as e:
            raise ValueError(f"Invalid function: {e}")

    def brent_method(self, f, a, b, tol, max_iter):
        """Brent’s method implementation."""
        fa, fb = f(a), f(b)
        if fa * fb >= 0:
            return None, "f(a) and f(b) must have opposite signs."

        if abs(fa) < abs(fb): a, b, fa, fb = b, a, fb, fa
        c, fc = a, fa
        d = e = b - a
        self.history = []

        print("\nIter |       a        b        s     f(s)       error")
        print("-" * 60)

        for i in range(1, max_iter + 1):
            if fb == 0: return b, "Exact root found"

            # Inverse quadratic interpolation or secant method
            if fa != fc and fb != fc:
                s = (a*fb*fc)/((fa-fb)*(fa-fc)) \
                  + (b*fa*fc)/((fb-fa)*(fb-fc)) \
                  + (c*fa*fb)/((fc-fa)*(fc-fb))
            else:
                s = b - fb * (b - a) / (fb - fa)

            cond1 = not ((3*a + b)/4 < s < b)
            cond2 = (e < tol) or (abs(s-b) >= abs(e)/2)
            cond3 = (abs(b-c) < tol) or (abs(c-d) < tol)
            if cond1 or cond2 or cond3:
                s = (a + b) / 2
                d = e = b - a
            else:
                d, e = e, b - s

            fs = f(s)
            self.history.append({'iteration': i, 'a': a, 'b': b, 's': s,
                                 'f(s)': fs, 'error': abs(b-a)})

            print(f"{i:3d} | {a:8.6f} {b:8.6f} {s:8.6f} {fs:10.2e} {abs(b-a):10.2e}")

            c, fc = b, fb
            if fa * fs < 0: b, fb = s, fs
            else: a, fa = s, fs
            if abs(fa) < abs(fb): a, b, fa, fb = b, a, fb, fa
            if abs(b - a) < tol: return b, "Converged"

        return b, "Max iterations reached"

    def plot_results(self, f, a, b, root):
        """Plot function and Brent’s method iterations."""
        if not self.history: return
        x = np.linspace(a, b, 1000)
        plt.plot(x, [f(val) for val in x], 'b-', label='f(x)')
        plt.axhline(0, color='k', ls='--', alpha=0.5)
        plt.axvline(root, color='r', ls='--', label=f'Root ≈ {root:.6f}')
        s_vals = [d['s'] for d in self.history]
        plt.plot(s_vals, [f(s) for s in s_vals], 'ro-', label="Iterations")
        plt.legend(); plt.grid(alpha=0.3)
        plt.title("Brent's Method Convergence")
        plt.show()


def main():
    calc = BrentsCalculator()
    print("="*60)
    print("BRENT'S METHOD CALCULATOR")
    print("="*60)

    while True:
        func_str = input("\nEnter function f(x) (or 'quit'): ").strip()
        if func_str.lower() == 'quit': break

        try:
            a = float(input("Left endpoint a: "))
            b = float(input("Right endpoint b: "))
            tol = float(input("Tolerance (e.g., 1e-6): "))
            max_iter = int(input("Max iterations: "))

            f = calc.parse_function(func_str)
            root, msg = calc.brent_method(f, a, b, tol, max_iter)

            if root is not None:
                print(f"\n🎯 Root ≈ {root:.10f}, f(root) = {f(root):.2e} ({msg})")
                calc.plot_results(f, a, b, root)
            else:
                print(f"\n❌ {msg}")

        except Exception as e:
            print(f"❌ Error: {e}")

        if input("\nAnother? (y/n): ").lower() != 'y': break
    print("\nDone.")


if __name__ == "__main__":
    main()
