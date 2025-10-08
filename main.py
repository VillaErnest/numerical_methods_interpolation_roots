import subprocess
import sys
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
import json

COLOR_BG_MAIN = "#f8fafc"
COLOR_BG_CARD = "#ffffff"
COLOR_ACCENT = "#3b82f6"
COLOR_ACCENT_HOVER = "#2563eb"
COLOR_TEXT_DARK = "#1f2937"
COLOR_TEXT_MUTED = "#6b7280"
COLOR_BORDER = "#e5e7eb"
COLOR_ERROR = "#ef4444"

CONFIG_FILE = Path(__file__).parent / "config.json"

def load_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            config_data = json.load(f)
            scripts = config_data.get("scripts", [])
            if not isinstance(scripts, list):
                raise TypeError("The 'scripts' value in config.json must be a list.")
            return scripts
    except FileNotFoundError:
        messagebox.showerror("Configuration Error", f"Config file not found: {CONFIG_FILE}")
        return []
    except json.JSONDecodeError:
        messagebox.showerror("Configuration Error", "Error decoding config.json. Check file format for valid JSON.")
        return []
    except TypeError as e:
        messagebox.showerror("Configuration Error", str(e))
        return []
    except Exception as e:
        messagebox.showerror("Configuration Error", f"Failed to load config: {type(e).__name__}: {e}")
        return []

def run_program(script_name, root):
    if not script_name or not isinstance(script_name, str):
        messagebox.showerror("Validation Error", "Invalid script file name provided.")
        return

    script_path = Path(__file__).parent / "scripts" / script_name
    
    if not script_path.exists():
        messagebox.showerror("Execution Error", f"Script file not found: {script_name}")
        return

    try:
        root.withdraw()
        process = subprocess.Popen([sys.executable, str(script_path)])
        process.wait()
        
        if process.returncode != 0:
            messagebox.showwarning("Script Finished", f"The script '{script_name}' finished with a non-zero exit code: {process.returncode}")

    except OSError as e:
        messagebox.showerror("Execution Error", f"Failed to start process for {script_name}. Check if Python is correctly installed/pathed. {e}")
    except Exception as e:
        messagebox.showerror("Execution Error", f"An unexpected error occurred while running {script_name}: {e}")
    finally:
        root.deiconify()

def create_program_card(parent, script, root, scroll_callback):
    if not isinstance(script, dict) or 'name' not in script or 'file' not in script:
        card = tk.Frame(parent, bg=COLOR_BG_CARD, bd=1, highlightbackground=COLOR_ERROR, highlightthickness=2)
        tk.Label(card, text="Invalid Script Configuration", font=("Segoe UI", 12, "bold"), fg=COLOR_ERROR, bg=COLOR_BG_CARD).pack(padx=15, pady=15)
        card.pack(fill="x", padx=16, pady=10)
        scroll_callback(card)
        return

    card = tk.Frame(parent, bg=COLOR_BG_CARD, bd=1, relief="flat", highlightbackground=COLOR_BORDER, highlightthickness=1)
    card.pack(fill="x", padx=16, pady=10)

    inner_frame = tk.Frame(card, bg=COLOR_BG_CARD, padx=15, pady=15)
    inner_frame.pack(fill="x")

    title = tk.Label(inner_frame, text=script["name"], font=("Segoe UI", 16, "bold"), fg=COLOR_TEXT_DARK, bg=COLOR_BG_CARD)
    title.pack(anchor="w", pady=(0, 2))

    authors = "; ".join(script.get("authors", ["Unknown"]))
    author_label = tk.Label(inner_frame, text=f"By: {authors}", font=("Segoe UI", 10), fg=COLOR_TEXT_MUTED, bg=COLOR_BG_CARD,
                             wraplength=550, justify="left")
    author_label.pack(anchor="w", pady=(0, 8))

    desc = script.get("description", "")
    truncated = desc if len(desc) <= 100 else desc[:97] + "..."
    desc_label = tk.Label(inner_frame, text=truncated, font=("Segoe UI", 10), fg=COLOR_TEXT_DARK, bg=COLOR_BG_CARD, wraplength=550, justify="left")
    desc_label.pack(anchor="w", pady=(0, 10))

    run_button = tk.Button(inner_frame, text="Run Program", 
                            bg=COLOR_ACCENT, fg="white", 
                            activebackground=COLOR_ACCENT_HOVER, activeforeground="white",
                            relief="flat", bd=0, padx=15, pady=6, 
                            font=("Segoe UI", 10, "bold"),
                            command=lambda f=script["file"]: run_program(f, root))
    run_button.pack(anchor="e") 
    
    scroll_callback(card)

def main():
    root = tk.Tk()
    root.title("Numerical Methods Launcher")
    root.geometry("700x850")
    root.minsize(550, 400)
    root.configure(bg=COLOR_BG_MAIN)

    header = tk.Frame(root, bg=COLOR_ACCENT, height=120)
    header.pack(fill="x")

    tk.Label(header, text="Numerical Methods", font=("Segoe UI", 24, "bold"), bg=COLOR_ACCENT, fg="white").pack(pady=(15, 0))
    tk.Label(header, text="Interpolations and Roots", font=("Segoe UI", 16), bg=COLOR_ACCENT, fg="#e5e7eb").pack()
    tk.Label(header, text="Prepared by ECE 3A", font=("Segoe UI", 10), bg=COLOR_ACCENT, fg="#d1d5db").pack(pady=(0, 15))

    main_canvas = tk.Canvas(root, bg=COLOR_BG_MAIN, highlightthickness=0)
    main_canvas.pack(side="left", fill="both", expand=True)

    scrollbar = tk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
    scrollbar.pack(side="right", fill="y")

    main_canvas.configure(yscrollcommand=scrollbar.set)
    
    content_frame = tk.Frame(main_canvas, bg=COLOR_BG_MAIN)
    
    def on_content_configure(event):
        main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        main_canvas.itemconfigure(content_window, width=main_canvas.winfo_width())

    content_window = main_canvas.create_window((0, 0), window=content_frame, anchor="nw", width=root.winfo_width())
    
    content_frame.bind("<Configure>", on_content_configure)
    main_canvas.bind('<Configure>', on_content_configure)
    
    def on_mouse_wheel(event):
        if sys.platform.startswith('win') or sys.platform.startswith('linux'):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        elif sys.platform == 'darwin':
            main_canvas.yview_scroll(int(-event.delta), "units")

    def bind_scroll_recursive(widget):
        widget.bind("<MouseWheel>", on_mouse_wheel)
        for child in widget.winfo_children():
            bind_scroll_recursive(child)
            
    main_canvas.bind("<MouseWheel>", on_mouse_wheel)
    content_frame.bind("<MouseWheel>", on_mouse_wheel)
    
    scripts = load_config()
    
    if scripts:
        for script in scripts:
            create_program_card(content_frame, script, root, bind_scroll_recursive)
    else:
        tk.Label(content_frame, text="No scripts configured or configuration failed to load.", 
                 fg=COLOR_TEXT_MUTED, bg=COLOR_BG_MAIN, font=("Segoe UI", 12)).pack(pady=50)

    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        messagebox.showerror("Fatal Error", f"An unhandled error occurred: {type(e).__name__}: {e}")
        sys.exit(1)