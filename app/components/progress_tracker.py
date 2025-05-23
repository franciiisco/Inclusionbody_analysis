import os
import sys
import tkinter as tk
import ttkbootstrap as ttk
from tkinter import scrolledtext

# Ajustar el path para importar desde el directorio raíz
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

class ProgressTracker:
    def __init__(self, status_text_widget, progress_bar_widget):
        """
        Initializes the ProgressTracker with existing UI widgets.
        :param status_text_widget: The ScrolledText widget for status messages.
        :param progress_bar_widget: The ttk.Progressbar widget.
        """
        self.status_text = status_text_widget
        self.progress_bar = progress_bar_widget

    def update_status(self, message, append=True, root=None):
        self.status_text.config(state=tk.NORMAL)
        if not append:
            self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)
        if root:
            root.update_idletasks()

    def update_progress(self, value=None, detail=None, root=None):
        if detail:
            self.update_status(detail, root=root)
        if value is not None:
            progress_percent = min(float(value), 100.0)
            self.progress_bar["value"] = progress_percent
            if root:
                root.update_idletasks()
