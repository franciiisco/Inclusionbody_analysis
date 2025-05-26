import os
import sys
import tkinter as tk
import ttkbootstrap as ttk
from tkinter import scrolledtext
import time

# Ajustar el path para importar desde el directorio raíz
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

class ProgressTracker:
    def __init__(self, status_text_widget, progress_bar_widget, time_estimate_label=None):
        """
        Initializes the ProgressTracker with existing UI widgets.
        :param status_text_widget: The ScrolledText widget for status messages.
        :param progress_bar_widget: The ttk.Progressbar widget.
        :param time_estimate_label: The ttk.Label widget for time estimation (optional).
        """
        self.status_text = status_text_widget
        self.progress_bar = progress_bar_widget
        self.time_estimate_label = time_estimate_label
        self.start_time = None
        self.total_items = 0
        self.processed_items = 0
    
    def start_tracking(self, total_items):
        """Initialize the progress tracking with the total number of items to process"""
        self.start_time = time.time()
        self.total_items = total_items
        self.processed_items = 0
        self.update_progress(0, f"Iniciando procesamiento de {total_items} imágenes...")
    
    def update_status(self, message, append=True, root=None):
        if self.status_text:
            self.status_text.config(state=tk.NORMAL)
            if not append:
                self.status_text.delete(1.0, tk.END)
            self.status_text.insert(tk.END, message + "\n")
            self.status_text.see(tk.END)  # Scroll to the end
            self.status_text.config(state=tk.DISABLED)
            if root:
                root.update_idletasks()

    def update_progress(self, value=None, detail=None, root=None):
        """
        Update progress bar and time estimate
        value: Either a percentage (0-100) or None to increment by 1 item
        detail: Optional status message to display
        """
        if detail:
            self.update_status(detail, root=root)
        
        if value is None and self.total_items > 0:
            # Increment processed items by 1
            self.processed_items += 1
            value = (self.processed_items / self.total_items) * 100
        
        if value is not None:
            progress_percent = min(float(value), 100.0)
            self.progress_bar["value"] = progress_percent
            
            # Update time estimate if we have start time and progress is between 1% and 99%
            if self.start_time and 0 < progress_percent < 100 and self.time_estimate_label:
                elapsed_time = time.time() - self.start_time
                if progress_percent > 0:
                    estimated_total_time = elapsed_time / (progress_percent / 100)
                    remaining_time = estimated_total_time - elapsed_time
                    
                    # Format the remaining time
                    if remaining_time > 3600:
                        time_str = f"{int(remaining_time // 3600)}h {int((remaining_time % 3600) // 60)}m"
                    elif remaining_time > 60:
                        time_str = f"{int(remaining_time // 60)}m {int(remaining_time % 60)}s"
                    else:
                        time_str = f"{int(remaining_time)}s"
                    
                    self.time_estimate_label.config(text=f"Tiempo restante: {time_str}")
            
            # When processing is complete
            elif progress_percent >= 100 and self.time_estimate_label:
                self.time_estimate_label.config(text="Procesamiento completado")
            
            if root:
                root.update_idletasks()
