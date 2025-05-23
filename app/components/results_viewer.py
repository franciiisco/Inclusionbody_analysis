import os
import sys
import tkinter as tk
import ttkbootstrap as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# Ajustar el path para importar desde el directorio raíz
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

class ResultsViewer:
    def __init__(self, parent):
        self.parent = parent

    def create_results_table(self, data, level_name):
        frame = ttk.Frame(self.parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        columns = ('grupo', 'total_images', 'mean_avg_inclusions_per_cell', 
                  'std_avg_inclusions_per_cell', 'mean_avg_inclusion_ratio', 
                  'std_avg_inclusion_ratio')
        tree = ttk.Treeview(frame, columns=columns, show='headings', bootstyle='primary')
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scrollbar.set)
        tree.heading('grupo', text=level_name)
        tree.heading('total_images', text='Imágenes')
        tree.heading('mean_avg_inclusions_per_cell', text='Media Incl./Célula')
        tree.heading('std_avg_inclusions_per_cell', text='DE Incl./Célula')
        tree.heading('mean_avg_inclusion_ratio', text='Media Ratio Incl.')
        tree.heading('std_avg_inclusion_ratio', text='DE Ratio Incl.')
        tree.column('grupo', width=200)
        tree.column('total_images', width=70)
        tree.column('mean_avg_inclusions_per_cell', width=120)
        tree.column('std_avg_inclusions_per_cell', width=120)
        tree.column('mean_avg_inclusion_ratio', width=120)
        tree.column('std_avg_inclusion_ratio', width=120)
        for group, stats in data.items():
            tree.insert('', tk.END, values=(
                group,
                stats.get('total_images', 0),
                f"{stats.get('mean_avg_inclusions_per_cell', 0):.2f}",
                f"{stats.get('std_avg_inclusions_per_cell', 0):.2f}",
                f"{stats.get('mean_avg_inclusion_ratio', 0)*100:.2f}%",
                f"{stats.get('std_avg_inclusion_ratio', 0)*100:.2f}%"
            ))

    def create_plots(self, data):
        ct_data = data.get('condition_time', {})
        if not ct_data:
            ttk.Label(self.parent, text="No hay datos suficientes para generar gráficos").pack()
            return
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.Figure(figsize=(10, 8), dpi=100)
        conditions = {}
        for key in ct_data.keys():
            condition, time = key.split('/')
            if condition not in conditions:
                conditions[condition] = []
            conditions[condition].append((time, key))
        for condition in conditions:
            conditions[condition].sort(key=lambda x: x[0])
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        for i, (condition, times) in enumerate(conditions.items()):
            color = colors[i % len(colors)]
            x_values = [time for time, _ in times]
            y_values = [ct_data[key]['mean_avg_inclusions_per_cell'] for _, key in times]
            yerr_values = [ct_data[key]['std_avg_inclusions_per_cell'] for _, key in times]
            ax1.errorbar(x_values, y_values, yerr=yerr_values, marker='o', label=condition, color=color, capsize=4)
        ax1.set_title("Promedio de inclusiones por célula", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Tiempo", fontsize=10)
        ax1.set_ylabel("Inclusiones por célula", fontsize=10)
        ax1.legend(fontsize=9)
        ax1.grid(True, linestyle='--', alpha=0.7)
        for i, (condition, times) in enumerate(conditions.items()):
            color = colors[i % len(colors)]
            x_values = [time for time, _ in times]
            y_values = [ct_data[key]['mean_avg_inclusion_ratio']*100 for _, key in times]
            yerr_values = [ct_data[key]['std_avg_inclusion_ratio']*100 for _, key in times]
            ax2.errorbar(x_values, y_values, yerr=yerr_values, marker='o', label=condition, color=color, capsize=4)
        ax2.set_title("Ratio promedio de área de inclusiones", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Tiempo", fontsize=10)
        ax2.set_ylabel("Ratio de inclusiones (%)", fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=self.parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
