import os
import sys
import tkinter as tk
import ttkbootstrap as ttk
from tkinter import scrolledtext

# Ajustar el path para importar desde el directorio raíz
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

class MethodologyTab:
    def __init__(self, parent):
        self.parent = parent
        self.setup_methodology_tab()

    def setup_methodology_tab(self):
        main_frame = ttk.Frame(self.parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        steps_frame = ttk.LabelFrame(main_frame, text="Pipeline de Análisis")
        steps_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5, ipadx=5, ipady=5)
        steps_frame.config(width=200)
        description_frame = ttk.LabelFrame(main_frame, text="Descripción Detallada")
        description_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.steps_list = tk.Listbox(steps_frame, selectmode=tk.SINGLE, font=("Segoe UI", 10), activestyle="dotbox", width=25)
        self.steps_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        steps = [
            "1. Preprocesamiento de imágenes",
            "2. Segmentación de células",
            "3. Detección de inclusiones",
            "4. Extracción de características",
            "5. Análisis estadístico",
            "6. Visualización de resultados"
        ]
        for step in steps:
            self.steps_list.insert(tk.END, step)
        self.description_text = scrolledtext.ScrolledText(description_frame, wrap=tk.WORD, font=("Segoe UI", 10))
        self.description_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.steps_list.bind('<<ListboxSelect>>', self.update_step_description)
        self.steps_list.selection_set(0)
        self.update_step_description(None)

    def update_step_description(self, event):
        try:
            selected_idx = self.steps_list.curselection()[0]
        except IndexError:
            selected_idx = 0
            self.steps_list.selection_set(0)
        self.description_text.config(state=tk.NORMAL)
        self.description_text.delete(1.0, tk.END)
        descriptions = self.get_step_descriptions()        
        self.description_text.insert(tk.END, descriptions[selected_idx])
        self.description_text.see(1.0)
        self.description_text.config(state=tk.DISABLED)
        
    def get_step_descriptions(self):
        """Devuelve las descripciones detalladas de cada paso del pipeline leyéndolas desde archivos"""
        descriptions = []
        
        # Definir los archivos para cada paso
        description_files = [
            "preprocess.txt",          # 1. Preprocesamiento de imágenes
            "segmentation.txt",        # 2. Segmentación de células
            "detection.txt",           # 3. Detección de inclusiones
            "analysis.txt",            # 4. Extracción de características
            "analysis.txt",            # 5. Análisis estadístico
            "visualization.txt"        # 6. Visualización de resultados
        ]
        
        # Obtener el directorio de las descripciones
        description_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "descriptions")
        
        # Leer cada archivo y agregar su contenido a las descripciones
        for file in description_files:
            file_path = os.path.join(description_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    descriptions.append(f.read())
            except Exception as e:
                descriptions.append(f"Error al cargar el archivo de descripción '{file}': {str(e)}")
        
        return descriptions
