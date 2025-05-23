import os
import sys
import tkinter as tk
import ttkbootstrap as ttk
from tkinter import scrolledtext

# Ajustar el path para importar desde el directorio raíz
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

class InfoTab:
    def __init__(self, parent):
        self.parent = parent
        self.setup_info_tab()

    def setup_info_tab(self):
        info_frame = ttk.Frame(self.parent)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        title_label = ttk.Label(info_frame, text="Análisis de Inclusiones de Polifosfatos", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        desc_text = scrolledtext.ScrolledText(info_frame, wrap=tk.WORD, height=20)
        desc_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        description = (
            "Esta aplicación automatiza el análisis de imágenes para detectar inclusiones de polifosfato en células bacterianas.\n\n"
            "Las condiciones de ensayo son las siguientes:\n"
            "     1. Aumentos: Tomar las imagenes con 1000x de aumento.\n"
            "     2. Densidad celular: La cantidad de células por campo debe ser de mayor de 100.\n"
        )
        desc_text.insert(tk.END, description)
        desc_text.config(state=tk.DISABLED)
