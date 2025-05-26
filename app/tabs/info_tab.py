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
            "     2. Densidad celular: La cantidad de células por campo debe ser de mayor de 100.\n" \
            "        Hemos puesto a punto mediante una curva de crecimiento la relación entre OD600 y el número de celulas por campo (microscopio)\n\n"
            "        Con esta ecuacion puedes estimar la cantidad de celulas por campo en función de la OD600 (x) en el rango de 10-1400 celulas:\n"
            "        y = 604,6x - 49,32\n\n"
            "        Cómo idea general, la OD600 de 0.246 corresponde a 100 células por campo.\n"
            "        Mientras que la OD600 de 0.577 corresponde a 300 células por campo.\n\n"
            "        Estos valores sirven como referencia, aunque las imagenes pueden contener hasta 3000 células por campo. El algoritmo puede adaptarse sin problemas.\n\n"
            "     3. Iluminación: La iluminación debe ser ni muy brillante, ni muy oscura. El programa implementa varios algoritmos que normalizan la iluminación de las imágenes para mejorar la segmentación, aunque siempre son buenas practicas el intentar mantener una iluminación uniforme entre imagenes.\n"
            "        Si estás trabajando en el microscopio invertido de CNTA, en el regulador de la lampara de la luz (abajo a la izquierda) hay una marquita que indica la posición óptima de la luz.\n\n"
            "     4. Fijación: La fijación de las células al portaobjetos se realiza mediante el tradicional metodo de frotis a la llama.\n"
            "        Se recomienda que las células estén bien fijadas, ya que si se mueven durante la toma de la imagen, el algoritmo puede no funcionar correctamente.\n\n"
            "     5. Toma de imágenes: Las imágenes deben tomarse con un microscopio óptico con un objetivo de 1000x de aumento. Es importante que las imagenes sean en blanco y negro (no en color), ya que el algoritmo está diseñado para trabajar con imágenes en escala de grises.\n\n"
            "     6. Tinción: La observacion de polifosfatos se realiza mediante tincion de NEISSER.\n\n"
            "REFERENCIAS BIBLIOGRÁFICAS - ALGORITMOS DE ANÁLISIS DE IMAGEN:\n\n"
            "Preprocesamiento y Normalización:\n"
            "• Gonzalez, R.C., Woods, R.E. (2018). Digital Image Processing, 4th Edition. Pearson.\n"
            "• Pizer, S.M., et al. (1987). Adaptive histogram equalization and its variations. Computer Vision, Graphics, and Image Processing, 39(3), 355-368.\n\n"
            "Segmentación:\n"
            "• Otsu, N. (1979). A threshold selection method from gray-level histograms. IEEE Transactions on Systems, Man, and Cybernetics, 9(1), 62-66.\n"
            "• Meyer, F. (1994). Topographic distance and watershed lines. Signal Processing, 38(1), 113-125.\n"
            "• Li, C.H., Lee, C.K. (1993). Minimum cross entropy thresholding. Pattern Recognition, 26(4), 617-625.\n\n"
            "Detección de Características:\n"
            "• Suzuki, S., Abe, K. (1985). Topological structural analysis of digitized binary images by border following. Computer Vision, Graphics, and Image Processing, 30(1), 32-46.\n"
            "• Hu, M.K. (1962). Visual pattern recognition by moment invariants. IRE Transactions on Information Theory, 8(2), 179-187.\n\n"
            "Morfología Matemática:\n"
            "• Serra, J. (1982). Image Analysis and Mathematical Morphology. Academic Press.\n"
            "• Soille, P. (2003). Morphological Image Analysis: Principles and Applications, 2nd Edition. Springer.\n\n"
            "Análisis de Conectividad:\n"
            "• Rosenfeld, A., Pfaltz, J.L. (1966). Sequential operations in digital picture processing. Journal of the ACM, 13(4), 471-494.\n"
            "• He, L., et al. (2017). A run-based two-scan labeling algorithm. IEEE Transactions on Image Processing, 16(10), 2494-2505.\n\n\n\n\n\n\n\n\n\n"
        )
        desc_text.insert(tk.END, description)
        desc_text.config(state=tk.DISABLED)
