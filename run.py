"""
Script principal para iniciar la aplicación POLIP Analyzer.
Ejecuta este script desde el directorio raíz del proyecto.
"""
import os
import sys

# Asegurarse de que los imports funcionen correctamente
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Importar la función principal de la GUI
from app.gui import main

if __name__ == "__main__":
    main()
