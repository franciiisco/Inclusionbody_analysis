import os
from tkinter import filedialog

# File handling functions

def select_input_directory():
    """Solicita al usuario seleccionar un directorio de entrada."""
    return filedialog.askdirectory(title="Seleccionar directorio de imágenes")

def select_output_directory():
    """Solicita al usuario seleccionar un directorio de salida."""
    return filedialog.askdirectory(title="Seleccionar directorio de salida")

def ensure_output_directory(directory):
    """Crea el directorio de salida si no existe."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def detect_image_file_pattern(directory):
    """Detecta las extensiones de imagen presentes en el directorio y construye el patrón de archivos."""
    image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']
    extension_count = {ext: 0 for ext in image_extensions}
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(filename)
            ext = ext.lower()
            if ext in extension_count:
                extension_count[ext] += 1
    found_extensions = [ext for ext, count in extension_count.items() if count > 0]
    if not found_extensions:
        return None
    if len(found_extensions) == 1:
        return f"*{found_extensions[0]}"
    else:
        return f"*.{{{','.join(ext.strip('.') for ext in found_extensions)}}}"