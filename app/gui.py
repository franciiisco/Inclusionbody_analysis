"""
Interfaz gráfica moderna para el análisis de inclusiones de polifosfatos.
Utiliza ttkbootstrap para una apariencia más agradable y profesional.
"""
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
import threading
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import traceback

# Ajustar el path para importar desde el directorio raíz
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# Importar funciones de procesamiento
from src.core import batch_process, process_image_v2 as process_image
from config import PREPROCESS_CONFIG, SEGMENT_CONFIG, DETECTION_V2_CONFIG
from src.analysis import export_results_to_excel
from utils.file_operations import select_input_directory, select_output_directory, ensure_output_directory, detect_image_file_pattern
from utils.visualization import create_results_table, create_plots

# Importar componentes y tabs refactorizados
from app.components.progress_tracker import ProgressTracker
from app.components.results_viewer import ResultsViewer
from app.tabs.analysis_tab import AnalysisTab
from app.tabs.methodology_tab import MethodologyTab
from app.tabs.info_tab import InfoTab

class POLIP_Analyzer_GUI:    
    def __init__(self, root):
        self.root = root
        self.root.title("POLIP Analyzer")        
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Cargar el icono de la aplicación
        self.load_app_icon()
        # Variables de control
        self.input_dir = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value="")
        self.file_pattern = tk.StringVar(value="*.tif")
        self.processing_status = tk.StringVar(value="Listo para procesar")
        
        # Variables para el estado
        self.is_running = False
        self.progress_value = 0.0
        
        # Almacena las referencias a todas las imágenes
        self.images = {}
        
        # Crear la interfaz
        self.setup_ui()
        
        # Cargar el logo en la esquina inferior izquierda
        self.load_logo()
    
    def load_app_icon(self):
        """Cargar el icono de la aplicación"""
        # Buscar el icono en múltiples ubicaciones posibles
        icon_paths = [
            "data/logo/icon.ico",  # Ruta relativa desde el directorio de trabajo
            os.path.join(root_dir, "data", "logo", "icon.ico"),  # Ruta absoluta basada en root_dir
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "logo", "icon.ico")  # Ruta relativa al archivo actual
        ]
        
        for icon_path in icon_paths:
            if os.path.exists(icon_path):
                try:
                    self.root.iconbitmap(icon_path)
                    print(f"Icono de aplicación cargado desde: {icon_path}")
                    return
                except Exception as e:
                    print(f"Error al cargar el icono desde {icon_path}: {e}")
        
        print("No se encontró el archivo de icono. Agregue 'icon.ico' en 'data/logo/'")
    
    def load_logo(self):
        """Cargar y colocar el logo en la esquina inferior izquierda"""
        # Buscar el logo en múltiples ubicaciones posibles
        logo_paths = [
            "data/logo/cnta.jpg",  # Ruta relativa
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "cnta.jpg"),  # Ruta absoluta basada en la ubicación del script
            os.path.join(os.getcwd(), "cnta.jpg")  # Ruta basada en el directorio de trabajo actual
        ]
        
        for logo_path in logo_paths:
            if os.path.exists(logo_path):
                try:
                    # Cargar la imagen
                    img = Image.open(logo_path)
                    
                    # Redimensionar la imagen
                    logo_width = 220  # Tamaño adecuado
                    width_percent = (logo_width / float(img.size[0]))
                    logo_height = int((float(img.size[1]) * float(width_percent)))
                    img = img.resize((logo_width, logo_height), Image.LANCZOS)
                    
                    # Crear el PhotoImage y almacenarlo en el diccionario
                    self.images['logo'] = ImageTk.PhotoImage(img)
                    
                    # Crear una etiqueta para mostrar la imagen
                    logo_label = tk.Label(self.root, image=self.images['logo'])
                    # Posicionar en la esquina inferior izquierda usando place
                    logo_label.place(relx=0.035, rely=0.95, anchor="sw")
                    
                    print(f"Logo cargado y posicionado desde: {logo_path}")
                    return
                except Exception as e:
                    print(f"Error al cargar el logo desde {logo_path}: {e}")
        
        print("No se encontró el archivo de logo. Puede añadir un archivo 'cnta.jpg' en el directorio del script.")    
    
    def setup_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        main_tab = ttk.Frame(notebook)
        methodology_tab = ttk.Frame(notebook)
        info_tab = ttk.Frame(notebook)
        notebook.add(main_tab, text="Análisis")
        notebook.add(methodology_tab, text="Metodología")
        notebook.add(info_tab, text="Información")        # Integrar pestañas refactorizadas
        self.analysis_tab = AnalysisTab(main_tab, self.input_dir, self.output_dir, self.file_pattern, self._process_batch, self._view_aggregated_results)
        # Usar los widgets existentes en analysis_tab para crear el progress_tracker
        self.progress_tracker = ProgressTracker(
            self.analysis_tab.status_text, 
            self.analysis_tab.progress_bar,
            self.analysis_tab.time_estimate_label  # Pass the time estimate label
        )
        self.methodology_tab = MethodologyTab(methodology_tab)
        self.info_tab = InfoTab(info_tab)

    # ...existing code for processing, results, etc. (puede ser refactorizado a métodos auxiliares o componentes)...
    def update_status(self, message, append=True):
        """Delegate to progress_tracker.update_status"""
        self.analysis_tab.update_status(message, append)

    def update_progress(self, value=None, detail=None):
        """Actualiza la barra de progreso y opcionalmente añade un mensaje al área de estado"""
        if detail:
            self.update_status(detail)
        
        if value is not None:
            self.progress_value += value
            progress_percent = min(self.progress_value * 100, 100)
            self.analysis_tab.update_progress(progress_percent)
            
    def _browse_input_dir(self):
        directory = select_input_directory()
        if directory:
            self.input_dir.set(directory)
            if not self.output_dir.get():
                self.output_dir.set(os.path.join(directory, "processed"))
            pattern = detect_image_file_pattern(directory)
            if pattern:
                self.file_pattern.set(pattern)

    def _browse_output_dir(self):
        directory = select_output_directory()
        if directory:
            self.output_dir.set(directory)

    def _process_batch(self):
        """Iniciar el procesamiento por lotes en un hilo separado"""
        # Validar entradas
        if not self.input_dir.get():
            messagebox.showerror("Error", "Por favor seleccione un directorio de entrada")
            return
        
        # Crear el directorio de salida si no existe
        if not os.path.exists(self.output_dir.get()):
            try:
                os.makedirs(self.output_dir.get(), exist_ok=True)
            except Exception as e:
                messagebox.showerror("Error", f"No se pudo crear el directorio de salida: {e}")
                return
        
        self.is_running = True
        self.progress_value = 0.0
        self.analysis_tab.update_progress(0)
        
        # Obtener el estado del switch de imágenes intermedias
        save_images = self.analysis_tab.save_intermediate_images.get()
        images_status = "ACTIVADA" if save_images else "DESACTIVADA"
        
        # Actualizar la configuración global antes de comenzar el procesamiento
        from config import DETECTION_V2_CONFIG, VISUALIZATION_SETTINGS
        DETECTION_V2_CONFIG['debug']['save_intermediate_images'] = save_images
        VISUALIZATION_SETTINGS['save_intermediate_images'] = save_images
        
        # Desactivar también las visualizaciones emergentes cuando se ejecuta desde la GUI
        if not save_images:
            VISUALIZATION_SETTINGS['show_preprocessing_steps'] = False
            VISUALIZATION_SETTINGS['show_segmentation_results'] = False
            VISUALIZATION_SETTINGS['show_inclusion_detection'] = False
            VISUALIZATION_SETTINGS['show_summary_plots'] = False
        
        # Mostrar mensaje inicial
        self.update_status(
            f"Iniciando procesamiento por lotes...\n"
            f"Directorio de entrada: {self.input_dir.get()}\n"
            f"Patrón de archivos: {self.file_pattern.get()}\n"
            f"Directorio de salida: {self.output_dir.get()}\n"
            f"Generación de imágenes intermedias: {images_status}",
            append=False
        )
          # Ejecutar el procesamiento en un hilo separado
        processing_thread = threading.Thread(
            target=self._run_batch_processing,
            daemon=True
        )
        processing_thread.start()

    def _run_batch_processing(self):
        """Ejecuta el procesamiento en un hilo separado"""
        try:
            # Usar la función de procesamiento por lotes
            # Configurar la redirección de la salida para capturar los mensajes de progreso
            import io
            import sys
            from contextlib import redirect_stdout
            import os
            import glob
            
            # Get total number of files to process
            input_dir = self.input_dir.get()
            file_pattern = self.file_pattern.get()
            image_files = glob.glob(os.path.join(input_dir, file_pattern))
            total_files = len(image_files)
            
            # Initialize progress tracking
            self.root.after(0, lambda: self.progress_tracker.start_tracking(total_files))
            
            # Define a progress callback function for batch_process
            def progress_callback(file_index, filename):
                self.root.after(0, lambda: self.progress_tracker.update_progress(
                    None,  # Will increment by 1 item
                    f"Procesando imagen {file_index+1}/{total_files}: {os.path.basename(filename)}",
                    self.root
                ))
            
            # Capturar la salida estándar
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                # Ejecutar el procesamiento
                result = batch_process(
                    input_dir=self.input_dir.get(),
                    output_dir=self.output_dir.get(),
                    file_pattern=self.file_pattern.get(),
                    enforce_naming_convention=True,  # Siempre validar formato
                    save_intermediate_images=self.analysis_tab.save_intermediate_images.get(),  # Pasar el estado del switch
                    progress_callback=progress_callback  # Pass the progress callback
                )
            
            # Mostrar la salida capturada en el área de resultados
            captured_output = output_buffer.getvalue()
            
            # Exportar resultados a Excel
            try:
                self.root.after(0, lambda: self.update_status("\n\nGenerando archivo Excel con resultados..."))
                excel_path = export_results_to_excel(
                    result.get('image_results', []), 
                    self.output_dir.get(),
                    progress_reporter=self.update_progress
                )
                self.root.after(0, lambda: self.update_status(f"\nArchivo Excel generado correctamente: {excel_path}"))
            except Exception as e:
                error_excel = f"\nError al generar archivo Excel: {e}"
                self.root.after(0, lambda: self.update_status(error_excel))
                print(f"Error al exportar a Excel: {str(e)}\n{traceback.format_exc()}")
            
            # Actualizar la interfaz en el hilo principal
            self.root.after(0, self._update_results, captured_output, result)
            
        except Exception as e:
            error_msg = f"Error durante el procesamiento: {e}\n{traceback.format_exc()}"
            self.root.after(0, self._show_error, error_msg)
        finally:
            # Habilitar botones al terminar - FIXME: Need to get button references from analysis_tab
            # self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            # self.root.after(0, lambda: self.results_btn.config(state=tk.NORMAL))
            self.is_running = False
    def _update_results(self, output_text, result):
        """Actualizar la interfaz con los resultados del procesamiento"""
        self.update_status(output_text)
        self.update_status("\n\nResumen:")
        self.update_status(f"Archivos procesados: {result['processed_files']}")
        self.update_status(f"Archivos omitidos: {result['skipped_files']}")
          # Establecer progreso al 100%
        self.analysis_tab.update_progress(100)
          # Mensaje de finalización
        self.processing_status.set("Procesamiento completado")
        
        # Determinar si se generaron imágenes intermedias
        images_note = ""
        if self.analysis_tab.save_intermediate_images.get():
            images_note = "\n\nSe han generado imágenes intermedias de cada paso del análisis."
          # Mostrar un mensaje de éxito
        messagebox.showinfo("Procesamiento completado", 
                           f"Se han procesado {result['processed_files']} imágenes.\n"
                           f"Los resultados están disponibles en:\n{self.output_dir.get()}\n\n"
                           f"Se ha generado un archivo Excel con el análisis completo.{images_note}")

    def _show_error(self, error_msg):
        """Mostrar un mensaje de error en la interfaz"""
        self.update_status(error_msg, append=False)
        self.processing_status.set("Error en el procesamiento")
        messagebox.showerror("Error", "Se ha producido un error durante el procesamiento")

    def _view_aggregated_results(self):
        """Mostrar los resultados agregados en una ventana separada"""
        # Verificar si existen resultados agregados en la nueva ubicación
        aggregated_file = os.path.join(
            self.output_dir.get(), 
            "Datos_crudos",
            "aggregated_results_v2.json"
        )
        
        if not os.path.exists(aggregated_file):
            messagebox.showerror("Error", "No se encontraron resultados agregados")
            return
        
        try:
            # Cargar los resultados agregados
            with open(aggregated_file, 'r') as f:
                data = json.load(f)
            
            # Mostrar una ventana con los resultados
            results_window = ttk.Toplevel(self.root)
            results_window.title("Resultados Agregados")
            results_window.geometry("900x600")
            
            # Crear notebook para las distintas vistas
            notebook = ttk.Notebook(results_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Pestaña para los resultados por CONDICIÓN/TIEMPO
            ct_frame = ttk.Frame(notebook)
            notebook.add(ct_frame, text="Por Condición/Tiempo")
            
            # Mostrar tabla con resultados
            ct_data = data.get('condition_time', {})
            self._create_results_table(ct_frame, ct_data, "Condición/Tiempo")
            
            # Pestaña para los resultados por CONDICIÓN/TIEMPO/REPLICA
            ctr_frame = ttk.Frame(notebook)
            notebook.add(ctr_frame, text="Por Condición/Tiempo/Réplica")
            
            # Mostrar tabla con resultados
            ctr_data = data.get('condition_time_replicate', {})
            self._create_results_table(ctr_frame, ctr_data, "Condición/Tiempo/Réplica")
            
            # Pestaña para gráficas
            plot_frame = ttk.Frame(notebook)
            notebook.add(plot_frame, text="Gráficas")
            
            # Crear gráficas básicas de las métricas principales
            self._create_plots(plot_frame, data)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al cargar los resultados: {str(e)}")
            
    def on_close(self):
        """Maneja el evento de cierre de la ventana"""
        if self.is_running:
            if messagebox.askyesno("Confirmar salida", "El procesamiento está en curso. ¿Seguro que desea salir?"):
                self.root.destroy()
        else:
            self.root.destroy()
    
    def _create_results_table(self, parent, data, level_name):
        """Crear una tabla para mostrar los resultados agregados"""
        create_results_table(parent, data, level_name)

    def _create_plots(self, parent, data):
        """Crear gráficos con los resultados agregados"""
        create_plots(parent, data)
    
    def setup_methodology_tab(self, parent):
        # Ahora la lógica y descripciones están en MethodologyTab
        pass

    def _update_step_description(self, event):
        pass

    def _get_step_descriptions(self):
        pass
def main():
    # Crear la ventana principal con ttkbootstrap
    root = ttk.Window(
        title="Analizador de Inclusiones de Polifosfatos",
        themename="cosmo",  # Temas: cosmo, flatly, journal, litera, lumen, etc.
        size=(1000, 700),
        minsize=(800, 600),
        resizable=(True, True),
    )
    
    # Intentar cargar el icono directamente en la ventana principal también
    icon_paths = [
        "data/logo/icon.ico",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "logo", "icon.ico")
    ]
    
    for icon_path in icon_paths:
        if os.path.exists(icon_path):
            try:
                root.iconbitmap(icon_path)
                print(f"Icono de aplicación cargado en ventana principal desde: {icon_path}")
                break
            except Exception as e:
                print(f"Error al cargar icono en ventana principal: {e}")
    
    # Crear la aplicación
    app = POLIP_Analyzer_GUI(root)
    
    # Iniciar bucle principal
    root.mainloop()


if __name__ == "__main__":
    main()