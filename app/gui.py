"""
Interfaz gráfica moderna para el análisis de inclusiones de polifosfatos.
Utiliza ttkbootstrap para una apariencia más agradable y profesional.
"""
import os
import tkinter as tk
# Reemplazar ttk por ttkbootstrap
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox, scrolledtext
import threading
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import traceback

# Importar funciones de procesamiento
from main import batch_process, process_image_v2 as process_image
from config import PREPROCESS_CONFIG, SEGMENT_CONFIG, DETECTION_V2_CONFIG
from src.analysis import export_results_to_excel

class POLIP_Analyzer_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("POLIP Analyzer")
        self.root.geometry("1000x700")
        self.root.minsize(800, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
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
        # Crear un notebook (pestañas)
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Crear pestañas
        main_tab = ttk.Frame(notebook)
        methodology_tab = ttk.Frame(notebook)
        info_tab = ttk.Frame(notebook)
        notebook.add(main_tab, text="Análisis")
        notebook.add(methodology_tab, text="Metodología")
        notebook.add(info_tab, text="Información")
        
        # Contenido de la pestaña principal (main_tab)
        self.setup_main_tab(main_tab)
        
        # Contenido de la pestaña de metodología (methodology_tab)
        self.setup_methodology_tab(methodology_tab)
        
        # Contenido de la pestaña de información (info_tab)
        self.setup_info_tab(info_tab)

    def setup_main_tab(self, parent):
        # Frame principal dividido en dos columnas
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # División en dos columnas: izquierda para opciones, derecha para salida
        options_frame = ttk.LabelFrame(main_frame, text="Configuración")
        options_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        output_frame = ttk.LabelFrame(main_frame, text="Estado del Análisis")
        output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Opciones (izquierda)
        self.setup_options_frame(options_frame)
        
        # Salida (derecha)
        self.setup_output_frame(output_frame)

    def setup_options_frame(self, parent):
        # Sección 1: Directorios y archivos
        file_section = ttk.LabelFrame(parent, text="1. Seleccionar Directorios")
        file_section.pack(fill=tk.X, padx=5, pady=5)
        
        # Directorio de entrada
        ttk.Label(file_section, text="Directorio de imágenes:").pack(anchor='w', padx=5, pady=2)
        input_frame = ttk.Frame(file_section)
        input_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_dir)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        input_btn = ttk.Button(input_frame, text="Examinar", command=self._browse_input_dir, bootstyle=INFO)
        input_btn.pack(side=tk.RIGHT)
        
        # Directorio de salida
        ttk.Label(file_section, text="Directorio de salida:").pack(anchor='w', padx=5, pady=2)
        output_frame = ttk.Frame(file_section)
        output_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_dir)
        self.output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        output_btn = ttk.Button(output_frame, text="Examinar", command=self._browse_output_dir, bootstyle=INFO)
        output_btn.pack(side=tk.RIGHT)
        
        # Nota sobre formato de nombre
        format_section = ttk.LabelFrame(parent, text="2. Formato de Archivo")
        format_section.pack(fill=tk.X, padx=5, pady=5)
        
        format_note = ttk.Label(
            format_section, 
            text="Los archivos deben seguir el formato:\nCONDICION_BOTE_REPLICA_TIEMPO_Nº\nEjemplo: MEI_B1_R3_t4_026_BF1.png",
            justify='left'
        )
        format_note.pack(padx=10, pady=5, anchor='w')
        
        # Sección 3: Botones de acción
        action_section = ttk.LabelFrame(parent, text="3. Acciones")
        action_section.pack(fill=tk.X, padx=5, pady=5)
        
        # Botón de procesar lote
        self.process_btn = ttk.Button(
            action_section, 
            text="Analizar", 
            command=self._process_batch, 
            bootstyle=SUCCESS
        )
        self.process_btn.pack(padx=5, pady=10, fill=tk.X)
        
        # Botón para ver resultados agregados
        self.results_btn = ttk.Button(
            action_section,
            text="Ver Resultados",
            command=self._view_aggregated_results,
            bootstyle=PRIMARY
        )
        self.results_btn.pack(padx=5, pady=10, fill=tk.X)
        
        # Sección 4: Información de contacto
        contact_section = ttk.LabelFrame(parent, text="Desarrollado por")
        contact_section.pack(fill=tk.X, padx=5, pady=5)
        
        # Panel para información y logo en la misma fila
        contact_panel = ttk.Frame(contact_section)
        contact_panel.pack(fill=tk.X, padx=5, pady=5)
        
        # Información de contacto
        contact_info = "Francisco Márquez Urbano\nfmarquez@cnta.es"
        contact_label = ttk.Label(contact_panel, text=contact_info, font=("Segoe UI", 9))
        contact_label.pack(padx=5, pady=5)

    def setup_output_frame(self, parent):
        # Área de texto para los mensajes de estado
        self.status_text = scrolledtext.ScrolledText(parent, wrap=tk.WORD)
        self.status_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.status_text.insert(tk.END, "Listo para analizar. Configure las opciones y haga clic en 'Analizar'. \n\nEl análisis puede tardar varios minutos, por favor sea paciente. Déjeme trabajando, usted puede hacer otras cosas mientras.")
        self.status_text.config(state=tk.DISABLED)
        
        # Barra de progreso
        progress_frame = ttk.Frame(parent)
        progress_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(progress_frame, text="Progreso:").pack(side=tk.LEFT)
        
        self.progress_bar = ttk.Progressbar(progress_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.progress_bar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(5, 0))

    def setup_info_tab(self, parent):
        # Información detallada de la aplicación
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        title_label = ttk.Label(info_frame, text="Análisis de Inclusiones de Polifosfatos", font=("Arial", 14, "bold"))
        title_label.pack(pady=10)
        
        # Descripción
        desc_text = scrolledtext.ScrolledText(info_frame, wrap=tk.WORD, height=20)
        desc_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        description = """
Esta aplicación automatiza el análisis de imágenes para detectar inclusiones de polifosfato en células bacterianas.

El proceso implica:
• Carga de imágenes de microscopía
• Preprocesamiento para mejorar la calidad de la imagen
• Segmentación para identificar células individuales
• Detección de inclusiones de polifosfato dentro de las células
• Generación de estadísticas detalladas
• Agregación de datos por condición/tiempo/réplica

El algoritmo implementa técnicas avanzadas para la detección de inclusiones con:
• Separación mejorada de inclusiones cercanas
• Ajuste automático de umbral según el contexto celular
• Filtrado por forma y tamaño para reducir falsos positivos
• Análisis estadísticos personalizados

Pasos para el uso:
1. Seleccione el directorio que contiene las imágenes a analizar
2. Especifique el patrón de archivos a procesar (e.g., *.tif, *.png)
3. Seleccione el directorio de salida para guardar los resultados
4. Haga clic en "Analizar" para iniciar el procesamiento
5. Una vez completado, puede ver los resultados agregados

Formato de nombre de archivo recomendado: 
CONDICION_BOTE_REPLICA_TIEMPO_NºIMAGEN
Ejemplo: Control_1_A_T0_1.tif

Las estadísticas y resultados se generan automáticamente y se guardan en el directorio de salida.
"""
        
        desc_text.insert(tk.END, description)
        desc_text.config(state=tk.DISABLED)
    def update_status(self, message, append=True):
        self.status_text.config(state=tk.NORMAL)
        if not append:
            self.status_text.delete(1.0, tk.END)
        self.status_text.insert(tk.END, f"\n{message}")
        self.status_text.see(tk.END)  # Auto-scroll
        self.status_text.config(state=tk.DISABLED)
        self.root.update_idletasks()

    def update_progress(self, value=None, detail=None):
        """Actualiza la barra de progreso y opcionalmente añade un mensaje al área de estado"""
        if detail:
            self.update_status(detail)
        
        if value is not None:
            self.progress_value += value
            progress_percent = min(self.progress_value * 100, 100)
            self.progress_bar["value"] = progress_percent
            self.root.update_idletasks()    
            
    def _browse_input_dir(self):
        """Solicitar al usuario el directorio de entrada"""
        directory = filedialog.askdirectory(title="Seleccionar directorio de imágenes")
        if directory:
            self.input_dir.set(directory)
            # Si no se ha seleccionado un directorio de salida, proponer uno
            if not self.output_dir.get():
                self.output_dir.set(os.path.join(directory, "processed"))
            
            # Detectar automáticamente las extensiones de imagen en el directorio
            self._update_file_pattern(directory)

    def _browse_output_dir(self):
        """Solicitar al usuario el directorio de salida"""
        directory = filedialog.askdirectory(title="Seleccionar directorio de salida")
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
        
        # Deshabilitar botones mientras se ejecuta
        self.process_btn.config(state=tk.DISABLED)
        self.results_btn.config(state=tk.DISABLED)
        self.is_running = True
        self.progress_value = 0.0
        self.progress_bar["value"] = 0
          # Mostrar mensaje inicial
        self.update_status(
            f"Iniciando procesamiento por lotes...\n"
            f"Directorio de entrada: {self.input_dir.get()}\n"
            f"Patrón de archivos: {self.file_pattern.get()}\n"
            f"Directorio de salida: {self.output_dir.get()}",
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
        try:            # Usar la función de procesamiento por lotes
            # Configurar la redirección de la salida para capturar los mensajes de progreso
            import io
            import sys
            from contextlib import redirect_stdout
            
            # Capturar la salida estándar
            output_buffer = io.StringIO()
            with redirect_stdout(output_buffer):
                # Ejecutar el procesamiento
                result = batch_process(
                    input_dir=self.input_dir.get(),
                    output_dir=self.output_dir.get(),
                    file_pattern=self.file_pattern.get(),
                    enforce_naming_convention=True  # Siempre validar formato
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
            # Habilitar botones al terminar
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.results_btn.config(state=tk.NORMAL))
            self.is_running = False

    def _update_results(self, output_text, result):
        """Actualizar la interfaz con los resultados del procesamiento"""
        self.update_status(output_text)
        self.update_status("\n\nResumen:")
        self.update_status(f"Archivos procesados: {result['processed_files']}")
        self.update_status(f"Archivos omitidos: {result['skipped_files']}")
        
        # Establecer progreso al 100%
        self.progress_bar["value"] = 100
          # Mensaje de finalización
        self.processing_status.set("Procesamiento completado")
        
        # Mostrar un mensaje de éxito
        messagebox.showinfo("Procesamiento completado", 
                           f"Se han procesado {result['processed_files']} imágenes.\n"
                           f"Los resultados están disponibles en:\n{self.output_dir.get()}\n\n"
                           f"Se ha generado un archivo Excel con el análisis completo.")

    def _show_error(self, error_msg):
        """Mostrar un mensaje de error en la interfaz"""
        self.update_status(error_msg, append=False)
        self.processing_status.set("Error en el procesamiento")
        messagebox.showerror("Error", "Se ha producido un error durante el procesamiento")

    def _view_aggregated_results(self):
        """Mostrar los resultados agregados en una ventana separada"""        # Verificar si existen resultados agregados
        aggregated_file = os.path.join(
            self.output_dir.get(), 
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
        # Crear un frame para la tabla
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Crear treeview para la tabla
        columns = ('grupo', 'total_images', 'mean_avg_inclusions_per_cell', 
                  'std_avg_inclusions_per_cell', 'mean_avg_inclusion_ratio', 
                  'std_avg_inclusion_ratio')
        
        tree = ttk.Treeview(frame, columns=columns, show='headings', bootstyle='primary')
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Agregar scrollbar
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.configure(yscrollcommand=scrollbar.set)
        
        # Definir encabezados
        tree.heading('grupo', text=level_name)
        tree.heading('total_images', text='Imágenes')
        tree.heading('mean_avg_inclusions_per_cell', text='Media Incl./Célula')
        tree.heading('std_avg_inclusions_per_cell', text='DE Incl./Célula')
        tree.heading('mean_avg_inclusion_ratio', text='Media Ratio Incl.')
        tree.heading('std_avg_inclusion_ratio', text='DE Ratio Incl.')
        
        # Ajustar anchos
        tree.column('grupo', width=200)
        tree.column('total_images', width=70)
        tree.column('mean_avg_inclusions_per_cell', width=120)
        tree.column('std_avg_inclusions_per_cell', width=120)
        tree.column('mean_avg_inclusion_ratio', width=120)
        tree.column('std_avg_inclusion_ratio', width=120)
        
        # Insertar datos
        for group, stats in data.items():
            tree.insert('', tk.END, values=(
                group,
                stats.get('total_images', 0),
                f"{stats.get('mean_avg_inclusions_per_cell', 0):.2f}",
                f"{stats.get('std_avg_inclusions_per_cell', 0):.2f}",
                f"{stats.get('mean_avg_inclusion_ratio', 0)*100:.2f}%",
                f"{stats.get('std_avg_inclusion_ratio', 0)*100:.2f}%"
            ))
    
    def _create_plots(self, parent, data):
        """Crear gráficos con los resultados agregados"""
        # Extraer datos para las gráficas
        ct_data = data.get('condition_time', {})
        
        if not ct_data:
            ttk.Label(parent, text="No hay datos suficientes para generar gráficos").pack()
            return
        
        # Crear figura con estilo moderno
        plt.style.use('seaborn-v0_8-darkgrid')
        fig = plt.Figure(figsize=(10, 8), dpi=100)
        
        # Agrupar por condición para crear las gráficas
        conditions = {}
        for key in ct_data.keys():
            condition, time = key.split('/')
            if condition not in conditions:
                conditions[condition] = []
            conditions[condition].append((time, key))
        
        # Ordenar los tiempos numéricamente dentro de cada condición
        for condition in conditions:
            conditions[condition].sort(key=lambda x: x[0])
        
        # Crear los gráficos
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        
        # Colores para cada condición
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        # Graficar inclusions per cell
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
        
        # Graficar inclusion ratio
        for i, (condition, times) in enumerate(conditions.items()):
            color = colors[i % len(colors)]
            x_values = [time for time, _ in times]
            y_values = [ct_data[key]['mean_avg_inclusion_ratio']*100 for _, key in times]
            yerr_values = [ct_data[key]['std_avg_inclusion_ratio']*100 for _, key in times]
            
            ax2.errorbar(x_values, yerr=yerr_values, marker='o', label=condition, color=color, capsize=4)
        
        ax2.set_title("Ratio promedio de área de inclusiones", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Tiempo", fontsize=10)
        ax2.set_ylabel("Ratio de inclusiones (%)", fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Ajustar diseño
        fig.tight_layout()
        
        # Integrar la figura en la interfaz
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def _update_file_pattern(self, directory):
        """
        Detecta las extensiones de imagen presentes en el directorio y actualiza el patrón de archivos.
        
        Args:
            directory: Directorio donde buscar las imágenes
        """
        # Extensiones comunes de imágenes
        image_extensions = ['.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp']
        
        # Diccionario para contar las ocurrencias de cada extensión
        extension_count = {ext: 0 for ext in image_extensions}
        
        # Contar archivos por extensión
        try:
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(filename)
                    ext = ext.lower()
                    if ext in extension_count:
                        extension_count[ext] += 1
        except Exception as e:
            print(f"Error al leer directorio: {e}")
            return
            
        # Filtrar solo las extensiones que existen en el directorio
        found_extensions = [ext for ext, count in extension_count.items() if count > 0]
        
        if not found_extensions:
            # Si no se encontraron extensiones conocidas, mantener el valor predeterminado
            return
            
        # Construir el patrón combinando todas las extensiones encontradas
        if len(found_extensions) == 1:
            # Si solo hay una extensión, usar un patrón simple
            self.file_pattern.set(f"*{found_extensions[0]}")
        else:
            # Si hay múltiples extensiones, usar un patrón con todas ellas
            combined_pattern = f"*.{{{','.join(ext.strip('.') for ext in found_extensions)}}}"
            self.file_pattern.set(combined_pattern)
            
        # Actualizar el mensaje de estado
        extensions_str = ', '.join(found_extensions)
        self.update_status(f"Extensiones de imagen detectadas: {extensions_str}", append=True)
    
    def setup_methodology_tab(self, parent):
        """Configura la pestaña de metodología con descripción detallada del pipeline"""
        # Frame principal dividido en dos columnas
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # División en dos columnas: izquierda para lista de pasos, derecha para descripción
        steps_frame = ttk.LabelFrame(main_frame, text="Pipeline de Análisis")
        steps_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5, ipadx=5, ipady=5)
        steps_frame.config(width=200)  # Ancho fijo para el panel izquierdo
        
        description_frame = ttk.LabelFrame(main_frame, text="Descripción Detallada")
        description_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Lista de pasos en el panel izquierdo
        self.steps_list = tk.Listbox(steps_frame, selectmode=tk.SINGLE, 
                                font=("Segoe UI", 10), 
                                activestyle="dotbox",
                                width=25)
        self.steps_list.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Añadir los pasos del pipeline
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
        
        # Área de texto para la descripción en el panel derecho
        self.description_text = scrolledtext.ScrolledText(description_frame, wrap=tk.WORD,
                                                     font=("Segoe UI", 10))
        self.description_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Vincular el evento de selección de la lista a la función de actualización de descripción
        self.steps_list.bind('<<ListboxSelect>>', self._update_step_description)
        
        # Seleccionar el primer elemento por defecto
        self.steps_list.selection_set(0)
        self._update_step_description(None)  # Llamar con None simulará el evento

    def _update_step_description(self, event):
        """Actualiza el panel de descripción según el paso seleccionado"""
        try:
            # Obtener índice seleccionado
            selected_idx = self.steps_list.curselection()[0]
        except IndexError:
            # Si no hay selección, usar el índice 0
            selected_idx = 0
            self.steps_list.selection_set(0)
        
        # Limpiar el área de texto
        self.description_text.config(state=tk.NORMAL)
        self.description_text.delete(1.0, tk.END)
        
        # Insertar la descripción correspondiente según el índice
        descriptions = self._get_step_descriptions()
        self.description_text.insert(tk.END, descriptions[selected_idx])
        
        # Desplazarse al principio y desactivar edición
        self.description_text.see(1.0)
        self.description_text.config(state=tk.DISABLED)

    def _get_step_descriptions(self):
        """Devuelve las descripciones detalladas de cada paso del pipeline"""
        descriptions = [
            # 1. Preprocesamiento de imágenes
            """El preprocesamiento es crucial para mejorar la calidad de las imágenes antes del análisis posterior. Este paso incluye:

Normalización de contraste:
- Min-Max: Expande el rango dinámico de la imagen para maximizar el contraste
- CLAHE (Contrast Limited Adaptive Histogram Equalization): Mejora el contraste local mientras limita la amplificación del ruido
- Ecualización de histograma: Redistribuye los niveles de gris para mejorar el contraste global

Reducción de ruido:
- Filtro Gaussiano: Suaviza la imagen reduciendo el ruido con un kernel gaussiano
- Filtro de mediana: Preserva bordes mientras elimina el ruido de tipo "sal y pimienta"
- Filtro bilateral: Reduce el ruido preservando los bordes importantes

Corrección de iluminación no uniforme:
- Sustracción de fondo: Elimina la iluminación no uniforme restando una versión muy suavizada de la imagen
- Operaciones morfológicas: Utiliza filtros morfológicos para estimar y corregir la iluminación de fondo
- Filtrado homomórfico: Separa los componentes de iluminación y reflectancia para normalizar la iluminación

El código implementa un pipeline configurable que permite aplicar diferentes técnicas según las características de las imágenes.

El resultado es una imagen con mejor contraste, menos ruido y una iluminación más uniforme, ideal para los pasos posteriores del análisis.""",

# 2. Segmentación de células
"""La segmentación es el proceso de identificar y delimitar las células bacterianas individuales en la imagen. Este paso incluye:

Umbralización (thresholding):
- Umbral adaptativo: Calcula umbrales localmente para adaptarse a diferentes regiones de la imagen
- Umbral de Otsu: Determina automáticamente el umbral óptimo basándose en el histograma
- Umbral binario: Aplica un valor de umbral fijo para separar objetos del fondo

Operaciones morfológicas:
- Erosión: Reduce el tamaño de los objetos y elimina pequeños detalles
- Dilatación: Aumenta el tamaño de los objetos y puede cerrar pequeños huecos
- Apertura/Cierre: Combinaciones de erosión y dilatación para suavizar contornos y eliminar ruido

## Algoritmo Watershed
- **Transformada de distancia**: Calcula la distancia de cada píxel al fondo
- **Marcadores**: Identifica núcleos de células mediante máximos locales
- **Watershed**: Separa células adyacentes utilizando la analogía de llenado de cuencas hidrográficas

## Filtrado de regiones
- **Área**: Elimina objetos demasiado pequeños o grandes para ser células
- **Circularidad**: Filtra por forma para descartar objetos no celulares
- **Relación de aspecto**: Descarta objetos alargados que pueden ser artefactos

El algoritmo implementa técnicas avanzadas para mejorar la segmentación de células que están en contacto, un problema común en imágenes de bacterias.

```python
# Ejemplo de segmentación con watershed
def segment_cells_enhanced(image, min_cell_size=60, min_distance=20):
    # Suavizado para reducir ruido
    img_blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
    
    # Umbralización mediante Otsu
    _, img_thresh = cv2.threshold(img_blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Calcular transformada de distancia
    distance = ndi.distance_transform_edt(img_thresh)
    
    # Encontrar marcadores para watershed
    local_max = peak_local_max(distance, min_distance=min_distance, labels=img_thresh)
    markers = np.zeros_like(distance, dtype=np.int32)
    markers[tuple(local_max.T)] = np.arange(1, local_max.shape[0] + 1)
    
    # Aplicar watershed
    labels_ws = segmentation.watershed(-distance, markers, mask=img_thresh)
    
    # Eliminar objetos pequeños
    labels_ws = morphology.remove_small_objects(labels_ws, min_size=min_cell_size)
    
    return labels_ws
```

El resultado final es una imagen donde cada célula tiene una etiqueta única, permitiendo su análisis individualizado.""",

            # 3. Detección de inclusiones
            """# Detección de inclusiones de polifosfatos

Este paso identifica y caracteriza las inclusiones de polifosfatos dentro de las células previamente segmentadas:

## Creación de máscaras celulares
- Cada célula segmentada se procesa independientemente
- Se generan máscaras binarias individuales para cada célula

## Mejora de contraste celular
- **CLAHE**: Mejora el contraste local dentro de cada célula
- **Ecualización de histograma**: Ajusta el contraste específicamente en la región celular

## Mejora de bordes
- **Filtro de Sobel**: Detecta bordes dentro de la célula
- **Ponderación adaptativa**: Combina bordes con la imagen original para mejorar la separación de inclusiones

## Umbralización multinivel
- **Umbral adaptativo local**: Ajusta la sensibilidad según la intensidad local
- **Umbralización múltiple**: Aplica varios niveles de umbral para detectar inclusiones con diferentes intensidades
- **Ajuste por estadísticas celulares**: Utiliza la media y desviación estándar de cada célula para calibrar los umbrales

## Separación de inclusiones cercanas
- **Watershed con marcadores**: Separa inclusiones que aparecen fusionadas
- **Transformada Top-Hat**: Resalta pequeñas estructuras dentro de las células
- **Ponderación de intensidad**: Considera tanto la distancia como la intensidad para la separación

## Filtrado de inclusiones
- **Tamaño**: Elimina objetos demasiado pequeños o grandes
- **Circularidad**: Filtra por forma para distinguir inclusiones reales
- **Contraste**: Verifica que las inclusiones tengan suficiente diferencia con su entorno
- **Análisis de textura**: Evalúa patrones de intensidad para confirmar inclusiones genuinas

```python
# Ejemplo simplificado de detección de inclusiones
def detect_inclusions_in_cell(original_image, cell_mask):
    # Mejorar contraste dentro de la célula
    enhanced_cell = enhance_cell_contrast(original_image, cell_mask, method='clahe')
    
    # Aplicar umbralización adaptativa local
    inclusion_candidates = adaptive_local_threshold(
        enhanced_cell, cell_mask, block_size=15, sensitivity=0.8
    )
    
    # Separar inclusiones cercanas
    separated_inclusions = separate_inclusions_watershed(
        inclusion_candidates, original_image, min_distance=5
    )
    
    # Filtrar inclusiones por criterios de forma y contraste
    labeled_inclusions, props = filter_inclusions(
        separated_inclusions, original_image,
        min_size=5, max_size=1500, min_circularity=0.4
    )
    
    return labeled_inclusions, props
```

El algoritmo implementado (versión 2.0) incluye técnicas avanzadas para manejar mejor las inclusiones cercanas, adaptarse a variaciones de contraste y reducir falsos positivos.""",

            # 4. Extracción de características
            """# Extracción de características

Este paso extrae información cuantitativa sobre las células y sus inclusiones de polifosfatos:

## Características geométricas
- **Área**: Superficie total ocupada por la célula o inclusión (en píxeles)
- **Perímetro**: Longitud del contorno exterior
- **Centroide**: Posición central (coordenadas x,y)
- **Ejes mayor y menor**: Dimensiones principales del objeto
- **Orientación**: Ángulo del eje principal respecto a la horizontal
- **Excentricidad**: Medida de elongación (0=círculo, 1=línea)
- **Solidez**: Proporción entre el área y el área de su envolvente convexa

## Características de intensidad
- **Intensidad media**: Valor promedio de los píxeles en la región
- **Intensidad mediana**: Valor mediano de los píxeles
- **Desviación estándar**: Medida de dispersión de las intensidades
- **Valores mínimo y máximo**: Extremos de intensidad en la región
- **Percentiles de intensidad**: Distribución de valores de intensidad

## Características de textura
- **Contraste local**: Diferencias de intensidad entre píxeles adyacentes
- **Homogeneidad**: Uniformidad de la distribución de intensidades
- **Energía**: Sumatoria de elementos al cuadrado de la matriz de co-ocurrencia
- **Correlación**: Medida de dependencia lineal de intensidades

## Características contextuales
- **Distancia al borde celular**: Proximidad de la inclusión al límite de la célula
- **Número de inclusiones vecinas**: Cantidad de inclusiones cercanas
- **Ratio inclusión/célula**: Proporción entre el área de la inclusión y la célula

```python
# Ejemplo de extracción de características para inclusiones
def extract_inclusion_features(labeled_image, intensity_image, regionprops=None):
    if regionprops is None:
        regionprops = measure.regionprops(labeled_image, intensity_image)
    
    features = []
    for prop in regionprops:
        # Extraer características básicas
        inclusion_data = {
            'area': prop.area,
            'centroid': prop.centroid,
            'mean_intensity': prop.mean_intensity,
            'max_intensity': prop.max_intensity,
            'min_intensity': prop.min_intensity,
            'eccentricity': prop.eccentricity,
            'solidity': prop.solidity,
            'perimeter': prop.perimeter
        }
        
        # Calcular características adicionales
        if prop.area > 0 and prop.perimeter > 0:
            # Circularidad: 4π*área/perímetro²
            circularity = (4 * np.pi * prop.area) / (prop.perimeter ** 2)
            inclusion_data['circularity'] = circularity
        
        features.append(inclusion_data)
    
    return features
```

Las características extraídas proporcionan una descripción cuantitativa completa de cada inclusión y célula, permitiendo análisis estadísticos posteriores y la clasificación automática de patrones.""",

            # 5. Análisis estadístico
            """# Análisis estadístico

Este componente procesa las características extraídas para generar estadísticas descriptivas y agregadas:

## Estadísticas por imagen
- **Total de células**: Número de células detectadas
- **Células con inclusiones**: Cantidad y porcentaje de células que contienen inclusiones
- **Inclusiones por célula**: Promedio, desviación estándar, mínimo y máximo
- **Tamaño de inclusiones**: Distribución estadística de áreas (media, mediana, percentiles)
- **Ratio inclusión/célula**: Proporción del área celular ocupada por inclusiones

## Agregación y agrupamiento
- **Agrupación jerárquica**: Organiza resultados por CONDICIÓN/TIEMPO/RÉPLICA
- **Normalización entre condiciones**: Ajusta valores para comparabilidad entre experimentos
- **Pruebas estadísticas**: Evalúa significancia de diferencias entre grupos

## Extracción de metadatos
- **Parseo de nombres de archivo**: Extrae información experimental (condición, tiempo, réplica)
- **Validación de formato**: Verifica que los archivos sigan la convención de nomenclatura
- **Organización jerárquica**: Estructura los resultados según los metadatos

```python
def summarize_inclusions(all_inclusions, segmented_image):
    # Inicializar estadísticas
    total_cells = len(all_inclusions)
    cells_with_inclusions = sum(1 for cell_id, incs in all_inclusions.items() if len(incs) > 0)
    total_inclusions = sum(len(incs) for incs in all_inclusions.values())
    
    # Porcentaje de células con inclusiones
    percent_cells_with_inclusions = (cells_with_inclusions / total_cells * 100) if total_cells > 0 else 0
    
    # Número de inclusiones por célula
    inclusions_per_cell = [len(incs) for incs in all_inclusions.values()]
    avg_inclusions_per_cell = np.mean(inclusions_per_cell) if inclusions_per_cell else 0
    std_inclusions_per_cell = np.std(inclusions_per_cell) if inclusions_per_cell else 0
    
    # Tamaño de inclusiones
    all_inclusion_areas = [inc['area'] for incs in all_inclusions.values() for inc in incs]
    avg_inclusion_area = np.mean(all_inclusion_areas) if all_inclusion_areas else 0
    std_inclusion_area = np.std(all_inclusion_areas) if all_inclusion_areas else 0
    
    # Ratio de área de inclusiones respecto a células
    inclusion_ratios = []
    for cell_id, incs in all_inclusions.items():
        cell_mask = (segmented_image == cell_id)
        cell_area = np.sum(cell_mask)
        
        if cell_area > 0 and incs:
            total_inclusion_area = sum(inc['area'] for inc in incs)
            inclusion_ratios.append(total_inclusion_area / cell_area)
    
    avg_inclusion_ratio = np.mean(inclusion_ratios) if inclusion_ratios else 0
    std_inclusion_ratio = np.std(inclusion_ratios) if inclusion_ratios else 0
    
    return {
        'total_cells': total_cells,
        'cells_with_inclusions': cells_with_inclusions,
        'percent_cells_with_inclusions': percent_cells_with_inclusions,
        # ... más estadísticas ...
    }
```

## Exportación de resultados
- **Generación de Excel**: Crea hojas de cálculo organizadas con todas las estadísticas
- **Serialización JSON**: Guarda los datos en formato estructurado para análisis posteriores
- **Informes automatizados**: Genera resúmenes estadísticos por condición experimental

El sistema permite agregar los resultados de múltiples imágenes para obtener análisis robustos a nivel de condición experimental, facilitando la identificación de patrones biológicos significativos.""",

            # 6. Visualización de resultados
            """# Visualización de resultados

Este componente genera representaciones visuales de los resultados del análisis:

## Visualización de segmentación celular
- **Overlay de segmentación**: Superpone células segmentadas sobre la imagen original
- **Contornos celulares**: Dibuja los límites de cada célula identificada
- **Etiquetado numérico**: Asigna identificadores únicos a cada célula

## Visualización de inclusiones detectadas
- **Marcado de inclusiones**: Resalta las inclusiones detectadas dentro de cada célula
- **Codificación por color**: Usa colores para representar características como tamaño o intensidad
- **Visualización de contornos**: Delimita el perímetro exacto de cada inclusión

## Visualizaciones estadísticas
- **Histogramas**: Distribución del número de inclusiones por célula, tamaños, etc.
- **Gráficos de barras**: Comparación entre diferentes condiciones experimentales
- **Gráficos de caja**: Visualización de la distribución estadística de las características
- **Gráficos de línea**: Evolución temporal de parámetros clave
- **Gráficos de dispersión**: Relaciones entre diferentes características

```python
def visualize_inclusions(original_image, segmented_image, all_inclusions):
    # Convertir a BGR si está en escala de grises
    if len(original_image.shape) == 2:
        display_img = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        display_img = original_image.copy()
    
    # Colorear células segmentadas
    segmentation_overlay = label2rgb(segmented_image, image=original_image, 
                                    bg_label=0, alpha=0.3)
    segmentation_overlay = (segmentation_overlay * 255).astype(np.uint8)
    segmentation_overlay_bgr = cv2.cvtColor(segmentation_overlay, cv2.COLOR_RGB2BGR)
    
    # Marcar inclusiones
    for cell_id, inclusions in all_inclusions.items():
        for inclusion in inclusions:
            # Obtener coordenadas del centroide
            y, x = inclusion['centroid']
            x, y = int(x), int(y)
            
            # Dibujar círculo en la posición de la inclusión
            radius = int(np.sqrt(inclusion['area'] / np.pi))
            radius = max(radius, 2)  # Asegurar un radio mínimo visible
            
            cv2.circle(segmentation_overlay_bgr, (x, y), radius, (0, 0, 255), 1)
    
    return segmentation_overlay_bgr
```

## Visualización interactiva
- **Selección de células**: Permite al usuario seleccionar células específicas para su análisis
- **Zoom adaptativo**: Ajusta la visualización para examinar detalles específicos
- **Filtrado dinámico**: Permite mostrar/ocultar inclusiones según sus características
- **Comparación lado a lado**: Facilita la comparación entre diferentes condiciones o tiempos

## Exportación de visualizaciones
- **Imágenes de alta resolución**: Guarda las visualizaciones para publicaciones científicas
- **Figuras compuestas**: Genera paneles múltiples que combinan diferentes visualizaciones
- **Formatos vectoriales**: Exporta a formatos como SVG o PDF para publicaciones

Las visualizaciones generadas no solo facilitan la validación visual de los resultados, sino que también ayudan a comunicar los hallazgos de manera efectiva y a identificar patrones que podrían no ser evidentes en los análisis numéricos puros."""
        ]
        
        return descriptions
def main():
    # Crear la ventana principal con ttkbootstrap
    root = ttk.Window(
        title="Analizador de Inclusiones de Polifosfatos",
        themename="cosmo",  # Temas: cosmo, flatly, journal, litera, lumen, etc.
        size=(1000, 700),
        minsize=(800, 600),
        resizable=(True, True),
    )
    
    # Crear la aplicación
    app = POLIP_Analyzer_GUI(root)
    
    # Iniciar bucle principal
    root.mainloop()


if __name__ == "__main__":
    main()