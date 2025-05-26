import os
import sys
import ttkbootstrap as ttk
from tkinter import filedialog, scrolledtext

# Ajustar el path para importar desde el directorio raíz
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

class AnalysisTab:
    def __init__(self, parent, input_dir_var, output_dir_var, file_pattern_var, process_callback, results_callback):
        self.parent = parent
        self.input_dir = input_dir_var
        self.output_dir = output_dir_var
        self.file_pattern = file_pattern_var
        self.process_callback = process_callback
        self.results_callback = results_callback
        
        # Variable para controlar la generación de imágenes intermedias
        self.save_intermediate_images = ttk.BooleanVar(value=False)
        
        # Create main layout with two columns - this will be the only container
        self.main_frame = ttk.Frame(self.parent)
        self.main_frame.pack(fill=ttk.BOTH, expand=True, padx=5, pady=5)
        
        # Left column for options
        self.options_frame = ttk.Frame(self.main_frame, width=300)
        self.options_frame.pack(side=ttk.LEFT, fill=ttk.Y, padx=(0, 5))
        self.options_frame.pack_propagate(False)  # Maintain width
        
        # Right column for status messages
        self.output_frame = ttk.Frame(self.main_frame)
        self.output_frame.pack(side=ttk.RIGHT, fill=ttk.BOTH, expand=True)
        
        # Setup the UI elements
        self.setup_options_frame()
        self.setup_output_frame()

    def setup_options_frame(self):
        # Sección 1: Directorios
        file_section = ttk.LabelFrame(self.options_frame, text="1. Seleccionar Directorios")
        file_section.pack(fill=ttk.X, padx=5, pady=5)
        
        ttk.Label(file_section, text="Directorio de imágenes:").pack(anchor='w', padx=5, pady=2)
        input_frame = ttk.Frame(file_section)
        input_frame.pack(fill=ttk.X, padx=5, pady=2)
        
        self.input_entry = ttk.Entry(input_frame, textvariable=self.input_dir)
        self.input_entry.pack(side=ttk.LEFT, fill=ttk.X, expand=True)
        input_btn = ttk.Button(input_frame, text="Examinar", command=self.browse_input_dir)
        input_btn.pack(side=ttk.RIGHT)
        
        ttk.Label(file_section, text="Directorio de salida:").pack(anchor='w', padx=5, pady=2)
        output_frame = ttk.Frame(file_section)
        output_frame.pack(fill=ttk.X, padx=5, pady=2)
        
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_dir)
        self.output_entry.pack(side=ttk.LEFT, fill=ttk.X, expand=True)
        output_btn = ttk.Button(output_frame, text="Examinar", command=self.browse_output_dir)
        output_btn.pack(side=ttk.RIGHT)
        
        # Sección 1.5: Opciones de visualización (añadida)
        viz_section = ttk.LabelFrame(self.options_frame, text="Opciones de visualización")
        viz_section.pack(fill=ttk.X, padx=5, pady=5)
        
        # Contenedor para el switch y su etiqueta
        switch_frame = ttk.Frame(viz_section)
        switch_frame.pack(fill=ttk.X, padx=5, pady=5)
        
        # Etiqueta para el switch
        ttk.Label(
            switch_frame, 
            text="Generar imágenes de\ncada paso del análisis:", 
            wraplength=160
        ).pack(side=ttk.LEFT, padx=5)
        
        # Switch usando Checkbutton con estilo round-toggle
        self.images_switch = ttk.Checkbutton(
            switch_frame,
            variable=self.save_intermediate_images,
            bootstyle="round-toggle",
            command=self._on_images_toggle
        )
        self.images_switch.pack(side=ttk.RIGHT, padx=10)
        
        # Sección 2: Formato de Archivo
        format_section = ttk.LabelFrame(self.options_frame, text="2. Formato de Archivo")
        format_section.pack(fill=ttk.X, padx=5, pady=5)
        
        format_note = ttk.Label(
            format_section, 
            text="Los archivos deben seguir el formato:\nCONDICION_BOTE_REPLICA_TIEMPO_Nº\nEjemplo: MEI_B1_R3_t4_026_BF1.png",
            justify='left'
        )
        format_note.pack(padx=10, pady=5, anchor='w')
        
        # Sección 3: Acciones
        action_section = ttk.LabelFrame(self.options_frame, text="3. Acciones")
        action_section.pack(fill=ttk.X, padx=5, pady=5)
        
        self.process_btn = ttk.Button(
            action_section, 
            text="Analizar", 
            command=self.process_callback
        )
        self.process_btn.pack(padx=5, pady=10, fill=ttk.X)
        
        self.results_btn = ttk.Button(
            action_section,
            text="Ver Resultados",
            command=self.results_callback
        )
        self.results_btn.pack(padx=5, pady=10, fill=ttk.X)
        
        # Sección 4: Información de contacto
        contact_section = ttk.LabelFrame(self.options_frame, text="Desarrollado por")
        contact_section.pack(fill=ttk.X, padx=5, pady=5)
        
        contact_panel = ttk.Frame(contact_section)
        contact_panel.pack(fill=ttk.X, padx=5, pady=5)
        
        contact_info = "Francisco Márquez Urbano\nfmarquez@cnta.es"
        contact_label = ttk.Label(contact_panel, text=contact_info, font=("Segoe UI", 9))
        contact_label.pack(padx=5, pady=5)

    def _on_images_toggle(self):
        """Callback cuando se activa/desactiva el switch de imágenes"""
        # Importar los módulos de configuración
        sys.path.insert(0, root_dir)
        try:
            from config import DETECTION_V2_CONFIG, VISUALIZATION_SETTINGS, DEVELOPMENT_MODE
            
            # Obtener el estado actual
            state = self.save_intermediate_images.get()
            
            # Actualizar configuración
            DETECTION_V2_CONFIG['debug']['save_intermediate_images'] = state
            VISUALIZATION_SETTINGS['save_intermediate_images'] = state
            
            # Desactivar también las visualizaciones emergentes cuando el usuario
            # desactiva la generación de imágenes desde la GUI
            if not state:
                VISUALIZATION_SETTINGS['show_preprocessing_steps'] = False
                VISUALIZATION_SETTINGS['show_segmentation_results'] = False
                VISUALIZATION_SETTINGS['show_inclusion_detection'] = False
                VISUALIZATION_SETTINGS['show_summary_plots'] = False
            
            # Mostrar mensaje en el área de estado
            status_text = "activado" if state else "desactivado"
            self.update_status(f"Generación de imágenes intermedias: {status_text}")
        except ImportError:
            self.update_status("Error: No se pudo cargar la configuración")
    def setup_output_frame(self):
        # Create a frame that will contain the status text and progress bar
        status_container = ttk.Frame(self.output_frame)
        status_container.pack(fill=ttk.BOTH, expand=True)
        
        # Área de texto para los mensajes de estado
        self.status_text = scrolledtext.ScrolledText(status_container, wrap=ttk.WORD)
        self.status_text.pack(fill=ttk.BOTH, expand=True, padx=5, pady=5)
        self.status_text.insert(ttk.END, "Listo para analizar. Configure las opciones y haga clic en 'Analizar'. \n\nEl análisis puede tardar varios minutos, por favor sea paciente. Déjeme trabajando, usted puede hacer otras cosas mientras.")
        self.status_text.config(state=ttk.DISABLED)
        
        # Barra de progreso
        progress_frame = ttk.Frame(status_container)
        progress_frame.pack(fill=ttk.X, padx=5, pady=5)
        
        ttk.Label(progress_frame, text="Progreso:").pack(side=ttk.LEFT)
        
        self.progress_bar = ttk.Progressbar(progress_frame, orient=ttk.HORIZONTAL, length=100, mode='determinate')
        self.progress_bar.pack(side=ttk.RIGHT, fill=ttk.X, expand=True, padx=(5, 0))

    def browse_input_dir(self):
        directory = filedialog.askdirectory(title="Seleccionar directorio de imágenes")
        if directory:
            self.input_dir.set(directory)
            if not self.output_dir.get():
                self.output_dir.set(directory + "/processed")

    def browse_output_dir(self):
        directory = filedialog.askdirectory(title="Seleccionar directorio de salida")
        if directory:
            self.output_dir.set(directory)
            
    def update_status(self, message, append=True):
        """Update the status text area with a new message"""
        self.status_text.config(state=ttk.NORMAL)
        if not append:
            self.status_text.delete(1.0, ttk.END)
        self.status_text.insert(ttk.END, message + "\n")
        self.status_text.see(ttk.END)  # Scroll to the end
        self.status_text.config(state=ttk.DISABLED)
        self.status_text.update()
        
    def update_progress(self, value):
        """Update the progress bar value (0-100)"""
        self.progress_bar["value"] = value
        self.progress_bar.update()