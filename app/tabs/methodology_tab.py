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
