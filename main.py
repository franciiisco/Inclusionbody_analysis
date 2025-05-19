"""
Script principal para el análisis de inclusiones de polifosfatos en células bacterianas.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import glob # Importar glob aquí
from src.preprocessing import preprocess_pipeline, visualize_preprocessing_steps
from src.segmentation import segment_cells, visualize_segmentation, threshold_image, apply_morphological_operations
from src.detection import detect_all_inclusions, visualize_inclusions, summarize_inclusions, plot_inclusion_statistics
from config import (
    PREPROCESS_CONFIG, SEGMENT_CONFIG, DETECTION_CONFIG,
    DEVELOPMENT_MODE, VISUALIZATION_SETTINGS, STANDARD_MODE_CONFIG
)


def process_image(
    image_path: str,
    preprocess_config=None,
    segment_config=None,
    detection_config=None,
    output_dir=None
):
    """
    Procesa una imagen para detectar células e inclusiones de polifosfatos.
    
    Args:
        image_path: Ruta a la imagen a procesar
        preprocess_config: Configuración para el preprocesamiento
        segment_config: Configuración para la segmentación
        detection_config: Configuración para la detección de inclusiones
        output_dir: Directorio para guardar resultados
    """
    ''' 
    ==========================================
    Configuración de preprocesamiento
    =========================================
    En esta sección se definen los parámetros para el preprocesamiento de la imagen.
    Se pueden ajustar según las características de las imágenes y los requisitos del análisis.
    '''
    if preprocess_config is None:
        preprocess_config = PREPROCESS_CONFIG # Usar configuración importada
    
    ''' 
    ==========================================
    Configuración de segmentación de células
    ==========================================
    En esta sección se definen los parámetros para la segmentación de células.
    Se pueden ajustar según las características de las imágenes y los requisitos del análisis.
    '''
    
    if segment_config is None:
        segment_config = SEGMENT_CONFIG # Usar configuración importada
    
    ''' 
    ==========================================
    Configuración de detección de inclusiones
    ==========================================

    En esta sección se definen los parámetros para la detección de inclusiones de polifosfatos.
    Se pueden ajustar según las características de las imágenes y los requisitos del análisis.
    '''

    if detection_config is None:
        detection_config = DETECTION_CONFIG # Usar configuración importada
    
    if output_dir is None:
        output_dir = "data/processed"
    
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Cargar imagen
    print(f"Cargando imagen: {image_path}")
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if original_image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
    
    # Guardar copia de la imagen original para detección de inclusiones
    original_for_detection = original_image.copy()
    
    # Paso 1: Preprocesamiento
    print("Aplicando preprocesamiento...")
    preprocessed = preprocess_pipeline(original_image, preprocess_config)
    
    # Visualizar el preprocesamiento si está habilitado
    if DEVELOPMENT_MODE and VISUALIZATION_SETTINGS.get('show_preprocessing_steps'):
        visualize_preprocessing_steps(original_image, preprocess_config)
    
    # Guardar imagen preprocesada
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    preprocessed_path = os.path.join(output_dir, f"{base_filename}_preprocessed.png")
    cv2.imwrite(preprocessed_path, preprocessed)
    
    # Paso 2: Segmentación
    print("Segmentando células...")
    
    # Umbralización y operaciones morfológicas para visualización
    binary = threshold_image(
        preprocessed,
        method=segment_config['threshold']['method'],
        params=segment_config['threshold']['params']
    )
    binary = apply_morphological_operations(binary, segment_config['morphological'])
    
    # Segmentación completa
    segmented = segment_cells(preprocessed, segment_config)
    
    # Visualizar la segmentación si está habilitado
    if DEVELOPMENT_MODE and VISUALIZATION_SETTINGS.get('show_segmentation_results'):
        visualize_segmentation(original_image, segmented, binary, draw_contours=False)
    
    # Guardar imagen segmentada
    segmented_path = os.path.join(output_dir, f"{base_filename}_segmented.png")
    segmented_vis = (segmented * 50).astype(np.uint8)  # Escalar para visualización
    cv2.imwrite(segmented_path, segmented_vis)
    
    # Paso 3: Detección de inclusiones
    print("Detectando inclusiones de polifosfatos...")
    # Invertir la imagen original para la detección de inclusiones (brillantes sobre fondo oscuro)
    inverted_for_detection = cv2.bitwise_not(original_for_detection)

    # Ajustar los parámetros de detección en la configuración si es necesario
    # Estos son los valores por defecto que se usarán si no se especifica nada en detection_config
    # Asegúrate de que detection_config en main.py (o donde se llame) esté ajustado
    # para inclusiones brillantes si no se pasan explícitamente.
    # Ejemplo de cómo se vería la configuración actualizada en main.py:
    # if detection_config is None:
    #     detection_config = {
    #         'min_size': 3,  
    #         'max_size': 50, # Ajusta según tus necesidades
    #         'threshold_offset': 0.2,  # Positivo para inclusiones brillantes
    #         'min_contrast': 0.1,      
    #         'contrast_window': 3,      
    #         'remove_border': True,    
    #         'min_circularity': 0.6     # Filtro de forma
    #     }

    all_inclusions = detect_all_inclusions(
        inverted_for_detection,  # Usar la imagen invertida
        segmented,
        detection_config
    )

    # Visualizar resultados (asegurándonos de mostrar la visualización)
    # Para la visualización, es mejor usar la imagen original no invertida para que los colores tengan sentido.
    result_image = visualize_inclusions(
        original_for_detection, # Usar la original (no invertida) para visualización
        segmented, 
        all_inclusions, 
        show_visualization=(DEVELOPMENT_MODE and VISUALIZATION_SETTINGS.get('show_inclusion_detection'))
    )
    
    # Guardar imagen con inclusiones detectadas
    inclusions_path = os.path.join(output_dir, f"{base_filename}_inclusions.png")
    cv2.imwrite(inclusions_path, result_image)
    
    # Generar estadísticas
    summary = summarize_inclusions(all_inclusions, segmented)
    
    # Guardar estadísticas en formato JSON
    stats_path = os.path.join(output_dir, f"{base_filename}_stats.json")
    with open(stats_path, 'w') as f:
        # Convertir valores numpy a tipos Python nativos para JSON
        clean_summary = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                         for k, v in summary.items()}
        json.dump(clean_summary, f, indent=4)
    
    # Imprimir resumen
    print("\nResumen de resultados:")
    print(f"Total de células: {summary['total_cells']}")
    print(f"Células con inclusiones: {summary['cells_with_inclusions']} ({summary['percent_cells_with_inclusions']:.1f}%)")
    print(f"Total de inclusiones: {summary['total_inclusions']}")
    print(f"Promedio de inclusiones por célula: {summary['avg_inclusions_per_cell']:.2f} ± {summary['std_inclusions_per_cell']:.2f}")
    print(f"Tamaño promedio de inclusiones: {summary['avg_inclusion_area']:.2f} ± {summary['std_inclusion_area']:.2f} píxeles")
    print(f"Ratio promedio inclusiones/célula: {summary['avg_inclusion_ratio']*100:.2f}% ± {summary['std_inclusion_ratio']*100:.2f}%")
    
    # Guardar gráficos estadísticos
    if DEVELOPMENT_MODE and VISUALIZATION_SETTINGS.get('show_summary_plots'):
        plt.figure(figsize=(12, 10))
        plot_inclusion_statistics(summary, all_inclusions)
        stats_plot_path = os.path.join(output_dir, f"{base_filename}_stats_plot.png")
        plt.savefig(stats_plot_path)
    
    print(f"Procesamiento completado. Resultados guardados en: {output_dir}")
    
    return {
        'original': original_image,
        'preprocessed': preprocessed,
        'segmented': segmented,
        'binary_mask': binary,  # Añadimos la máscara binaria para referencia
        'inclusions': all_inclusions,
        'summary': summary,
        'result_image': result_image
    }


def batch_process(
    input_dir: str,
    output_dir: str = None,
    file_pattern: str = "*.jpg",
    **configs
):
    """
    Procesa un lote de imágenes en un directorio.
    
    Args:
        input_dir: Directorio con imágenes a procesar
        output_dir: Directorio para guardar resultados
        file_pattern: Patrón para seleccionar archivos
        **configs: Configuraciones para procesamiento
    """
    # Configurar directorio de salida
    if output_dir is None:
        output_dir = os.path.join(input_dir, "processed")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Encontrar todas las imágenes
    image_paths = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not image_paths:
        print(f"No se encontraron imágenes con el patrón {file_pattern} en {input_dir}")
        return
    
    print(f"Procesando {len(image_paths)} imágenes...")
    
    # Procesar cada imagen
    results = {}
    for i, img_path in enumerate(image_paths):
        print(f"\nProcesando imagen {i+1}/{len(image_paths)}: {os.path.basename(img_path)}")
        try:
            result = process_image(
                img_path, 
                output_dir=output_dir, 
                **configs
            )
            results[os.path.basename(img_path)] = result['summary']
        except Exception as e:
            print(f"Error procesando {img_path}: {e}")
    
    # Guardar resultados consolidados
    summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(summary_path, 'w') as f:
        # Limpiar valores numpy para JSON
        clean_results = {}
        for img, summary in results.items():
            clean_results[img] = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                 for k, v in summary.items()}
        json.dump(clean_results, f, indent=4)
    
    print(f"\nProcesamiento por lotes completado. Resultados guardados en: {output_dir}")


if __name__ == "__main__":
    # Ejemplo de uso
    image_path = "data/raw/sample_image.png"  # Ajusta la ruta según tu estructura
    
    # Las configuraciones de preprocesamiento y segmentación se tomarán de config.py por defecto.
    # Si se necesita una configuración de detección personalizada para una ejecución específica, 
    # se puede pasar como argumento a process_image o batch_process.
    # custom_detection_config = {
    #     'min_size': 4,  
    #     'max_size': 350, 
    #     'threshold_offset': 0.2, 
    #     'min_contrast': 0.12,    
    #     'contrast_window': 3,      
    #     'remove_border': False,    
    #     'min_circularity': 0.7
    # }
    
    try:
        if DEVELOPMENT_MODE:
            print("Modo Desarrollo Activado: Mostrando visualizaciones según VISUALIZATION_SETTINGS.")
            # En modo desarrollo, procesamos una imagen individual.
            # Ahora usará DETECTION_CONFIG de config.py por defecto
            results = process_image(
                image_path
                # detection_config=custom_detection_config # Comentado para usar config.py
            )
            # Si los gráficos de resumen están habilitados, muéstralos.
            if VISUALIZATION_SETTINGS.get('show_summary_plots'):
                plt.show()

        else:
            print("Modo Estándar Activado: No se mostrarán visualizaciones interactivas.")
            run_type = STANDARD_MODE_CONFIG.get('run_type', 'single')

            if run_type == 'single':
                print("Ejecutando análisis para una sola imagen.")
                # Ahora usará DETECTION_CONFIG de config.py por defecto
                results = process_image(
                    image_path
                    # detection_config=custom_detection_config # Comentado para usar config.py
                )
            elif run_type == 'batch':
                print("Ejecutando análisis por lotes.")
                # Ahora usará DETECTION_CONFIG de config.py por defecto
                batch_process(
                    "data/raw", # Directorio de entrada de ejemplo
                    "data/processed", # Directorio de salida de ejemplo
                    "*.png" # Patrón de archivo de ejemplo
                    # detection_config=custom_detection_config # Comentado para usar config.py
                )
            else:
                print(f"Tipo de ejecución no reconocido en STANDARD_MODE_CONFIG: {run_type}")
        
    except Exception as e:
        print(f"Error durante el procesamiento: {e}")
        import traceback
        traceback.print_exc()