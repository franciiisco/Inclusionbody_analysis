"""
Script principal para el análisis de inclusiones de polifosfatos en células bacterianas.
"""

import os
import glob
import json
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
import traceback
from src.preprocessing import preprocess_pipeline, visualize_preprocessing_steps
from src.segmentation import segment_cells, visualize_segmentation, threshold_image, apply_morphological_operations
from src.analysis import (
    summarize_inclusions, plot_inclusion_statistics, 
    extract_metadata_from_filename, validate_filename_format, 
    aggregate_inclusion_data
)
from src.detection_v2_0 import detect_all_inclusions_v2, visualize_inclusions_v2, summarize_inclusions_v2, plot_inclusion_statistics_v2
from config import (
    PREPROCESS_CONFIG, SEGMENT_CONFIG, DETECTION_V2_CONFIG,
    DEVELOPMENT_MODE, VISUALIZATION_SETTINGS, STANDARD_MODE_CONFIG
)


def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


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
    # Utilizamos directamente la versión 2 de detección
    return process_image_v2(
        image_path=image_path,
        preprocess_config=preprocess_config,
        segment_config=segment_config,
        detection_config=detection_config,
        output_dir=output_dir
    )

def process_image_v2(
    image_path: str,
    preprocess_config=None,
    segment_config=None,
    detection_config=None,
    output_dir=None,
    save_intermediate_images=None
):
    """
    Procesa una imagen para detectar células e inclusiones de polifosfatos usando la versión 2.0 
    de la detección de inclusiones, que mejora la separación de inclusiones cercanas.
    
    Args:
        image_path: Ruta a la imagen a procesar
        preprocess_config: Configuración para el preprocesamiento
        segment_config: Configuración para la segmentación
        detection_config: Configuración para la detección de inclusiones v2
        output_dir: Directorio para guardar resultados
        save_intermediate_images: Si True, guarda las imágenes intermedias (sobreescribe configuración)
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
    Configuración de detección de inclusiones v2.0
    ==========================================
    En esta sección se definen los parámetros para la detección de inclusiones de polifosfatos
    usando el método mejorado v2.0 con mejor separación de inclusiones cercanas.
    '''    
    if detection_config is None:
        detection_config = DETECTION_V2_CONFIG # Usar configuración v2
    
    if output_dir is None:
        output_dir = "data/processed"
    
    # Si se especificó explícitamente, usar ese valor para guardar imágenes intermedias
    if save_intermediate_images is not None:
        VISUALIZATION_SETTINGS['save_intermediate_images'] = save_intermediate_images
    
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
      # Guardar imagen preprocesada solo si está habilitado en la configuración
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    if VISUALIZATION_SETTINGS.get('save_intermediate_images'):
        preprocessed_path = os.path.join(output_dir, f"{base_filename}_preprocessed_v2.png")
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
      # Guardar imagen segmentada solo si está habilitado en la configuración
    if VISUALIZATION_SETTINGS.get('save_intermediate_images'):
        segmented_path = os.path.join(output_dir, f"{base_filename}_segmented_v2.png")
        segmented_vis = (segmented * 50).astype(np.uint8)  # Escalar para visualización
        cv2.imwrite(segmented_path, segmented_vis)
    
    # Paso 3: Detección de inclusiones con el método v2.0
    print("Detectando inclusiones de polifosfatos (método v2.0)...")
    # Invertir la imagen original para la detección de inclusiones (brillantes sobre fondo oscuro)
    inverted_for_detection = cv2.bitwise_not(original_for_detection)

    # Aplicar el nuevo método de detección v2.0
    all_inclusions = detect_all_inclusions_v2(
        inverted_for_detection,  # Usar la imagen invertida
        segmented,
        detection_config
    )

    # Visualizar resultados con el nuevo visualizador (asegurándonos de mostrar la visualización)
    result_image = visualize_inclusions_v2(
        original_for_detection,  # Usar la original (no invertida) para visualización
        segmented, 
        all_inclusions, 
        show_visualization=(DEVELOPMENT_MODE and VISUALIZATION_SETTINGS.get('show_inclusion_detection'))
    )
      # Guardar imagen con inclusiones detectadas (siempre se guarda como resultado principal)
    inclusions_path = os.path.join(output_dir, f"{base_filename}_inclusions_v2.png")
    cv2.imwrite(inclusions_path, result_image)
      # Generar estadísticas
    summary = summarize_inclusions_v2(all_inclusions, segmented)
    
    # Convert numpy types before JSON serialization
    summary = convert_numpy_types(summary)
    
    # Guardar estadísticas en formato JSON
    stats_path = os.path.join(output_dir, f"{base_filename}_stats_v2.json")
    with open(stats_path, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Imprimir resumen
    print("\nResumen de resultados (detección v2.0):")
    print(f"Total de células: {summary['total_cells']}")
    print(f"Células con inclusiones: {summary['cells_with_inclusions']} ({summary['percent_cells_with_inclusions']:.1f}%)")
    print(f"Total de inclusiones: {summary['total_inclusions']}")
    print(f"Promedio de inclusiones por célula: {summary['avg_inclusions_per_cell']:.2f} ± {summary['std_inclusions_per_cell']:.2f}")
    print(f"Tamaño promedio de inclusiones: {summary['avg_inclusion_area']:.2f} ± {summary['std_inclusion_area']:.2f} píxeles")
    print(f"Ratio promedio inclusiones/célula: {summary['avg_inclusion_ratio']*100:.2f}% ± {summary['std_inclusion_ratio']*100:.2f}%")
    
    # Guardar gráficos estadísticos
    if DEVELOPMENT_MODE and VISUALIZATION_SETTINGS.get('show_summary_plots'):
        plot_inclusion_statistics_v2(summary, all_inclusions)
    
    print(f"Procesamiento v2.0 completado. Resultados guardados en: {output_dir}")
    
    return {
        'original': original_image,
        'preprocessed': preprocessed,
        'segmented': segmented,
        'binary_mask': binary,
        'inclusions': all_inclusions,
        'summary': summary,
        'result_image': result_image
    }


def batch_process_v2(
    input_dir: str,
    output_dir: str = None,
    file_pattern: str = "*.jpg",
    enforce_naming_convention: bool = True,  # Nuevo parámetro para controlar la validación
    save_intermediate_images: bool = None,   # Nuevo parámetro para imágenes intermedias
    progress_callback=None,  # New parameter for progress tracking
    **configs
):
    """
    Procesa un lote de imágenes en un directorio usando el algoritmo de detección v2.0 mejorado.
    
    Args:
        input_dir: Directorio con imágenes a procesar
        output_dir: Directorio para guardar resultados
        file_pattern: Patrón para seleccionar archivos
        enforce_naming_convention: Si es True, valida que los nombres de archivo cumplan con el formato
                                 CONDICION_BOTE_REPLICA_TIEMPO_NºIMAGEN
        save_intermediate_images: Si True, guarda las imágenes intermedias (sobreescribe configuración)
        progress_callback: Función callback para reportar progreso (file_index, filename)
        **configs: Configuraciones para procesamiento
    """
    import os
    import glob
    import json
    import cv2
    import numpy as np
    import traceback
    
    # Configurar directorio de salida
    if output_dir is None:
        output_dir = os.path.join(input_dir, "processed_v2")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Encontrar todas las imágenes
    image_paths = glob.glob(os.path.join(input_dir, file_pattern))
    
    if not image_paths:
        print(f"No se encontraron imágenes con el patrón {file_pattern} en {input_dir}")
        return
    
    print(f"Procesando {len(image_paths)} imágenes con el algoritmo v2.0...")
    
    # Procesar cada imagen
    results = {}
    image_results = []  # Para la agregación de datos
    skipped_files = []
    for i, img_path in enumerate(image_paths):
        base_name = os.path.basename(img_path)
        print(f"\nProcesando imagen {i+1}/{len(image_paths)}: {base_name}")
        
        # Call the progress callback if provided
        if progress_callback:
            progress_callback(i, img_path)
        
        # Validar formato del nombre de archivo
        if enforce_naming_convention and not DEVELOPMENT_MODE and not validate_filename_format(img_path):
            print(f"Error: El archivo {base_name} no cumple con el formato requerido (CONDICION_BOTE_REPLICA_TIEMPO_NºIMAGEN)")
            print("Omitiendo este archivo...")
            skipped_files.append(base_name)
            continue

        try:
            result = process_image_v2(
                img_path, 
                output_dir=output_dir,
                save_intermediate_images=save_intermediate_images,
                **configs
            )
            
            # Guardar los resultados para el resumen
            results[base_name] = result['summary']
            
            # Añadir a la lista para agregación
            image_results.append((img_path, result['summary']))
            
        except Exception as e:
            print(f"Error procesando {base_name}: {e}")
            traceback.print_exc()
            skipped_files.append(base_name)
    
    # Guardar resultados consolidados
    summary_path = os.path.join(output_dir, "batch_summary_v2.json")
    with open(summary_path, 'w') as f:
        # Limpiar valores numpy para JSON
        clean_results = {}
        for img, summary in results.items():
            clean_summary = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                            for k, v in summary.items()}
            clean_results[img] = clean_summary
        json.dump(clean_results, f, indent=4)
    
    # Generar agregaciones por condición/tiempo/réplica
    if image_results:
        aggregated_results = aggregate_inclusion_data(image_results)
        
        # Guardar resultados agregados
        aggregated_path = os.path.join(output_dir, "aggregated_results_v2.json")
        with open(aggregated_path, 'w') as f:
            # Convertir valores numpy a tipos Python nativos para JSON
            clean_aggregated = {}
            for hierarchy, hierarchy_results in aggregated_results.items():
                clean_hierarchy = {}
                for key, result in hierarchy_results.items():
                    clean_result = {k: float(v) if isinstance(v, (np.float32, np.float64)) else v 
                                  for k, v in result.items()}
                    clean_hierarchy[key] = clean_result
                clean_aggregated[hierarchy] = clean_hierarchy
            json.dump(clean_aggregated, f, indent=4)
        
        print(f"\nResultados agregados guardados en: {aggregated_path}")
        print(f"\nProcesamiento por lotes v2.0 completado. Resultados guardados en: {output_dir}")
    
    # Si se omitieron archivos, mostrar un resumen
    if skipped_files:
        print(f"\nSe omitieron {len(skipped_files)} archivos por errores o formato de nombre incorrecto:")
        for file in skipped_files[:10]:  # Mostrar solo los primeros 10 para no saturar la consola
            print(f"- {file}")
        if len(skipped_files) > 10:
            print(f"... y {len(skipped_files) - 10} archivos más")
    
    return {
        'processed_files': len(results),
        'skipped_files': len(skipped_files),
        'results': results,
        'aggregated_results': aggregated_results if image_results else None,
        'image_results': image_results
    }


def batch_process(
    input_dir: str,
    output_dir: str = None,
    file_pattern: str = "*.jpg",
    enforce_naming_convention: bool = True,  # Nuevo parámetro para controlar la validación
    save_intermediate_images: bool = None,    # Nuevo parámetro para controlar imágenes intermedias
    progress_callback=None,  # New parameter for progress tracking
    **configs
):
    """
    Procesa un lote de imágenes en un directorio.
    
    Args:
        input_dir: Directorio con imágenes a procesar
        output_dir: Directorio para guardar resultados
        file_pattern: Patrón para seleccionar archivos
        enforce_naming_convention: Si es True, valida que los nombres de archivo cumplan con el formato
                                 CONDICION_BOTE_REPLICA_TIEMPO_NºIMAGEN
        save_intermediate_images: Si True, guarda las imágenes intermedias (sobreescribe configuración)
        progress_callback: Función callback para reportar progreso (file_index, filename)
        **configs: Configuraciones para procesamiento
    """
    # Utilizamos directamente la versión 2 de batch_process
    return batch_process_v2(
        input_dir=input_dir,
        output_dir=output_dir,
        file_pattern=file_pattern,
        enforce_naming_convention=enforce_naming_convention,
        save_intermediate_images=save_intermediate_images,
        progress_callback=progress_callback,  # Pass through the progress callback
        **configs
    )


if __name__ == "__main__":
    # Esta sección solo se ejecuta cuando se llama directamente al script
    # La GUI llamará directamente a las funciones batch_process o process_image
    import os
    import argparse
    import traceback
    import numpy as np
    import matplotlib.pyplot as plt
    import cv2
    import json
    import glob
    
    # Ejemplo de uso
    image_path = "data/raw/sample_image.png"  # Ajusta la ruta según tu estructura
      # Configuración de la línea de comandos
    parser = argparse.ArgumentParser(description='Análisis de inclusiones de polifosfatos en células bacterianas.')
    parser.add_argument('--batch', action='store_true',
                       help='Procesar un lote de imágenes')
    parser.add_argument('--input', type=str, default='data/raw',
                       help='Directorio de entrada para el procesamiento por lotes')
    parser.add_argument('--output', type=str, default=None,
                       help='Directorio de salida para el procesamiento por lotes')
    parser.add_argument('--pattern', type=str, default='*.png',
                       help='Patrón para seleccionar archivos (ej: *.png, *.tif)')
    parser.add_argument('--no-enforce-naming', action='store_true',
                       help='No validar formato de nombres de archivo (formato: CONDICION_BOTE_REPLICA_TIEMPO_NºIMAGEN)')
    args = parser.parse_args()
    
    # Configuración de parámetros
    enforce_naming = not args.no_enforce_naming
    try:
        if args.batch:
            print(f"Iniciando procesamiento por lotes desde: {args.input}")
            print(f"Patrón de archivos: {args.pattern}")
            print(f"Validación de nombres: {'desactivada' if args.no_enforce_naming else 'activada'}")
            
            batch_process(
                input_dir=args.input,
                output_dir=args.output,
                file_pattern=args.pattern,
                enforce_naming_convention=enforce_naming
            )
        else:
            if DEVELOPMENT_MODE:
                print("Modo de desarrollo activado. Se mostrarán visualizaciones interactivas.")
                process_image_v2(image_path)
            else:
                print("Modo estándar. Procesando imagen individual...")
                process_image_v2(image_path, **STANDARD_MODE_CONFIG)
    except Exception as e:
        print(f"Error durante el procesamiento: {e}")
        traceback.print_exc()