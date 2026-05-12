"""
Script principal para el análisis de inclusiones de polifosfatos en células bacterianas.
"""
from src.core import batch_process, process_image_v2
from config import DEVELOPMENT_MODE, STANDARD_MODE_CONFIG, get_configs_for_morphology

if __name__ == "__main__":
    import argparse
    import traceback

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
    parser.add_argument('--morph', type=str, default='bacilli', choices=['bacilli', 'bifid'],
                        help='Morfología bacteriana: bacilli (bacilos) o bifid (bifidobacterias)')
    args = parser.parse_args()

    enforce_naming = not args.no_enforce_naming
    preprocess_config, segment_config, detection_config = get_configs_for_morphology(args.morph)
    print(f"Morfología seleccionada: {args.morph}")
    try:
        if args.batch:
            print(f"Iniciando procesamiento por lotes desde: {args.input}")
            print(f"Patrón de archivos: {args.pattern}")
            print(f"Validación de nombres: {'desactivada' if args.no_enforce_naming else 'activada'}")
            batch_process(
                input_dir=args.input,
                output_dir=args.output,
                file_pattern=args.pattern,
                enforce_naming_convention=enforce_naming,
                preprocess_config=preprocess_config,
                segment_config=segment_config,
                detection_config=detection_config
            )
        else:
            image_path = "data/raw/Bifid.tif"
            if DEVELOPMENT_MODE:
                print("Modo de desarrollo activado. Se mostrarán visualizaciones interactivas.")
                process_image_v2(image_path,
                                 preprocess_config=preprocess_config,
                                 segment_config=segment_config,
                                 detection_config=detection_config)
            else:
                print("Modo estándar. Procesando imagen individual...")
                process_image_v2(image_path,
                                 preprocess_config=preprocess_config,
                                 segment_config=segment_config,
                                 detection_config=detection_config,
                                 **STANDARD_MODE_CONFIG)
    except Exception as e:
        print(f"Error durante el procesamiento: {e}")
        traceback.print_exc()