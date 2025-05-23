"""
Script principal para el análisis de inclusiones de polifosfatos en células bacterianas.
"""
from src.core import batch_process, process_image_v2
from config import DEVELOPMENT_MODE, STANDARD_MODE_CONFIG

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
    args = parser.parse_args()

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
            image_path = "data/raw/sample_image.png"
            if DEVELOPMENT_MODE:
                print("Modo de desarrollo activado. Se mostrarán visualizaciones interactivas.")
                process_image_v2(image_path)
            else:
                print("Modo estándar. Procesando imagen individual...")
                process_image_v2(image_path, **STANDARD_MODE_CONFIG)
    except Exception as e:
        print(f"Error durante el procesamiento: {e}")
        traceback.print_exc()