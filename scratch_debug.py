import sys
import os

# Add repo root to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from src.core import batch_process
from config import get_configs_for_morphology

preprocess_config, segment_config, detection_config = get_configs_for_morphology("bacilli")

try:
    result = batch_process(
        input_dir="/home/francisco/Descargas/Imágenes209",
        output_dir="/home/francisco/Descargas/Imágenes209/processed_test",
        file_pattern="*.tif",
        enforce_naming_convention=True,
        save_intermediate_images=False,
        preprocess_config=preprocess_config,
        segment_config=segment_config,
        detection_config=detection_config
    )
    print("SUCCESS")
    print(result)
except Exception as e:
    import traceback
    print("FAILED")
    traceback.print_exc()
