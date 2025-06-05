"""
Test unitario para la función extract_metadata_from_filename.

Este test comprueba que la función es capaz de extraer correctamente los metadatos
de diferentes formatos de nombres de archivo.
"""
import unittest
import sys
import os

# Añadir directorio src al path para poder importar los módulos
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from analysis import extract_metadata_from_filename


class TestFilenameParser(unittest.TestCase):
    """Pruebas para la función extract_metadata_from_filename."""

    def test_ideal_pattern(self):
        """Prueba el patrón ideal: CONDICION_BOTE_REPLICA_TIEMPO_NUMERO."""
        filename = "MEI_B1_R1_t48_005_BF.tif"
        metadata = extract_metadata_from_filename(filename)
        self.assertEqual(metadata['condicion'], 'MEI')
        self.assertEqual(metadata['bote'], 'B1')
        self.assertEqual(metadata['replica'], 'R1')
        self.assertEqual(metadata['tiempo'], 't48')
        self.assertEqual(metadata['numero_imagen'], '005')

    def test_without_replica(self):
        """Prueba el patrón sin réplica: CONDICION_BOTE_TIEMPO_NUMERO."""
        filename = "LP-MEI_B1_t0_00_DIA.tif"
        metadata = extract_metadata_from_filename(filename)
        self.assertEqual(metadata['condicion'], 'LP-MEI')
        self.assertEqual(metadata['bote'], 'B1')
        self.assertEqual(metadata['replica'], 'R1')  # Valor por defecto
        self.assertEqual(metadata['tiempo'], 't0')
        self.assertEqual(metadata['numero_imagen'], '00')

    def test_other_pattern_without_replica(self):
        """Prueba otro patrón sin réplica."""
        filename = "MEI_B1_t24_20_BF.tif"
        metadata = extract_metadata_from_filename(filename)
        self.assertEqual(metadata['condicion'], 'MEI')
        self.assertEqual(metadata['bote'], 'B1')
        self.assertEqual(metadata['replica'], 'R1')  # Valor por defecto
        self.assertEqual(metadata['tiempo'], 't24')
        self.assertEqual(metadata['numero_imagen'], '20')

    def test_without_bote(self):
        """Prueba el patrón sin bote: CONDICION_REPLICA_TIEMPO_NUMERO."""
        filename = "MEI_R2_t36_007_BF.tif"
        metadata = extract_metadata_from_filename(filename)
        self.assertEqual(metadata['condicion'], 'MEI')
        self.assertEqual(metadata['bote'], 'B1')  # Valor por defecto
        self.assertEqual(metadata['replica'], 'R2')
        self.assertEqual(metadata['tiempo'], 't36')
        self.assertEqual(metadata['numero_imagen'], '007')

    def test_without_numero_imagen(self):
        """Prueba el patrón sin número de imagen: CONDICION_BOTE_REPLICA_TIEMPO."""
        filename = "MEI_B2_R3_t72.tif"
        metadata = extract_metadata_from_filename(filename)
        self.assertEqual(metadata['condicion'], 'MEI')
        self.assertEqual(metadata['bote'], 'B2')
        self.assertEqual(metadata['replica'], 'R3')
        self.assertEqual(metadata['tiempo'], 't72')
        # Aquí el número de imagen debería ser extraído de algún otro lugar o usar valor por defecto

    def test_minimal_pattern(self):
        """Prueba el patrón mínimo: CONDICION_REPLICA_TIEMPO."""
        filename = "CONTROL_R1_t12.tif"
        metadata = extract_metadata_from_filename(filename)
        self.assertEqual(metadata['condicion'], 'CONTROL')
        self.assertEqual(metadata['bote'], 'B1')  # Valor por defecto
        self.assertEqual(metadata['replica'], 'R1')
        self.assertEqual(metadata['tiempo'], 't12')

    def test_nonstandard_order(self):
        """Prueba un orden no estándar con identificadores claros."""
        filename = "SAMPLE_t24_B3_R2_045.jpg"
        metadata = extract_metadata_from_filename(filename)
        self.assertEqual(metadata['condicion'], 'SAMPLE')
        self.assertEqual(metadata['bote'], 'B3')        
        self.assertEqual(metadata['replica'], 'R2')
        self.assertEqual(metadata['tiempo'], 't24')
        self.assertEqual(metadata['numero_imagen'], '045')
        
    def test_hyphen_separated(self):
        """Prueba nombres con guiones en lugar de guiones bajos."""
        filename = "LP-MEI_B1_t3_10_DIA.tif"
        metadata = extract_metadata_from_filename(filename)
        self.assertEqual(metadata['condicion'], 'LP-MEI')
        self.assertEqual(metadata['bote'], 'B1')
        self.assertEqual(metadata['replica'], 'R1')  # Valor por defecto
        self.assertEqual(metadata['tiempo'], 't3')
        self.assertEqual(metadata['numero_imagen'], '10')

    def test_with_additional_info(self):
        """Prueba nombres con información adicional."""
        filename = "MEI_B1_R1_t48_005_BF_extra_info.tif"
        metadata = extract_metadata_from_filename(filename)
        self.assertEqual(metadata['condicion'], 'MEI')
        self.assertEqual(metadata['bote'], 'B1')
        self.assertEqual(metadata['replica'], 'R1')
        self.assertEqual(metadata['tiempo'], 't48')
        self.assertEqual(metadata['numero_imagen'], '005')

    def test_malformed_name(self):
        """Prueba un nombre que no cumple con los criterios mínimos."""
        filename = "imagen_microscopio.tif"
        metadata = extract_metadata_from_filename(filename)
        self.assertIsNone(metadata)  # Debería devolver None

    def test_multiple_numbers(self):
        """Prueba nombres con múltiples números que podrían confundir la extracción."""
        filename = "MEI123_B1_t24_987_BF.tif"
        metadata = extract_metadata_from_filename(filename)
        self.assertEqual(metadata['condicion'], 'MEI123')
        self.assertEqual(metadata['tiempo'], 't24')
        self.assertEqual(metadata['numero_imagen'], '987')

    def test_lowercase_identifiers(self):
        """Prueba nombres con identificadores en minúsculas."""
        filename = "control_b2_r3_t6_021.tif"
        metadata = extract_metadata_from_filename(filename)
        self.assertEqual(metadata['bote'], 'B2')
        self.assertEqual(metadata['replica'], 'R3')
        self.assertEqual(metadata['tiempo'], 't6')

    def test_mixed_case_identifiers(self):
        """Prueba nombres con identificadores en mayúsculas y minúsculas mezcladas."""
        filename = "Test_B1_r2_T24_045.png"
        metadata = extract_metadata_from_filename(filename)
        self.assertEqual(metadata['bote'], 'B1')
        self.assertEqual(metadata['replica'], 'R2')
        # Nota: dependiendo de la implementación, el tiempo podría ser t24 o T24


if __name__ == '__main__':
    unittest.main()