import unittest

import pyterrier as pt
if not pt.started():
    pt.init()

from ir_axioms.backend.pyterrier import TerrierIndexContext


class TestStringRepresentations(unittest.TestCase):
    def test_string_representation_of_terrier_index_context_01(self):
        # Needed for caching
        expected = 'TerrierIndexContext(index_location)'
        actual = str(TerrierIndexContext('index_location'))

        self.assertEqual(expected, actual)

    def test_string_representation_of_terrier_index_context_02(self):
        # Needed for caching
        expected = 'TerrierIndexContext(index_location)'
        actual = str(TerrierIndexContext('ignore/absolute/path/index_location'))

        self.assertEqual(expected, actual)

    def test_string_representation_of_terrier_index_context_03(self):
        # Needed for caching
        expected = 'TerrierIndexContext(index_location)'
        actual = str(TerrierIndexContext('ignore/absolute/path/index_location ignore suffix'))

        self.assertEqual(expected, actual)