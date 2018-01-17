import unittest

import shipname

class TestShipname(unittest.TestCase):
    def test_bootstrap_gen(self):
        name = shipname.generate(0)
        self.assertIn('bootstrap', name)

    def test_detect_name(self):
        string = '000017-model.index'
        detected_name = shipname.detect_model_name(string)
        self.assertEqual(detected_name, '000017-model')
        string = '000123-golden-horse-staple-battery.index'
        detected_name = shipname.detect_model_name(string)
        self.assertEqual(detected_name, '000123-golden-horse-staple-battery')

    def test_detect_num(self):
        string = '000017-model.index'
        detected_name = shipname.detect_model_num(string)
        self.assertEqual(detected_name, 17)
