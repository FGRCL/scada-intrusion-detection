import unittest

from numpy import linspace

from src.preprocess.featureselection import cca, ica, pca


class FeatureselectionTest(unittest.TestCase):
    def test_pca(self):
        features = linspace(0, 100, 500).reshape(-1, 5)

        result = pca(features)

        self.assertIsNotNone(result)
        self.assertEqual((100, 1), result.shape)

    def test_cca(self):
        features = linspace(0, 100, 500).reshape(-1, 5)
        labels = linspace(0, 100, 100).reshape(-1)

        result = cca(features, labels)

        self.assertIsNotNone(result)
        self.assertEqual((100, 1), result.shape)

    def test_ica(self):
        features = linspace(0, 100, 500).reshape(-1, 5)

        result = ica(features)

        self.assertIsNotNone(result)
        self.assertEqual((100, 1), result.shape)


if __name__ == '__main__':
    unittest.main()
