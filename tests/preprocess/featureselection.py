import unittest

from numpy import linspace

from src.preprocess.featureselection import get_first_cca_feature, get_first_ica_feature, get_first_pca_feature


class FeatureselectionTest(unittest.TestCase):
    def test_pca(self):
        features = linspace(0, 100, 500).reshape(-1, 5)

        result = get_first_pca_feature(features)

        self.assertIsNotNone(result)
        self.assertEqual((100, 1), result.shape)

    def test_cca(self):
        features = linspace(0, 100, 500).reshape(-1, 5)
        labels = linspace(0, 100, 100).reshape(-1)

        result = get_first_cca_feature(features, labels)

        self.assertIsNotNone(result)
        self.assertEqual((100, 1), result.shape)

    def test_ica(self):
        features = linspace(0, 100, 500).reshape(-1, 5)

        result = get_first_ica_feature(features)

        self.assertIsNotNone(result)
        self.assertEqual((100, 1), result.shape)


if __name__ == '__main__':
    unittest.main()
