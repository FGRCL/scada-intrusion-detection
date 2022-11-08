import unittest

from numpy import array, concatenate, count_nonzero, linspace, mean, ones, zeros
from sklearn.preprocessing import StandardScaler

from src.preprocess.dataset import balance_dataset, convert_binary_labels, remove_missing_values, scale_features


class PreprocessDatasetTest(unittest.TestCase):
    def test_balance_dataset(self):
        features = concatenate(
            (zeros((18, 3)), ones((36, 3))),
            axis=0
        )
        labels = concatenate(
            (zeros((18,)), ones((36,))),
            axis=0
        )
        features_balanced, labels_balanced = balance_dataset(features, labels)

        self.assertIsNotNone(features_balanced)
        self.assertIsNotNone(labels_balanced)
        self.assertEqual(count_nonzero(labels_balanced==0), count_nonzero(labels_balanced==1))

    def test_remove_missing_values(self):
        features = array([
            [None, 1, 1],
            [1, 1, 1],
            [1, None, 1],
            [1, 1, 1],
        ])
        labels = array([1, 1, 1, None, ])
        features_clean, labels_clean = remove_missing_values(features, labels)

        self.assertIsNotNone(features_clean)
        self.assertIsNotNone(labels_clean)
        self.assertCountEqual((2, 3), features_clean.shape)
        self.assertCountEqual((3,), labels_clean.shape)

    def test_scale_features(self):
        features = linspace(1, 100, 50).reshape(-1, 5)
        features_scaled, scaler = scale_features(features)

        self.assertIsNotNone(features_scaled)
        for m in mean(features_scaled, axis=0):
            self.assertAlmostEqual(0, m, 5)
        self.assertTrue(isinstance(scaler, StandardScaler))

    def test_convert_binary_labels(self):
        labels = array([
            0,
            1,
            2,
            7,
            0,
            9
        ])
        result = convert_binary_labels(labels)

        expected_labels = array([
            0,
            1,
            1,
            1,
            0,
            1
        ])
        self.assertIsNotNone(result)
        self.assertTrue(all(expected_labels == result))

if __name__ == '__main__':
    unittest.main()
