import unittest

from numpy import array, concatenate, count_nonzero, ones, zeros

from src.preprocess.dataset import balance_dataset


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


if __name__ == '__main__':
    unittest.main()
