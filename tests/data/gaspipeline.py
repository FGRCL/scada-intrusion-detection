import unittest

from src.data.gaspipeline import load_gaspipeline_dataset


class GaspipelineDatasetTest(unittest.TestCase):
    def test_load_gaspipeline_data(self):
        x_train, x_test, y_train, y_test = load_gaspipeline_dataset()

        self.assertIsNotNone(x_train)
        self.assertIsNotNone(x_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)
        self.assertEqual((77615, 26), x_train.shape)
        self.assertEqual((19404, 26), x_test.shape)
        self.assertEqual((77615,), y_train.shape)
        self.assertEqual((19404,), y_test.shape)


if __name__ == '__main__':
    unittest.main()
