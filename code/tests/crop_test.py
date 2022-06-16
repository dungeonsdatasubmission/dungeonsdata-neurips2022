import unittest

import numpy as np
import torch

from hackrl.models import baseline


class CropTest(unittest.TestCase):
    def test_crop(self):
        crop = baseline.Crop(7, 7, 3, 3)
        x = torch.arange(0, 3 * 7 * 7).view(3, 7, 7)
        y = torch.tensor([[2, 2], [4, 3], [6, 6]])  # -- x, y for first matrix in batch
        result = crop(x, y)

        expected = []

        expected.append(np.array([[8, 9, 10], [15, 16, 17], [22, 23, 24]]))

        expected.append(np.array([[66, 67, 68], [73, 74, 75], [80, 81, 82]]))

        expected.append(np.array([[138, 139, 0], [145, 146, 0], [0, 0, 0]]))

        for _ in range(x.shape[0]):
            self.assertTrue(np.array_equal(result[1], expected[1]))


if __name__ == "__main__":
    unittest.main()
