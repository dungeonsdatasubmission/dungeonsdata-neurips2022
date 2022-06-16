import unittest

import torch

from hackrl.models import baseline


class RewardNormTest(unittest.TestCase):
    def test_rewardnorm(self):
        # https://math.stackexchange.com/a/103025/5051 and
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        net = baseline.NetHackNet()
        all_data = []
        for i in range(100):
            # number of new data points.
            new_count = torch.randint(1, 16, (1,))
            # new data points
            new_data = torch.rand(new_count)

            net.update_running_moments(new_data)
            all_data += list(new_data)
            if i >= 10:
                self.assertAlmostEqual(
                    net.get_running_std().item(),
                    torch.std(torch.Tensor(all_data), unbiased=False).item(),
                    places=6,
                )


if __name__ == "__main__":
    unittest.main()
