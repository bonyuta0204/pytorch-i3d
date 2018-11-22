import sys
import unittest
sys.path.append("../")  # isort:block
import mit_data  # isort:endblock
import torch



class MITDataTestCase(unittest.TestCase):
    def setUp(self):
        self.mlb = mit_data.make_label_binarizer("test/test_index.csv")
        self.dataset = mit_data.MITDataset(self.mlb, split_file="test/test_split.json")

    def test_getitem_has_correct_shape(self):
        sample_video = self.dataset[0]["video"]
        assertEqual(sample_video.size(), torch.Size([3, 90, 256, 256]),
                    "wrong size")


if __name__ == "__main__":
    unittest.main()
