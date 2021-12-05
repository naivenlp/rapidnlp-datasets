import unittest


class DatasetForSequenceClassificationTest(unittest.TestCase):
    """Dataset for sequence classification"""

    def test_dataset_for_sequence_classification_pt(self):
        from naivenlp_datasets.pt import DatasetForSequenceClassification
        import torch

        dataset = DatasetForSequenceClassification.from_jsonl_files(input_files=[], vocab_file=None)
        for idx, batch in enumerate(torch.utils.data.DataLoader(dataset, num_workers=2, shuffle=True, batch_size=32)):
            print()
            print("NO.%d batch: \n%s".format(idx, batch))

    def test_dataset_for_sequence_classification_tf(self):
        pass


if __name__ == "__main__":
    unittest.main()
