import unittest

LABEL_MAP = {
    "O": 0,
    "B-BRAND": 1,
    "I-BRAND": 2,
}


def _label_to_id(label):
    return LABEL_MAP[label]


class DatasetTest(unittest.TestCase):
    """Dataset test for token classification"""

    def test_dataset_tf(self):
        from rapidnlp_datasets.tf import TFDatasetForTokenClassification

        dataset, d = TFDatasetForTokenClassification.from_jsonl_files(
            input_files=["testdata/token_classification.jsonl"],
            vocab_file="testdata/vocab.txt",
            label_to_id=_label_to_id,
            return_self=True,
            batch_size=4,
        )
        for idx, batch in enumerate(iter(dataset)):
            print()
            print("No.{} batch:\n{}".format(idx, batch))

        d.save_tfrecord("testdata/token_classification.tfrecord")

        dataset = TFDatasetForTokenClassification.from_tfrecord_files(
            input_files="testdata/token_classification.tfrecord",
            batch_size=4,
        )

    def test_dataset_pt(self):
        import torch
        from rapidnlp_datasets.pt import DatasetForTokenClassification

        dataset = DatasetForTokenClassification.from_jsonl_files(
            input_files=["testdata/token_classification.jsonl"],
            vocab_file="testdata/vocab.txt",
            label_to_id=_label_to_id,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=1, batch_size=4, collate_fn=dataset.batch_padding_collate
        )
        for idx, batch in enumerate(dataloader):
            print()
            print("No.{} batch:\n{}".format(idx, batch))


if __name__ == "__main__":
    unittest.main()
