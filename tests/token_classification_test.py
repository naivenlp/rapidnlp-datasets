import unittest

from rapidnlp_datasets import DatasetForTokenClassification
from rapidnlp_datasets.charlevel_tokenizer import BertCharLevelTokenizer

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
        tokenizer = BertCharLevelTokenizer.from_file("testdata/vocab.txt")
        dataset = DatasetForTokenClassification(
            tokenizer,
        )
        dataset.add_jsonl_files("testdata/token_classification.jsonl", label2id=_label_to_id)

        tf_dataset = dataset.to_tf_dataset()
        for idx, batch in enumerate(iter(tf_dataset)):
            print()
            print("No.{} batch:\n{}".format(idx, batch))

        dataset.save_tfrecord("testdata/token_classification.tfrecord")

        from rapidnlp_datasets.tf import TFDatasetForTokenClassification

        tf_dataset = TFDatasetForTokenClassification.from_tfrecord_files(
            input_files="testdata/token_classification.tfrecord",
            batch_size=4,
        )
        for idx, batch in enumerate(iter(tf_dataset)):
            print()
            print("No.{} batch:\n{}".format(idx, batch))

    def test_dataset_pt(self):
        tokenizer = BertCharLevelTokenizer.from_file("testdata/vocab.txt")
        dataset = DatasetForTokenClassification(
            tokenizer,
        )
        dataset.add_jsonl_files("testdata/token_classification.jsonl", label2id=_label_to_id)

        pt_dataset = dataset.to_pt_dataset()

        import torch

        dataloader = torch.utils.data.DataLoader(
            pt_dataset, num_workers=1, batch_size=4, collate_fn=pt_dataset.batch_padding_collator
        )
        for idx, batch in enumerate(dataloader):
            print()
            print("No.{} batch:\n{}".format(idx, batch))


if __name__ == "__main__":
    unittest.main()
