import unittest

from rapidnlp_datasets.sequence_classification import DatasetForSequenceClassification
from tokenizers import BertWordPieceTokenizer


class DatasetForSequenceClassificationTest(unittest.TestCase):
    """Dataset for sequence classification"""

    def test_dataset_for_sequence_classification_pt(self):
        import torch

        tokenizer = BertWordPieceTokenizer.from_file("testdata/vocab.txt")
        dataset = DatasetForSequenceClassification(tokenizer)
        dataset.add_jsonl_files(
            input_files=["testdata/sequence_classification.jsonl"],
            num_parallels=None,
        )
        pt_dataset = dataset.to_pt_dataset()
        for idx, batch in enumerate(
            torch.utils.data.DataLoader(
                pt_dataset, num_workers=2, shuffle=True, batch_size=32, collate_fn=pt_dataset.batch_padding_collator
            )
        ):
            print()
            print("NO.{} batch: \n{}".format(idx, batch))

    def test_dataset_for_sequence_classification_tf(self):
        tokenizer = BertWordPieceTokenizer.from_file("testdata/vocab.txt")
        dataset = DatasetForSequenceClassification(tokenizer)
        dataset.add_jsonl_files(
            input_files=["testdata/sequence_classification.jsonl"],
            num_parallels=None,
        )
        tf_dataset = dataset.to_tf_dataset(batch_size=32)
        for idx, batch in enumerate(iter(tf_dataset)):
            print()
            print("NO.{} batch: \n{}".format(idx, batch))

        print("Size of examples: ", len(dataset.examples))
        dataset.save_tfrecord("testdata/sequence_classification.tfrecord")

        from rapidnlp_datasets.tf import TFDatasetForSequenceClassifiation

        dataset = TFDatasetForSequenceClassifiation.from_tfrecord_files("testdata/sequence_classification.tfrecord")
        for idx, batch in enumerate(iter(dataset)):
            print()
            print("NO.{} batch: \n{}".format(idx, batch))


if __name__ == "__main__":
    unittest.main()
