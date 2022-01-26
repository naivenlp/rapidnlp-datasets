import unittest

from rapidnlp_datasets.simcse import DatasetForSimCSE
from tokenizers import BertWordPieceTokenizer


class DatasetForSimCSETest(unittest.TestCase):
    """SimCSE dataset tests"""

    def test_unsup_simcse_pt(self):
        dataset = DatasetForSimCSE(
            tokenizer=BertWordPieceTokenizer.from_file("testdata/vocab.txt"),
            with_positive_sequence=False,
            with_negative_sequence=False,
            max_sequence_length=128,
        )
        dataset.add_jsonl_files("testdata/simcse.jsonl", num_parallels=None)

        import torch

        pt_dataset = dataset.to_pt_dataset()
        for idx, batch in enumerate(
            torch.utils.data.DataLoader(
                pt_dataset, num_workers=2, shuffle=True, batch_size=32, collate_fn=pt_dataset.batch_padding_collator
            )
        ):
            print()
            print("NO.{} batch: \n{}".format(idx, batch))

    def test_sup_simcse_pt(self):
        dataset = DatasetForSimCSE(
            tokenizer=BertWordPieceTokenizer.from_file("testdata/vocab.txt"),
            with_positive_sequence=True,
            with_negative_sequence=False,
            max_sequence_length=128,
        )
        dataset.add_jsonl_files("testdata/simcse.jsonl", positive_sequence_column="pos_sequence", num_parallels=None)

        import torch

        pt_dataset = dataset.to_pt_dataset()
        for idx, batch in enumerate(
            torch.utils.data.DataLoader(
                pt_dataset, num_workers=2, shuffle=True, batch_size=32, collate_fn=pt_dataset.batch_padding_collator
            )
        ):
            print()
            print("NO.{} batch: \n{}".format(idx, batch))

    def test_hardneg_simcse_pt(self):
        dataset = DatasetForSimCSE(
            tokenizer=BertWordPieceTokenizer.from_file("testdata/vocab.txt"),
            with_positive_sequence=True,
            with_negative_sequence=True,
            max_sequence_length=128,
        )
        dataset.add_jsonl_files(
            "testdata/simcse.jsonl",
            positive_sequence_column="pos_sequence",
            negative_sequence_column="neg_sequence",
            num_parallels=None,
        )

        import torch

        pt_dataset = dataset.to_pt_dataset()
        for idx, batch in enumerate(
            torch.utils.data.DataLoader(
                pt_dataset, num_workers=2, shuffle=True, batch_size=32, collate_fn=pt_dataset.batch_padding_collator
            )
        ):
            print()
            print("NO.{} batch: \n{}".format(idx, batch))

    def test_unsup_simcse_tf(self):
        dataset = DatasetForSimCSE(
            tokenizer=BertWordPieceTokenizer.from_file("testdata/vocab.txt"),
            with_positive_sequence=False,
            with_negative_sequence=False,
            max_sequence_length=128,
        )
        dataset.add_jsonl_files("testdata/simcse.jsonl", num_parallels=None)

        tf_dataset = dataset.to_tf_dataset()
        for idx, batch in enumerate(iter(tf_dataset)):
            print()
            print("NO.{} batch: \n{}".format(idx, batch))

        dataset.save_tfrecord("testdata/simcse.tfrecord")

        from rapidnlp_datasets.tf import TFDatasetForSimCSE

        tf_dataset = TFDatasetForSimCSE.from_tfrecord_files(
            "testdata/simcse.tfrecord",
            with_positive_sequence=False,
            with_negative_sequence=False,
            max_sequence_length=128,
        )
        for idx, batch in enumerate(iter(tf_dataset)):
            print()
            print("NO.{} batch: \n{}".format(idx, batch))

    def test_sup_simcse_tf(self):
        dataset = DatasetForSimCSE(
            tokenizer=BertWordPieceTokenizer.from_file("testdata/vocab.txt"),
            with_positive_sequence=True,
            with_negative_sequence=False,
            max_sequence_length=128,
        )
        dataset.add_jsonl_files("testdata/simcse.jsonl", positive_sequence_column="pos_sequence", num_parallels=None)

        tf_dataset = dataset.to_tf_dataset()
        for idx, batch in enumerate(iter(tf_dataset)):
            print()
            print("NO.{} batch: \n{}".format(idx, batch))

        dataset.save_tfrecord("testdata/simcse_sup.tfrecord")

        from rapidnlp_datasets.tf import TFDatasetForSimCSE

        tf_dataset = TFDatasetForSimCSE.from_tfrecord_files(
            "testdata/simcse_sup.tfrecord",
            with_positive_sequence=True,
            with_negative_sequence=False,
            max_sequence_length=128,
        )
        for idx, batch in enumerate(iter(tf_dataset)):
            print()
            print("NO.{} batch: \n{}".format(idx, batch))

    def test_hardneg_simcse_tf(self):
        dataset = DatasetForSimCSE(
            tokenizer=BertWordPieceTokenizer.from_file("testdata/vocab.txt"),
            with_positive_sequence=True,
            with_negative_sequence=True,
            max_sequence_length=128,
        )
        dataset.add_jsonl_files(
            "testdata/simcse.jsonl",
            positive_sequence_column="pos_sequence",
            negative_sequence_column="neg_sequence",
            num_parallels=None,
        )

        tf_dataset = dataset.to_tf_dataset()
        for idx, batch in enumerate(iter(tf_dataset)):
            print()
            print("NO.{} batch: \n{}".format(idx, batch))

        dataset.save_tfrecord("testdata/simcse_hardneg.tfrecord")

        from rapidnlp_datasets.tf import TFDatasetForSimCSE

        tf_dataset = TFDatasetForSimCSE.from_tfrecord_files(
            "testdata/simcse_hardneg.tfrecord",
            with_positive_sequence=True,
            with_negative_sequence=True,
            max_sequence_length=128,
        )
        for idx, batch in enumerate(iter(tf_dataset)):
            print()
            print("NO.{} batch: \n{}".format(idx, batch))


if __name__ == "__main__":
    unittest.main()
