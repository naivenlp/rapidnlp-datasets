import unittest

from rapidnlp_datasets.masked_lm import DatasetForMaskedLanguageModel
from tokenizers import BertWordPieceTokenizer


class DatasetTest(unittest.TestCase):
    """Dataset test"""

    def test_dataset_tf(self):
        tokenizer = BertWordPieceTokenizer.from_file("testdata/vocab.txt")
        dataset = DatasetForMaskedLanguageModel(tokenizer)
        dataset.add_jsonl_files(
            input_files=["testdata/mlm.jsonl"],
            num_parallels=None,
        )

        tf_dataset = dataset.to_tf_dataset(
            batch_size=4,
        )
        for idx, batch in enumerate(iter(tf_dataset)):
            print()
            print("No.{} batch:\n{}".format(idx, batch))

        dataset.save_tfrecord("testdata/mlm.tfrecord")

        from rapidnlp_datasets.tf.masked_lm_dataset import TFDatasetForMaksedLanguageModel

        dataset = TFDatasetForMaksedLanguageModel.from_tfrecord_files(
            input_files="testdata/mlm.tfrecord",
            batch_size=4,
        )
        for idx, batch in enumerate(iter(dataset)):
            print()
            print("No.{} batch:\n{}".format(idx, batch))

    def test_dataset_pt(self):
        import torch

        tokenizer = BertWordPieceTokenizer.from_file("testdata/vocab.txt")
        dataset = DatasetForMaskedLanguageModel(tokenizer)
        dataset.add_jsonl_files(
            input_files=["testdata/mlm.jsonl"],
            num_parallels=None,
        )
        pt_dataset = dataset.to_pt_dataset()

        dataloader = torch.utils.data.DataLoader(
            pt_dataset, batch_size=4, num_workers=1, collate_fn=pt_dataset.batch_padding_collator
        )
        for idx, batch in enumerate(dataloader):
            print()
            print("No.{} batch:\n{}".format(idx, batch))


if __name__ == "__main__":
    unittest.main()
