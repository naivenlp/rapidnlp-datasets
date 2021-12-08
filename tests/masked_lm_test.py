import unittest


class DatasetTest(unittest.TestCase):
    """Dataset test"""

    def test_dataset_tf(self):
        from rapidnlp_datasets.tf import TFDatasetForMaksedLanguageModel

        dataset, d = TFDatasetForMaksedLanguageModel.from_jsonl_files(
            input_files=["testdata/mlm.jsonl"],
            vocab_file="testdata/vocab.txt",
            return_self=True,
            batch_size=4,
        )
        for idx, batch in enumerate(iter(dataset)):
            print()
            print("No.{} batch:\n{}".format(idx, batch))

        d.save_tfrecord("testdata/mlm.tfrecord")

        dataset = TFDatasetForMaksedLanguageModel.from_tfrecord_files(
            input_files="testdata/mlm.tfrecord",
            batch_size=4,
        )
        for idx, batch in enumerate(iter(dataset)):
            print()
            print("No.{} batch:\n{}".format(idx, batch))

    def test_dataset_pt(self):
        import torch
        from rapidnlp_datasets.pt import DatasetForMaskedLanguageModel

        dataset = DatasetForMaskedLanguageModel.from_jsonl_files(
            input_files=["testdata/mlm.jsonl"],
            vocab_file="testdata/vocab.txt",
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=4, num_workers=1, collate_fn=dataset.batch_padding_collate
        )
        for idx, batch in enumerate(dataloader):
            print()
            print("No.{} batch:\n{}".format(idx, batch))


if __name__ == "__main__":
    unittest.main()
