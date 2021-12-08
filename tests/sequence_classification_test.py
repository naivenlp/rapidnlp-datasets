import unittest


class DatasetForSequenceClassificationTest(unittest.TestCase):
    """Dataset for sequence classification"""

    def test_dataset_for_sequence_classification_pt(self):
        import torch
        from rapidnlp_datasets.pt import DatasetForSequenceClassification

        dataset = DatasetForSequenceClassification.from_jsonl_files(
            input_files=["testdata/sequence_classification.jsonl"],
            vocab_file="testdata/vocab.txt",
        )
        for idx, batch in enumerate(
            torch.utils.data.DataLoader(
                dataset, num_workers=2, shuffle=True, batch_size=32, collate_fn=dataset.batch_padding_collate
            )
        ):
            print()
            print("NO.{} batch: \n{}".format(idx, batch))

    def test_dataset_for_sequence_classification_tf(self):
        from rapidnlp_datasets.tf import TFDatasetForSequenceClassifiation

        dataset, d = TFDatasetForSequenceClassifiation.from_jsonl_files(
            input_files=["testdata/sequence_classification.jsonl"],
            vocab_file="testdata/vocab.txt",
            return_self=True,
        )
        for idx, batch in enumerate(iter(dataset)):
            print()
            print("NO.{} batch: \n{}".format(idx, batch))

        print("Size of examples: ", len(d.examples))
        d.save_tfrecord("testdata/sequence_classification.tfrecord")

        dataset = TFDatasetForSequenceClassifiation.from_tfrecord_files("testdata/sequence_classification.tfrecord")
        for idx, batch in enumerate(iter(dataset)):
            print()
            print("NO.{} batch: \n{}".format(idx, batch))


if __name__ == "__main__":
    unittest.main()
