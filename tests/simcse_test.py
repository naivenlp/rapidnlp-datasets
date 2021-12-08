import unittest


class DatasetForSimCSETest(unittest.TestCase):
    """SimCSE dataset tests"""

    def test_unsup_simcse_tf(self):
        from rapidnlp_datasets.tf import TFDatasetForSimCSE, TFDatasetForUnsupSimCSE

        dataset, d = TFDatasetForUnsupSimCSE.from_jsonl_files(
            input_files=["testdata/simcse.jsonl"],
            vocab_file="testdata/vocab.txt",
            return_self=True,
        )
        for idx, batch in enumerate(iter(dataset)):
            print()
            print("No.{} batch: {}".format(idx, batch))

        d.save_tfrecord("testdata/simcse_unsup.tfrecord")

        dataset = TFDatasetForUnsupSimCSE.from_tfrecord_files(
            input_files=["testdata/simcse_unsup.tfrecord"],
            vocab_file="testdata/vocab.txt",
        )
        for idx, batch in enumerate(iter(dataset)):
            print()
            print("No.{} batch: {}".format(idx, batch))

        # v2
        dataset, d = TFDatasetForSimCSE.from_jsonl_files(
            input_files=["testdata/simcse.jsonl"],
            vocab_file="testdata/vocab.txt",
            return_self=True,
            with_pos_sequence=False,
            with_neg_sequence=False,
        )
        for idx, batch in enumerate(iter(dataset)):
            print()
            print("No.{} batch: {}".format(idx, batch))

        d.save_tfrecord("testdata/simcse_unsup_v2.tfrecord")

        dataset = TFDatasetForSimCSE.from_tfrecord_files(
            input_files=["testdata/simcse_unsup_v2.tfrecord"],
            vocab_file="testdata/vocab.txt",
            with_pos_sequence=False,
            with_neg_sequence=False,
        )
        for idx, batch in enumerate(iter(dataset)):
            print()
            print("No.{} batch: {}".format(idx, batch))

    def test_sup_simcse_tf(self):
        from rapidnlp_datasets.tf import TFDatasetForSimCSE, TFDatasetForSupervisedSimCSE

        dataset, d = TFDatasetForSupervisedSimCSE.from_jsonl_files(
            input_files=["testdata/simcse.jsonl"],
            vocab_file="testdata/vocab.txt",
            return_self=True,
        )
        for idx, batch in enumerate(iter(dataset)):
            print()
            print("No.{} batch: {}".format(idx, batch))

        d.save_tfrecord("testdata/simcse_sup.tfrecord")

        dataset = TFDatasetForSupervisedSimCSE.from_tfrecord_files(
            input_files=["testdata/simcse_sup.tfrecord"],
            vocab_file="testdata/vocab.txt",
        )
        for idx, batch in enumerate(iter(dataset)):
            print()
            print("No.{} batch: {}".format(idx, batch))

        # v2
        dataset, d = TFDatasetForSimCSE.from_jsonl_files(
            input_files=["testdata/simcse.jsonl"],
            vocab_file="testdata/vocab.txt",
            return_self=True,
            with_pos_sequence=True,
            with_neg_sequence=False,
        )
        for idx, batch in enumerate(iter(dataset)):
            print()
            print("No.{} batch: {}".format(idx, batch))

        d.save_tfrecord("testdata/simcse_sup_v2.tfrecord")

        dataset = TFDatasetForSimCSE.from_tfrecord_files(
            input_files=["testdata/simcse_sup_v2.tfrecord"],
            vocab_file="testdata/vocab.txt",
            with_pos_sequence=True,
            with_neg_sequence=False,
        )
        for idx, batch in enumerate(iter(dataset)):
            print()
            print("No.{} batch: {}".format(idx, batch))

    def test_hardneg_simcse_tf(self):
        from rapidnlp_datasets.tf import TFDatasetForHardNegativeSimCSE, TFDatasetForSimCSE

        dataset, d = TFDatasetForHardNegativeSimCSE.from_jsonl_files(
            input_files=["testdata/simcse.jsonl"],
            vocab_file="testdata/vocab.txt",
            return_self=True,
        )
        for idx, batch in enumerate(iter(dataset)):
            print()
            print("No.{} batch: {}".format(idx, batch))

        d.save_tfrecord("testdata/simcse_hardneg.tfrecord")

        dataset = TFDatasetForHardNegativeSimCSE.from_tfrecord_files(
            input_files=["testdata/simcse_hardneg.tfrecord"],
            vocab_file="testdata/vocab.txt",
        )
        for idx, batch in enumerate(iter(dataset)):
            print()
            print("No.{} batch: {}".format(idx, batch))

        # v2
        dataset, d = TFDatasetForSimCSE.from_jsonl_files(
            input_files=["testdata/simcse.jsonl"],
            vocab_file="testdata/vocab.txt",
            return_self=True,
            with_pos_sequence=True,
            with_neg_sequence=True,
        )
        for idx, batch in enumerate(iter(dataset)):
            print()
            print("No.{} batch: {}".format(idx, batch))

        d.save_tfrecord("testdata/simcse_hardneg_v2.tfrecord")

        dataset = TFDatasetForSimCSE.from_tfrecord_files(
            input_files=["testdata/simcse_hardneg_v2.tfrecord"],
            vocab_file="testdata/vocab.txt",
            with_pos_sequence=True,
            with_neg_sequence=True,
        )
        for idx, batch in enumerate(iter(dataset)):
            print()
            print("No.{} batch: {}".format(idx, batch))

    def test_simcse_pt(self):
        import torch
        from rapidnlp_datasets.pt import DatasetForSimCSE

        # unsup
        dataset = DatasetForSimCSE.from_jsonl_files(
            input_files=["testdata/simcse.jsonl"],
            vocab_file="testdata/vocab.txt",
            with_pos_sequence=False,
            with_neg_sequence=False,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=1, batch_size=4, collate_fn=dataset.batch_padding_collate
        )
        for idx, batch in enumerate(dataloader):
            print()
            print("No.{} batch: {}".format(idx, batch))

        # sup
        dataset = DatasetForSimCSE.from_jsonl_files(
            input_files=["testdata/simcse.jsonl"],
            vocab_file="testdata/vocab.txt",
            with_pos_sequence=True,
            with_neg_sequence=False,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=1, batch_size=4, collate_fn=dataset.batch_padding_collate
        )
        for idx, batch in enumerate(dataloader):
            print()
            print("No.{} batch: {}".format(idx, batch))

        # hardneg
        dataset = DatasetForSimCSE.from_jsonl_files(
            input_files=["testdata/simcse.jsonl"],
            vocab_file="testdata/vocab.txt",
            with_pos_sequence=True,
            with_neg_sequence=True,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, num_workers=1, batch_size=4, collate_fn=dataset.batch_padding_collate
        )
        for idx, batch in enumerate(dataloader):
            print()
            print("No.{} batch: {}".format(idx, batch))


if __name__ == "__main__":
    unittest.main()
