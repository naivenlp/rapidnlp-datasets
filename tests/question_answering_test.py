import os
import unittest

from tokenizers import BertWordPieceTokenizer


class QuestionAnsweringDatasetTest(unittest.TestCase):
    """Question answering dataset test"""

    def test_question_answering_dataset_pt(self):
        import torch
        from rapidnlp_datasets.question_answering import DatasetForQuestionAnswering

        tokenizer = BertWordPieceTokenizer.from_file("testdata/vocab.txt")
        dataset = DatasetForQuestionAnswering(tokenizer)

        dataset.add_jsonl_files(input_files="testdata/qa.jsonl", num_parallels=None)
        pt_dataset = dataset.to_pt_dataset()
        print()
        print("Number of examples: ", len(pt_dataset))
        for idx, batch in enumerate(
            torch.utils.data.DataLoader(
                pt_dataset,
                num_workers=2,
                shuffle=True,
                batch_size=32,
                collate_fn=pt_dataset.batch_padding_collator,
            )
        ):
            print()
            print("NO.{} batch: \n{}".format(idx, batch))

    def test_question_answering_dataset_tf(self):
        from rapidnlp_datasets.question_answering import DatasetForQuestionAnswering

        tokenizer = BertWordPieceTokenizer.from_file("testdata/vocab.txt")
        dataset = DatasetForQuestionAnswering(tokenizer)

        dataset.add_jsonl_files(input_files="testdata/qa.jsonl", num_parallels=None)
        tf_dataset = dataset.to_tf_dataset()
        for idx, batch in enumerate(iter(tf_dataset)):
            print()
            print("NO.{} batch: \n{}".format(idx, batch))

        dataset.save_tfrecord("testdata/qa.tfrecord")

        from rapidnlp_datasets.tf.question_answering_dataset import TFDatasetForQuestionAnswering

        tf_dataset = TFDatasetForQuestionAnswering.from_tfrecord_files(
            "testdata/qa.tfrecord",
            batch_size=32,
            padding="batch",
        )
        for idx, batch in enumerate(iter(tf_dataset)):
            print()
            print("NO.{} batch: \n{}".format(idx, batch))


if __name__ == "__main__":
    unittest.main()
