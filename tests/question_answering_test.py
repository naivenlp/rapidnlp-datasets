import os
import unittest

from rapidnlp_datasets.question_answering import ExampleParserForQuestionAnswering, read_dureader_rubost


class ExampleParserTest(unittest.TestCase):
    """Example prser tests"""

    def test_wordpiece_parser(self):
        p = ExampleParserForQuestionAnswering("testdata/vocab.txt", tokenization="bert-wordpiece", do_lower_case=True)
        input_file = os.path.join(os.environ["DUREADER_ROBUST_PATH"], "dev.json")
        idx = 0
        for instance in read_dureader_rubost(input_file):
            e = p.parse_instance(instance)
            idx += 1
            if not e:
                print("index=", idx)

    def test_charlevel_parser(self):
        p = ExampleParserForQuestionAnswering("testdata/vocab.txt", tokenization="bert-charlevel", do_lower_case=True)
        input_file = os.path.join(os.environ["DUREADER_ROBUST_PATH"], "dev.json")
        idx = 0
        for instance in read_dureader_rubost(input_file):
            e = p.parse_instance(instance)
            idx += 1
            if not e:
                print("index=", idx)


class QuestionAnsweringDatasetTest(unittest.TestCase):
    """Question answering dataset test"""

    def test_question_answering_dataset_pt(self):
        import torch
        from rapidnlp_datasets.pt import DatasetForQuestionAnswering

        dataset = DatasetForQuestionAnswering.from_jsonl_files(
            input_files="testdata/qa.jsonl",
            vocab_file="testdata/vocab.txt",
        )
        for idx, batch in enumerate(
            torch.utils.data.DataLoader(
                dataset, num_workers=2, shuffle=True, batch_size=32, collate_fn=dataset.batch_padding_collate
            )
        ):
            print()
            print("NO.{} batch: \n{}".format(idx, batch))

    def test_question_answering_dataset_tf(self):
        from rapidnlp_datasets.tf import TFDatasetForQuestionAnswering

        dataset, d = TFDatasetForQuestionAnswering.from_jsonl_files(
            input_files="testdata/qa.jsonl",
            vocab_file="testdata/vocab.txt",
            return_self=True,
        )
        for idx, batch in enumerate(iter(dataset)):
            print()
            print("NO.{} batch: \n{}".format(idx, batch))

        print("Size of examples: ", len(d.examples))
        d.save_tfrecord("testdata/qa.tfrecord")

        dataset = TFDatasetForQuestionAnswering.from_tfrecord_files(
            "testdata/qa.tfrecord",
            batch_size=32,
            padding="batch",
        )
        for idx, batch in enumerate(iter(dataset)):
            print()
            print("NO.{} batch: \n{}".format(idx, batch))


if __name__ == "__main__":
    unittest.main()
