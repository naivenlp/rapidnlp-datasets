import os
import unittest

from smile_datasets.question_answering import readers
from smile_datasets.question_answering.dataset import DatapipeForQuestionAnswering, DatasetForQuestionAnswering
from smile_datasets.question_answering.example import ExampleForQuestionAnswering
from smile_datasets.question_answering.parsers import ParserForQuestionAnswering

DUREADER_RUBOST_INPUT_FILE = os.path.join(os.environ["DUREADER_ROBUST_PATH"], "dev.json")
DUREADER_CHECKLIST_INPUT_FILE = os.path.join(os.environ["DUREADER_CHECKLIST_PATH"], "dev.json")
VOCAB_FILE = os.environ["BERT_VOCAB_PATH"]


class DuReaderDatasetForQuestionAnswering(DatasetForQuestionAnswering):
    """ """

    def __init__(self, input_files, vocab_file, subset="rubost", **kwargs) -> None:
        super().__init__()
        self.parser = ParserForQuestionAnswering(tokenizer=None, vocab_file=vocab_file, **kwargs)
        if subset == "rubost":
            self.instances = list(readers.read_dureader_rubost(input_files, **kwargs))
        else:
            self.instances = list(readers.read_dureader_checklist(input_files, **kwargs))
        self.examples = []
        for instance in self.instances:
            e = self.parser.parse(instance)
            if not e:
                continue
            self.examples.append(e)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> ExampleForQuestionAnswering:
        return self.examples[index]


class DatasetTest(unittest.TestCase):
    """ """

    def test_dataset_save_dureader_rubost_tfrecord(self):
        d = DuReaderDatasetForQuestionAnswering(DUREADER_RUBOST_INPUT_FILE, VOCAB_FILE)
        print("\nShowing examples: ")
        for idx, e in enumerate(d):
            print(e)
            if idx == 5:
                break

        print("Load datapipe from dataset:")
        dataset = DatapipeForQuestionAnswering.from_dataset(d)
        print()
        print(next(iter(dataset)))

        print("Save examples to tfrecord:")
        d.save_tfrecord("testdata/dureader_rubost_dev.tfrecord")

        print("Load datapipe from tfrecord files:")
        dataset = DatapipeForQuestionAnswering.from_tfrecord_files("testdata/dureader_rubost_dev.tfrecord")
        print()
        print(next(iter(dataset)))

    def test_dataset_save_dureader_checklist_tfrecord(self):
        d = DuReaderDatasetForQuestionAnswering(DUREADER_CHECKLIST_INPUT_FILE, VOCAB_FILE, subset="checklist")
        print("\nShowing examples: ")
        for idx, e in enumerate(d):
            print(e)
            if idx == 5:
                break

        print("Load datapipe from dataset:")
        dataset = DatapipeForQuestionAnswering.from_dataset(d)
        print()
        print(next(iter(dataset)))

        print("Save examples to tfrecord:")
        d.save_tfrecord("testdata/dureader_checklist_dev.tfrecord")

        print("Load datapipe from tfrecord files:")
        dataset = DatapipeForQuestionAnswering.from_tfrecord_files("testdata/dureader_checklist_dev.tfrecord")
        print()
        print(next(iter(dataset)))

    def test_datapipe_from_dureader_rubost(self):
        dataset = DatapipeForQuestionAnswering.from_dureader_robust(DUREADER_RUBOST_INPUT_FILE, vocab_file=VOCAB_FILE)
        print()
        print(next(iter(dataset)))

    def test_datapipe_from_dureader_checklist(self):
        dataset = DatapipeForQuestionAnswering.from_dureader_checklist(DUREADER_CHECKLIST_INPUT_FILE, vocab_file=VOCAB_FILE)
        print()
        print(next(iter(dataset)))


if __name__ == "__main__":
    unittest.main()
