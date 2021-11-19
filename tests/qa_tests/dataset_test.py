import os
import unittest

from smile_datasets.qa import readers
from smile_datasets.qa.dataset import DatapipeForQuestionAnswering, DatasetForQuestionAnswering
from smile_datasets.qa.example import ExampleForQuestionAnswering
from smile_datasets.qa.parsers import ParserForQuestionAnswering

DUREADER_RUBOST_INPUT_FILE = os.path.join(os.environ["DUREADER_ROBUST_PATH"], "dev.json")
DUREADER_CHECKLIST_INPUT_FILE = os.path.join(os.environ["DUREADER_CHECKLIST_PATH"], "dev.json")
VOCAB_FILE = os.environ["BERT_VOCAB_PATH"]


class DuReaderDatasetForQuestionAnswering(DatasetForQuestionAnswering):
    """ """

    def __init__(self, input_files, vocab_file, subset="rubost", **kwargs) -> None:
        super().__init__()
        if subset == "rubost":
            self.instances = list(readers.read_dureader_rubost(input_files, **kwargs))
        else:
            self.instances = list(readers.read_dureader_checklist(input_files, **kwargs))
        self.parser = ParserForQuestionAnswering(tokenizer=None, vocab_file=vocab_file, **kwargs)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, index) -> ExampleForQuestionAnswering:
        instance = self.instances[index]
        return self.parser.parse(instance)


class DatasetTest(unittest.TestCase):
    """ """

    def test_dataset_save_dureader_rubost_tfrecord(self):
        d = DuReaderDatasetForQuestionAnswering(DUREADER_RUBOST_INPUT_FILE, VOCAB_FILE)
        for idx, e in enumerate(d):
            print(e)
            if idx == 5:
                break

        dataset = DatapipeForQuestionAnswering.from_dataset(d)
        print()
        print(next(iter(dataset)))

        d.save_tfrecord("testdata/dureader_rubost_dev.tfrecord")

        dataset = DatapipeForQuestionAnswering.from_tfrecord_files("testdata/dureader_rubost_dev.tfrecord")
        print()
        print(next(iter(dataset)))

    def test_dataset_save_dureader_checklist_tfrecord(self):
        d = DuReaderDatasetForQuestionAnswering(DUREADER_CHECKLIST_INPUT_FILE, VOCAB_FILE, subset="checklist")
        for idx, e in enumerate(d):
            print(e)
            if idx == 5:
                break

        dataset = DatapipeForQuestionAnswering.from_dataset(d)
        print()
        print(next(iter(dataset)))

        d.save_tfrecord("testdata/dureader_checklist_dev.tfrecord")

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
