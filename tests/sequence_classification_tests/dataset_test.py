import os
import unittest

from smile_datasets.sequence_classification import readers
from smile_datasets.sequence_classification.dataset import DatapipeForSequenceClassifiation, DatasetForSequenceClassification
from smile_datasets.sequence_classification.example import ExampleForSequenceClassification
from smile_datasets.sequence_classification.parsers import ParserForSequenceClassification

VOCAB_FILE = os.environ["BERT_VOCAB_PATH"]


class MyDatasetForSequenceClassification(DatasetForSequenceClassification):
    """ """

    def __init__(self, input_files, vocab_file, **kwargs) -> None:
        super().__init__()
        self.parser = ParserForSequenceClassification.from_vocab_file(vocab_file=vocab_file, **kwargs)
        self.instances = readers.read_jsonl_files(input_files, **kwargs)
        self.examples = []
        for instance in self.instances:
            e = self.parser.parse(instance, **kwargs)
            if not e:
                continue
            self.examples.append(e)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class DatasetTest(unittest.TestCase):
    """Tests for sequence classification datasets"""

    def test_dataset_for_sequence_classification(self):
        d = MyDatasetForSequenceClassification("testdata/sequence_classification.jsonl", vocab_file=VOCAB_FILE)
        print("\nShowing examples:")
        for idx, e in enumerate(d):
            print(e)
        print()

        print("Save examples to tfrecord:")
        d.save_tfrecord("testdata/sequence_classification.tfrecord")

        print("Load datapipe from dataset:")
        dataset = DatapipeForSequenceClassifiation.from_dataset(d)
        print(next(iter(dataset)))

        print("Load datapipe from tfrecord:")
        dataset = DatapipeForSequenceClassifiation.from_tfrecord_files("testdata/sequence_classification.tfrecord")
        print(next(iter(dataset)))

    def test_dataset_for_seqpair_classification(self):
        d = MyDatasetForSequenceClassification("testdata/seqpair_classification.jsonl", vocab_file=VOCAB_FILE)
        print("\nShowing examples:")
        for idx, e in enumerate(d):
            print(e)
        print()

        print("Save examples to tfrecord:")
        d.save_tfrecord("testdata/seqpair_classification.tfrecord")

        print("Load datapipe from dataset:")
        dataset = DatapipeForSequenceClassifiation.from_dataset(d)
        print(next(iter(dataset)))

        print("Load datapipe from tfrecord:")
        dataset = DatapipeForSequenceClassifiation.from_tfrecord_files("testdata/seqpair_classification.tfrecord")
        print(next(iter(dataset)))

    def test_datapipe_from_jsonl_files(self):
        print()
        print("Load datapipe from sequence classification:")
        dataset = DatapipeForSequenceClassifiation.from_jsonl_files(
            "testdata/sequence_classification.jsonl", vocab_file=VOCAB_FILE
        )
        print(next(iter(dataset)))

        print("Load datapipe from seqpair classification:")
        dataset = DatapipeForSequenceClassifiation.from_jsonl_files(
            "testdata/seqpair_classification.jsonl", vocab_file=VOCAB_FILE
        )
        print(next(iter(dataset)))


if __name__ == "__main__":
    unittest.main()
