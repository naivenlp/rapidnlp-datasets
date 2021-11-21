import unittest

from smile_datasets import dataset
from smile_datasets.token_classification import readers
from smile_datasets.token_classification.dataset import DatapipeForTokenClassification, DatasetForTokenClassification
from smile_datasets.token_classification.example import ExampleForTokenClassification
from smile_datasets.token_classification.parsers import ParserForTokenClassification
from tensorflow._api.v2 import data


class MyDatasetForTokenClassification(DatasetForTokenClassification):
    """ """

    def __init__(self, input_files, feature_vocab_file, label_vocab_file, file_format="jsonl", **kwargs) -> None:
        super().__init__()
        if file_format == "jsonl":
            self.instances = readers.read_jsonl_files(input_files, **kwargs)
        elif file_format == "conll":
            self.instances = readers.read_conll_files(input_files, **kwargs)

        self.parser = ParserForTokenClassification.from_vocab_file(feature_vocab_file, label_vocab_file, **kwargs)

        examples = []
        for instance in self.instances:
            e = self.parser.parse(instance, **kwargs)
            if not e:
                continue
            examples.append(e)
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> ExampleForTokenClassification:
        return self.examples[index]


class DatasetTest(unittest.TestCase):
    """Dataset tests for token classification"""

    def test_dataset_from_jsonl(self):
        d = MyDatasetForTokenClassification(
            "testdata/token_classification.jsonl",
            feature_vocab_file="testdata/vocab.txt",
            label_vocab_file="testdata/token_classification_vocab.txt",
            file_format="jsonl",
        )
        print("\nShowing examples:")
        for _, e in enumerate(d):
            print(e)

        print("Load datapipe from dataset:")
        dataset = DatapipeForTokenClassification.from_dataset(d)
        print(next(iter(dataset)))

        print("Examples to tfrecord:")
        d.save_tfrecord("testdata/token_classification.tfrecord")

        print("Load datapipe from tfrecord:")
        dataset = DatapipeForTokenClassification.from_tfrecord_files("testdata/token_classification.tfrecord")
        print(next(iter(dataset)))

    def test_dataset_from_conll(self):
        d = MyDatasetForTokenClassification(
            "testdata/token_classification.conll.txt",
            feature_vocab_file="testdata/vocab.txt",
            label_vocab_file="testdata/token_classification_vocab.txt",
            file_format="conll",
        )
        print("\nShowing examples:")
        for _, e in enumerate(d):
            print(e)

        print("Load datapipe from dataset:")
        dataset = DatapipeForTokenClassification.from_dataset(d)
        print(next(iter(dataset)))

        print("Examples to tfrecord:")
        d.save_tfrecord("testdata/token_classification.conll.tfrecord")

        print("Load datapipe from tfrecord:")
        dataset = DatapipeForTokenClassification.from_tfrecord_files("testdata/token_classification.conll.tfrecord")
        print(next(iter(dataset)))

    def test_datapipe_from_jsonl_files(self):
        print()
        dataset = DatapipeForTokenClassification.from_jsonl_files(
            "testdata/token_classification.jsonl",
            feature_vocab_file="testdata/vocab.txt",
            label_vocab_file="testdata/token_classification_vocab.txt",
        )
        print(next(iter(dataset)))

    def test_datapipe_from_conll_files(self):
        print()
        dataset = DatapipeForTokenClassification.from_conll_files(
            "testdata/token_classification.conll.txt",
            feature_vocab_file="testdata/vocab.txt",
            label_vocab_file="testdata/token_classification_vocab.txt",
        )
        print(next(iter(dataset)))


if __name__ == "__main__":
    unittest.main()
