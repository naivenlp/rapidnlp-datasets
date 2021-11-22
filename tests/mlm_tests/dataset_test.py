import unittest

from smile_datasets.mlm import readers
from smile_datasets.mlm.dataset import DatapipeForMaksedLanguageModel, DatasetForMaskedLanguageModel
from smile_datasets.mlm.example import ExampleForMaskedLanguageModel
from smile_datasets.mlm.parsers import ParserForMaskedLanguageModel


class MyDatasetForMaskedLanguageModel(DatasetForMaskedLanguageModel):
    """ """

    def __init__(self, input_files, vocab_file, **kwargs) -> None:
        super().__init__()
        examples = []
        parser = ParserForMaskedLanguageModel.from_vocab_file(vocab_file, **kwargs)
        for instance in readers.read_jsonl_files(input_files, **kwargs):
            e = parser.parse(instance, **kwargs)
            if not e:
                continue
            examples.append(e)
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class DatasetTest(unittest.TestCase):
    """ """

    def test_dataset_from_jsonl_files(self):
        d = MyDatasetForMaskedLanguageModel("testdata/mlm.jsonl", vocab_file="testdata/vocab.txt")
        print("Showing examples:")
        for _, e in enumerate(d):
            print(e)

        print("Load datapipe from dataset:")
        dataset = DatapipeForMaksedLanguageModel.from_dataset(d)
        print(next(iter(dataset)))

        print("Save to tfrecord:")
        d.save_tfrecord("testdata/mlm.tfrecord")

        print("Load datapipe from tfrecord:")
        dataset = DatapipeForMaksedLanguageModel.from_tfrecord_files("testdata/mlm.tfrecord")
        print(next(iter(dataset)))

    def test_datapipe_from_jsonl_files(self):
        print()
        dataset = DatapipeForMaksedLanguageModel.from_jsonl_files("testdata/mlm.jsonl", vocab_file="testdata/vocab.txt")
        print(next(iter(dataset)))


if __name__ == "__main__":
    unittest.main()
