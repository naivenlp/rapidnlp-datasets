import unittest

from smile_datasets.simcse import readers
from smile_datasets.simcse.example import ExampleForHardNegativeSimCSE, ExampleForSupervisedSimCSE, ExampleForUnsupervisedSimCSE
from smile_datasets.simcse.hardneg_dataset import DatapipeForHardNegativeSimCSE, DatasetForHardNegativeSimCSE
from smile_datasets.simcse.parsers import ParserForHardNegSimCSE, ParserForSupervisedSimCSE, ParserForUnsupervisedSimCSE
from smile_datasets.simcse.sup_dataset import DatapipeForSupervisedSimCSE, DatasetForSupervisedSimCSE
from smile_datasets.simcse.unsup_dataset import DatapipeForUnsupervisedSimCSE, DatasetForUnsupervisedSimCSE


class MyDatasetForUnsupSimCSE(DatasetForUnsupervisedSimCSE):
    """ """

    def __init__(self, input_files, vocab_file, **kwargs) -> None:
        super().__init__()
        examples = []
        parser = ParserForUnsupervisedSimCSE.from_vocab_file(vocab_file, **kwargs)
        for instance in readers.read_jsonl_files_for_unsup_simcse(input_files, **kwargs):
            e = parser.parse(instance, **kwargs)
            if not e:
                continue
            examples.append(e)
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> ExampleForUnsupervisedSimCSE:
        return self.examples[index]


class MyDatasetForSupervisedSimCSE(DatasetForSupervisedSimCSE):
    """ """

    def __init__(self, input_files, vocab_file, **kwargs) -> None:
        super().__init__()
        examples = []
        parser = ParserForSupervisedSimCSE.from_vocab_file(vocab_file, **kwargs)
        for instance in readers.read_jsonl_files_for_supervised_simcse(input_files, **kwargs):
            e = parser.parse(instance, **kwargs)
            if not e:
                continue
            examples.append(e)
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> ExampleForSupervisedSimCSE:
        return self.examples[index]


class MyDatasetForHardNegativeSimCSE(DatasetForHardNegativeSimCSE):
    """ """

    def __init__(self, input_files, vocab_file, **kwargs) -> None:
        super().__init__()
        examples = []
        parser = ParserForHardNegSimCSE.from_vocab_file(vocab_file, **kwargs)
        for instance in readers.read_jsonl_files_for_hardneg_simcse(input_files, **kwargs):
            e = parser.parse(instance, **kwargs)
            if not e:
                continue
            examples.append(e)
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index) -> ExampleForHardNegativeSimCSE:
        return self.examples[index]


class DatasetTest(unittest.TestCase):
    """ """

    def test_unsup_dataset_from_jsonl(self):
        d = MyDatasetForUnsupSimCSE("testdata/simcse.jsonl", vocab_file="testdata/vocab.txt")
        print()
        for _, e in enumerate(d):
            print(e)

        print("Load datapipe from dataset:")
        dataset = DatapipeForUnsupervisedSimCSE.from_dataset(d)
        print(next(iter(dataset)))

        print("Save tfrecord:")
        d.save_tfrecord("testdata/simcse.unsup.tfrecord")

        print("Load datapipe from tfrecord:")
        dataset = DatapipeForUnsupervisedSimCSE.from_tfrecord_files("testdata/simcse.unsup.tfrecord")
        print(next(iter(dataset)))

    def test_sup_datapipe_from_jsonl(self):
        d = MyDatasetForSupervisedSimCSE("testdata/simcse.jsonl", vocab_file="testdata/vocab.txt")
        print()
        for _, e in enumerate(d):
            print(e)

        print("Load datapipe from dataset:")
        dataset = DatapipeForSupervisedSimCSE.from_dataset(d)
        print(next(iter(dataset)))

        print("Save tfrecord:")
        d.save_tfrecord("testdata/simcse.sup.tfrecord")

        print("Load datapipe from tfrecord:")
        dataset = DatapipeForSupervisedSimCSE.from_tfrecord_files("testdata/simcse.sup.tfrecord")
        print(next(iter(dataset)))

    def test_hardnef_dataset_from_jsonl(self):
        d = MyDatasetForHardNegativeSimCSE("testdata/simcse.jsonl", vocab_file="testdata/vocab.txt")
        print()
        for _, e in enumerate(d):
            print(e)

        print("Load datapipe from dataset:")
        dataset = DatapipeForHardNegativeSimCSE.from_dataset(d)
        print(next(iter(dataset)))

        print("Save tfrecord:")
        d.save_tfrecord("testdata/simcse.hardneg.tfrecord")

        print("Load datapipe from tfrecord:")
        dataset = DatapipeForHardNegativeSimCSE.from_tfrecord_files("testdata/simcse.hardneg.tfrecord")
        print(next(iter(dataset)))

    def test_unsup_datapipe_from_jsonl(self):
        print()
        dataset = DatapipeForUnsupervisedSimCSE.from_jsonl_files("testdata/simcse.jsonl", vocab_file="testdata/vocab.txt")
        print(next(iter(dataset)))

    def test_sup_dataset_from_jsonl(self):
        print()
        dataset = DatapipeForSupervisedSimCSE.from_jsonl_files("testdata/simcse.jsonl", vocab_file="testdata/vocab.txt")
        print(next(iter(dataset)))

    def test_hardneg_datapipe_from_jsonl(self):
        print()
        dataset = DatapipeForHardNegativeSimCSE.from_jsonl_files("testdata/simcse.jsonl", vocab_file="testdata/vocab.txt")
        print(next(iter(dataset)))


if __name__ == "__main__":
    unittest.main()
