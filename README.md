# smile-datasets

![Python package](https://github.com/luozhouyang/smile-datasets/workflows/Python%20package/badge.svg)
[![PyPI version](https://badge.fury.io/py/smile-datasets.svg)](https://badge.fury.io/py/smile-datasets)
[![Python](https://img.shields.io/pypi/pyversions/smile-datasets.svg?style=plastic)](https://badge.fury.io/py/smile-datasets)


La**S**t **mile** Datasets: Use `tf.data` to solve the last mile data loading problem for tensorflow.

If you want to load public datasets, try:

* [tensorflow/datasets](https://github.com/tensorflow/datasets)
* [huggingface/datasets](https://github.com/huggingface/datasets)

If you want to load local, personal dataset with minimized boilerplate, use **Smile Dataset**!

## Support Matrix

| task                   | supported  | core abstractions |
|:-----------------------|:-----------|:------------------|
| question answering     | [x]        | `ExampleForQuestionAnswering`, `DatasetForQuestionAnswering`, `DatapipeForQuestionAnswering`|
| masked language model  | [x]        | `ExampleForMaskedLanguageModel`, `DatasetForMaskedLanguageModel`, `DatapipeForMaskedLanguageModel`|
| sequence classification| [x]        | `ExampleForSequenceClassification`, `DatasetForSequenceClassification`, `DatapipeForSequenceClassification`|
| token classification   | [x]        | `ExampleForTokenClassification`, `DatasetForTokenClassification`, `DatapipeForTokenClassification`|
| unsupervised simcse    | [x]        | `ExampleForUnsupervisedSimCSE`, `DatasetForUnsupervisedSimCSE`, `DatapipeForUnsupervisedSimCSE`|
| supervised simcse      | [x]        | `ExampleForSupervisedSimCSE`, `DatasetForSupervisedSimCSE`, `DatapipeForSupervisedSimCSE`|
| hard negative simcse   | [x]        | `ExampleForHardNegativeSimCSE`, `DatasetForHardNegativeSimCSE`, `DatapipeForHardNegativeSimCSE`|


## Usage

All datapipes for different tasks has the same interface.

Here is an example for question answering task, but you can use datapipe the same way for other tasks.

### Example for Question Answering

```python

from smile_datasets import DatasetForQuestionAnswering, DatapipeForQuestionAnswering

# each line is a JSON {"sequece": "我喜欢自然语言处理(NLP)"}
train_input_jsonl_files = ["data/train.jsonl"]
train_dataset = DatapipeForQuestionAnswering.from_jsonl_files(
    input_files=train_input_jsonl_files, 
    vocab_file="bert/vocab.txt",
    batch_size=32,
)

# check dataset
print(next(iter(train_dataset)))

# model = build_keras_model(...)
# model.compile(...)
# train model
model.fit(train_dataset, callbacks=[...])

```


For maximum flexibility, you can always subclass `DatasetForQuestionAnswering` to load your dataset, just like `torch.utils.data.Dataset`:

```python
from smile_datasets import DatasetForQuestionAnswering, DatapipeForQuestionAnswering, ParserForQuestionAnswering

class DuReaderDatasetForQuestionAnswering(DatasetForQuestionAnswering):
    """Dataset reader for DuReader dataset."""

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


dataset = DuReaderDatasetForQuestionAnswering(input_files=["data/trian.jsonl"], vocab_file="bert/vocab.txt")
train_dataset = DatapipeForQuestionAnswering.from_dataset(dataset, batch_size=32)

# check dataset
print(next(iter(train_dataset)))

# model = build_keras_model(...)
# model.compile(...)
# train model
model.fit(train_dataset, callbacks=[...])
```

For better performance, you can convert `dataset` to `tfrecord` ahead of time, and then build datapipe from tfrecord files directly:

```python
# save dataset in tfrecord format
dataset.save_tfrecord(output_files="data/train.tfrecord")

# build datapipe from tfrecord files
train_dataset = DatapipeForQuestionAnswering.from_tfrecord_files(input_files="data/train.tfrecord", batch_size=32)

# check dataset
print(next(iter(train_dataset)))

# model = build_keras_model(...)
# model.compile(...)
# train model
model.fit(train_dataset, callbacks=[...])
```
