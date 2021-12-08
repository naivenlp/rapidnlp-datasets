# rapidnlp-datasets

![Python package](https://github.com/naivenlp/rapidnlp-datasets/workflows/Python%20package/badge.svg)
[![PyPI version](https://badge.fury.io/py/rapidnlp-datasets.svg)](https://badge.fury.io/py/rapidnlp-datasets)
[![Python](https://img.shields.io/pypi/pyversions/rapidnlp-datasets.svg?style=plastic)](https://badge.fury.io/py/rapidnlp-datasets)


Data pipelines for both **TensorFlow** and **PyTorch** !

If you want to load public datasets, try:

* [tensorflow/datasets](https://github.com/tensorflow/datasets)
* [huggingface/datasets](https://github.com/huggingface/datasets)

If you want to load local, personal dataset with minimized boilerplate, use **rapidnlp-datasets**!

## installation

```bash
pip install -U rapidnlp-datasets
```

> If you work with PyTorch, you should install [PyTorch](https://pytorch.org/get-started/locally/) first.

> If you work with TensorFlow, you should install [TensorFlow](https://github.com/tensorflow/tensorflow) first.

## Usage

Here are few examples to show you how to use this library.

* [QuickStart: Sequence Classification Task](#sequence-classification-quickstart)
* [QuickStart: Question Answering Task](#question-answering-quickstart)
* [QuickStart: Token Classification Task](#token-classification-quickstart)
* [QuickStart: Masked Language Model Task](#masked-language-model-quickstart)
* [QuickStart: SimCSE(Sentence Embedding)](#simcse-quickstart)

### sequence-classification-quickstart

In PyTorch,

```bash
>>> import torch
>>> from rapidnlp_datasets.pt import DatasetForSequenceClassification
>>> dataset = DatasetForSequenceClassification.from_jsonl_files(
        input_files=["testdata/sequence_classification.jsonl"],
        vocab_file="testdata/vocab.txt",
    )
>>> dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=32, collate_fn=dataset.batch_padding_collate)
>>> for idx, batch in enumerate(dataloader):
...     print("No.{} batch: \n{}".format(idx, batch))
... 
```

In TensorFlow,

```bash
>>> from rapidnlp_datasets.tf import TFDatasetForSequenceClassifiation
>>> dataset, d = TFDatasetForSequenceClassifiation.from_jsonl_files(
        input_files=["testdata/sequence_classification.jsonl"],
        vocab_file="testdata/vocab.txt",
        return_self=True,
    )
>>> for idx, batch in enumerate(iter(dataset)):
...     print("No.{} batch: \n{}".format(idx, batch))
... 
```

Especially, you can save dataset to `tfrecord` format when working with TensorFlow, and then build dataset from tfrecord files directly!

```bash
>>> d.save_tfrecord("testdata/sequence_classification.tfrecord")
2021-12-08 14:52:41,295    INFO             utils.py  128] Finished to write 2 examples to tfrecords.
>>> dataset = TFDatasetForSequenceClassifiation.from_tfrecord_files("testdata/sequence_classification.tfrecord")
>>> for idx, batch in enumerate(iter(dataset)):
...     print("No.{} batch: \n{}".format(idx, batch))
... 
```

### question-answering-quickstart

In PyTorch:
```bash
>>> import torch
>>> from rapidnlp_datasets.pt import DatasetForQuestionAnswering
>>>
>>> dataset = DatasetForQuestionAnswering.from_jsonl_files(
        input_files="testdata/qa.jsonl",
        vocab_file="testdata/vocab.txt",
    )
>>> dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=32, collate_fn=dataset.batch_padding_collate)
>>> for idx, batch in enumerate(dataloader):
...     print("No.{} batch: \n{}".format(idx, batch))
... 
```

In TensorFlow,

```bash
>>> from rapidnlp_datasets.tf import TFDatasetForQuestionAnswering
>>> dataset, d = TFDatasetForQuestionAnswering.from_jsonl_files(
        input_files="testdata/qa.jsonl",
        vocab_file="testdata/vocab.txt",
        return_self=True,
    )
2021-12-08 15:09:06,747    INFO question_answering_dataset.py  101] Read 3 examples in total.
>>> for idx, batch in enumerate(iter(dataset)):
        print()
        print("NO.{} batch: \n{}".format(idx, batch))
... 
```

Especially, you can save dataset to `tfrecord` format when working with TensorFlow, and then build dataset from tfrecord files directly!

```bash
>>> d.save_tfrecord("testdata/qa.tfrecord")
2021-12-08 15:09:31,329    INFO             utils.py  128] Finished to write 3 examples to tfrecords.
>>> dataset = TFDatasetForQuestionAnswering.from_tfrecord_files(
        "testdata/qa.tfrecord",
        batch_size=32,
        padding="batch",
    )
>>> for idx, batch in enumerate(iter(dataset)):
        print()
        print("NO.{} batch: \n{}".format(idx, batch))
... 

```


### token-classification-quickstart


### masked-language-models-quickstart


### simcse-quickstart

