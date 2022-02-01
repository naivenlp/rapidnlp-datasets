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

## Usage

Here are few examples to show you how to use this library.

* [QuickStart: Sequence Classification Task](#sequence-classification-quickstart)
* [QuickStart: Question Answering Task](#question-answering-quickstart)
* [QuickStart: Token Classification Task](#token-classification-quickstart)
* [QuickStart: Masked Language Model Task](#masked-language-model-quickstart)
* [QuickStart: SimCSE(Sentence Embedding)](#simcse-quickstart)

### sequence-classification-quickstart

```python
import torch
from tokenizers import BertWordPieceTokenizer
from rapidnlp_datasets import DatasetForSequenceClassification
from rapidnlp_datasets.tf import TFDatasetForSequenceClassifiation

# build dataset
tokenizer = BertWordPieceTokenizer.from_file("testdata/vocab.txt")
dataset = DatasetForSequenceClassification(tokenizer)
dataset.add_jsonl_files(input_files=["testdata/sequence_classification.jsonl"])

# convert to tf.data.Dataset
tf_dataset = dataset.to_tf_dataset(batch_size=32)
for idx, batch in enumerate(iter(tf_dataset)):
    print("No.{} batch: \n{}".format(idx, batch))

# save tfrecord
dataset.save_tfrecord("testdata/sequence_classification.tfrecord")
# build dataset from tfrecord files
dataset = TFDatasetForSequenceClassifiation.from_tfrecord_files("testdata/sequence_classification.tfrecord")
for idx, batch in enumerate(iter(dataset)):
    print("No.{} batch: \n{}".format(idx, batch))

# convert to torch.utils.data.Dataset
pt_dataset = dataset.to_pt_dataset()
dataloader = torch.utils.data.DataLoader(
    pt_dataset, num_workers=2, shuffle=True, batch_size=32, collate_fn=pt_dataset.batch_padding_collator
)
for idx, batch in enumerate(dataloader):
    print("No.{} batch: \n{}".format(idx, batch))
```

### question-answering-quickstart

```python
import torch
from tokenizers import BertWordPieceTokenizer
from rapidnlp_datasets import DatasetForQuestionAnswering
from rapidnlp_datasets.tf import TFDatasetForQuestionAnswering

# build dataset
tokenizer = BertWordPieceTokenizer.from_file("testdata/vocab.txt")
dataset = DatasetForQuestionAnswering(tokenizer)
dataset.add_jsonl_files(input_files="testdata/qa.jsonl")

# convert to tf.data.Dataset
tf_dataset = dataset.to_tf_dataset()
for idx, batch in enumerate(iter(tf_dataset)):
    print("NO.{} batch: \n{}".format(idx, batch))

# save to tfrecord
dataset.save_tfrecord("testdata/qa.tfrecord")

# build dataset from tfrecord files
tf_dataset = TFDatasetForQuestionAnswering.from_tfrecord_files(
    "testdata/qa.tfrecord", 
    batch_size=32, 
    padding="batch"
)
for idx, batch in enumerate(iter(tf_dataset)):
    print()
    print("No.{} batch: \n{}".format(idx, batch))

# convert to torch.utils.data.Dataset
pt_dataset = dataset.to_pt_dataset()
dataloader = torch.utils.data.DataLoader(
    pt_dataset,
    batch_size=32,
    collate_fn=pt_dataset.batch_padding_collator,
)
for idx, batch in enumerate(dataloader):
    print("No.{} batch: \n{}".format(idx, batch))
```


### token-classification-quickstart

```python
import torch
from tokenizers import BertWordPieceTokenizer
from rapidnlp_datasets import DatasetForTokenClassification
from rapidnlp_datasets.tf import TFDatasetForTokenClassification

# build dataset
tokenizer = BertCharLevelTokenizer.from_file("testdata/vocab.txt")
dataset = DatasetForTokenClassification(tokenizer)
dataset.add_jsonl_files("testdata/token_classification.jsonl", label2id=_label_to_id)

# conver to tf.data.Dataset
tf_dataset = dataset.to_tf_dataset()
for idx, batch in enumerate(iter(tf_dataset)):
    print("No.{} batch:\n{}".format(idx, batch))

# save dataset to tfrecord
dataset.save_tfrecord("testdata/token_classification.tfrecord")
# build dataset from tfrecord files
tf_dataset = TFDatasetForTokenClassification.from_tfrecord_files(
    input_files="testdata/token_classification.tfrecord",
    batch_size=4,
)
for idx, batch in enumerate(iter(tf_dataset)):
    print("No.{} batch:\n{}".format(idx, batch))

# convert to torch.utils.data.Dataset
pt_dataset = dataset.to_pt_dataset()
dataloader = torch.utils.data.DataLoader(
    pt_dataset, num_workers=1, batch_size=4, collate_fn=pt_dataset.batch_padding_collator
)
for idx, batch in enumerate(dataloader):
    print("No.{} batch:\n{}".format(idx, batch))
```

### masked-language-models-quickstart

```python
import torch
from tokenizers import BertWordPieceTokenizer
from rapidnlp_datasets import DatasetForMaskedLanguageModel
from rapidnlp_datasets.tf import TFDatasetForMaksedLanguageModel

# build dataset
tokenizer = BertWordPieceTokenizer.from_file("testdata/vocab.txt")
dataset = DatasetForMaskedLanguageModel(tokenizer)
dataset.add_jsonl_files(input_files=["testdata/mlm.jsonl"])
dataset.add_text_files(input_files=["/path/to/text/files"])

# convert to tf.data.Dataset
tf_dataset = dataset.to_tf_dataset(batch_size=4)
for idx, batch in enumerate(iter(tf_dataset)):
    print("No.{} batch:\n{}".format(idx, batch))

# save dataset as tfrecord
dataset.save_tfrecord("testdata/mlm.tfrecord")
# load tf.data.Dataset from tfrecord files
dataset = TFDatasetForMaksedLanguageModel.from_tfrecord_files(input_files="testdata/mlm.tfrecord", batch_size=4)
for idx, batch in enumerate(iter(dataset)):
    print("No.{} batch:\n{}".format(idx, batch))

# convert to torch.utils.data.Dataset
pt_dataset = dataset.to_pt_dataset()
# build dataloader
dataloader = torch.utils.data.DataLoader(
    pt_dataset, batch_size=4, num_workers=1, collate_fn=pt_dataset.batch_padding_collator
)
for idx, batch in enumerate(dataloader):
    print("No.{} batch:\n{}".format(idx, batch))

````

### simcse-quickstart

```python
import torch
from tokenizers import BertWordPieceTokenizer
from rapidnlp_datasets import DatasetForSimCSE
from rapidnlp_datasets.tf import TFDatasetForSimCSE

# build dataset
dataset = DatasetForSimCSE(
    tokenizer=BertWordPieceTokenizer.from_file("testdata/vocab.txt"),
    with_positive_sequence=False,
    with_negative_sequence=False,
)
dataset.add_jsonl_files("testdata/simcse.jsonl")

# convert to tf.data.Dataset
tf_dataset = dataset.to_tf_dataset()
for idx, batch in enumerate(iter(tf_dataset)):
    print()
    print("No.{} batch: \n{}".format(idx, batch))

# save to tfrecord
dataset.save_tfrecord("testdata/simcse.tfrecord")
# build dataset from tfrecord files
tf_dataset = TFDatasetForSimCSE.from_tfrecord_files(
    "testdata/simcse.tfrecord",
    with_positive_sequence=False,
    with_negative_sequence=False,
)
for idx, batch in enumerate(iter(tf_dataset)):
    print("No.{} batch: \n{}".format(idx, batch))

# convert to torch.utils.data.Dataset
pt_dataset = dataset.to_pt_dataset()
dataloader = torch.utils.data.DataLoader(
    pt_dataset, num_workers=2, shuffle=True, batch_size=32, collate_fn=pt_dataset.batch_padding_collator
)
for idx, batch in enumerate(dataloader):
    print("No.{} batch: \n{}".format(idx, batch))
```

