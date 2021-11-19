from collections import namedtuple

ExampleForSequenceClassification = namedtuple(
    "ExampleForSequenceClassification", ["tokens", "input_ids", "segment_ids", "attention_mask", "label"]
)
