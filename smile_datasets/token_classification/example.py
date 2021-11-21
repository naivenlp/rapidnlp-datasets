from collections import namedtuple

ExampleForTokenClassification = namedtuple(
    "ExampleForTokenClassification", ["tokens", "labels", "input_ids", "segment_ids", "attention_mask", "label_ids"]
)
