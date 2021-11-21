from collections import namedtuple

ExampleForUnsupervisedSimCSE = namedtuple(
    "ExampleForUnsupervisedSimCSE",
    [
        "tokens",
        "input_ids",
        "segment_ids",
        "attention_mask",
    ],
)
ExampleForSupervisedSimCSE = namedtuple(
    "ExampleForSupervisedSimCSE",
    [
        "tokens",
        "input_ids",
        "segment_ids",
        "attention_mask",
        "pos_tokens",
        "pos_input_ids",
        "pos_segment_ids",
        "pos_attention_mask",
    ],
)

ExampleForHardNegativeSimCSE = namedtuple(
    "ExampleForHardNegativeSimCSE",
    [
        "tokens",
        "input_ids",
        "segment_ids",
        "attention_mask",
        "pos_tokens",
        "pos_input_ids",
        "pos_segment_ids",
        "pos_attention_mask",
        "neg_tokens",
        "neg_input_ids",
        "neg_segment_ids",
        "neg_attention_mask",
    ],
)
