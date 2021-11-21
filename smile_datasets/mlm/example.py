from collections import namedtuple

ExampleForMaskedLanguageModel = namedtuple(
    "ExampleForMaskedLanguageModel",
    ["tokens", "input_ids", "segment_ids", "attention_mask", "masked_ids", "masked_pos"],
)
