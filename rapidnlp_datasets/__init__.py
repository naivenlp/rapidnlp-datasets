import logging

from .charlevel_tokenizer import BertCharLevelTokenizer, CharLevelEncoding
from .masked_lm import (
    AbcMaskingForLanguageModel,
    DatasetForMaskedLanguageModel,
    ExampleForMaskedLanguageModel,
    WholeWordMaskingForLanguageModel,
)
from .sequence_classification import DatasetForSequenceClassification, ExampleForSequenceClassification
from .simcse import DatasetForSimCSE, ExampleForSimCSE
from .token_classification import DatasetForTokenClassification, ExampleForTokenClassification

__name__ = "rapidnlp_datasets"
__version__ = "0.1.0"

logging.basicConfig(format="%(asctime)s %(levelname)7s %(filename)20s %(lineno)4d] %(message)s", level=logging.INFO)
