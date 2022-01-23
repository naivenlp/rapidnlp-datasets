import logging

from .charlevel_tokenizer import BertCharLevelTokenizer, CharLevelEncoding
from .masked_lm import (
    AbcMaskingForLanguageModel,
    DatasetForMaskedLanguageModel,
    ExampleForMaskedLanguageModel,
    WholeWordMaskingForLanguageModel,
)
from .parsers import AbstractExampleParser
from .readers import AbstractFileReader, CsvFileReader, JsonlFileReader
from .sequence_classification import DatasetForSequenceClassification, ExampleForSequenceClassification
from .simcse import ExampleForSimCSE, ExampleParserForSimCSE
from .token_classification import ExampleForTokenClassification, ExampleParserForTokenClassification

__name__ = "rapidnlp_datasets"
__version__ = "0.1.0"

logging.basicConfig(format="%(asctime)s %(levelname)7s %(filename)20s %(lineno)4d] %(message)s", level=logging.INFO)
