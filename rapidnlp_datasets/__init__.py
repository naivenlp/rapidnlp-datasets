import logging

from .charlevel_tokenizer import BertCharLevelTokenizer, CharLevelEncoding
from .masked_lm import (
    AbstractMaskingStrategy,
    ExampleForMaskedLanguageModel,
    ExampleParserForMaskedLanguageModel,
    WholeWordMask,
)
from .parsers import AbstractExampleParser
from .question_answering import ExampleForQuestionAnswering, ExampleParserForQuestionAnswering
from .readers import AbstractFileReader, CsvFileReader, JsonlFileReader
from .sequence_classification import ExampleForSequenceClassification, ExampleParserForSequenceClassification
from .simcse import ExampleForSimCSE, ExampleParserForSimCSE
from .token_classification import ExampleForTokenClassification, ExampleParserForTokenClassification

__name__ = "rapidnlp_datasets"
__version__ = "0.1.0"

logging.basicConfig(format="%(asctime)s %(levelname)7s %(filename)20s %(lineno)4d] %(message)s", level=logging.INFO)
