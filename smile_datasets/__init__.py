import logging

from smile_datasets.dataset import Datapipe
from smile_datasets.qa.dataset import DatapipeForQuestionAnswering, DatasetForQuestionAnswering
from smile_datasets.qa.example import ExampleForQuestionAnswering
from smile_datasets.qa.parsers import ParserForQuestionAnswering
from smile_datasets.sequence_classification.dataset import DatapipeForSequenceClassifiation, DatasetForSequenceClassification
from smile_datasets.sequence_classification.example import ExampleForSequenceClassification
from smile_datasets.sequence_classification.parsers import ParserForSequenceClassification

__name__ = "smile_datasets"
__version__ = "0.0.2"

logging.basicConfig(format="%(asctime)s %(levelname)7s %(filename)20s %(lineno)4d] %(message)s", level=logging.INFO)
