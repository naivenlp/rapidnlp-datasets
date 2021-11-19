import logging

from smile_datasets.qa.dataset import DatapipeForQuestionAnswering, DatasetForQuestionAnswering
from smile_datasets.qa.example import ExampleForQuestionAnswering
from smile_datasets.qa.parsers import ParserForQuestionAnswering

__name__ = "smile_datasets"
__version__ = "0.0.1"

logging.basicConfig(format="%(asctime)s %(levelname)7s %(filename)20s %(lineno)4d] %(message)s", level=logging.INFO)
