import abc


class AbstractExampleParser(abc.ABC):
    """Abstract example parser"""

    @abc.abstractmethod
    def parse_instance(self, instance, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def parse_instances(self, instances, **kwargs):
        raise NotImplementedError()
