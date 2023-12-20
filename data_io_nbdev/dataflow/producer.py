# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/06.dataflow.producer.ipynb.

# %% auto 0
__all__ = ['T_RAW', 'T_HMI', 'Producer']

# %% ../../nbs/06.dataflow.producer.ipynb 3
import abc
from threading import Event
from dataclasses import dataclass
from typing import Optional, TypeVar, Generic

# %% ../../nbs/06.dataflow.producer.ipynb 4
from .pipeline.queue import Pipeline  # type: ignore
from .pipeline.deque import PipelineDQ  # type: ignore

# %% ../../nbs/06.dataflow.producer.ipynb 5
T_RAW = TypeVar("T_RAW")  # Generic type for raw data
T_HMI = TypeVar("T_HMI")  # Generic type for HMI data

# %% ../../nbs/06.dataflow.producer.ipynb 6
@dataclass
class Producer(
    abc.ABC, Generic[T_RAW, T_HMI]
):  # Pycharm false positive warning for non-unique generic type
    """Producer produce data into the pipeline. It provides the unified interface for data capturing interface produce()"""

    def __post_init__(self):
        super().__init__()

    @abc.abstractmethod
    def produce(
        self,
        raw_pipeline: PipelineDQ[
            T_RAW
        ],  # Raw pipeline is deque to keep data fresh and ignore stale data, such as one with dict[str,str]
        hmi_pipeline: Optional[
            Pipeline[T_HMI]
        ] = None,  # HMI pipeline is Queue, such as one with str
        exit_event: Optional[Event] = None,
    ):
        """
        Produce data into the pipeline
        """
        pass