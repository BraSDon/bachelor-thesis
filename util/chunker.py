from abc import abstractmethod, ABCMeta


class Chunker(metaclass=ABCMeta):
    @abstractmethod
    def chunk(self, world_size, indices):
        pass


class SequentialChunker(Chunker):
    def chunk(self, world_size, indices):
        len_per_rank = len(indices) // world_size
        return [indices[i * len_per_rank: (i + 1) * len_per_rank] for i in range(world_size)]


class StepChunker(Chunker):
    def chunk(self, world_size, indices):
        return [indices[rank::world_size] for rank in range(world_size)]
