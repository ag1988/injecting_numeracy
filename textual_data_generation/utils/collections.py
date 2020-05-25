
from enum import Enum
from itertools import chain, combinations


class ExtendedEnum(Enum):
    @classmethod
    def names(cls):
        return list(map(lambda c: c.name, cls))

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))


def get_list_subsets(lst, max_subset_size=2):
    return list(chain.from_iterable(combinations(lst, r) for r in range(max_subset_size+1)))

