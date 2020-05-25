
from collections import namedtuple

from utils.collections import ExtendedEnum


class ElementType(ExtendedEnum):
    number = "NUM"
    attribute = "ATTR"
    verb = "VERB"
    container = "CONT"
    entity = "ENT"


class VerbType(ExtendedEnum):
    observation = "OBS"
    positive = "POS"
    negative = "NEG"
    positive_transfer = "POSTRN"
    negative_transfer = "NEGTRN"
    construct = "CONS"
    destroy = "DEST"


class ContainerType(ExtendedEnum):
    agent = "AGT"
    environment = "ENV"


AbsTokenInfo = namedtuple("AbsTokenInfo",
                          ["type", "type_idx", "subtype", "reverse_idx"])

DependentElementTypes = [
    ElementType.number.value, ElementType.attribute.value
]
IndependentElementTypes = [
    elem_type for elem_type in ElementType.values()
    if elem_type not in DependentElementTypes
]

