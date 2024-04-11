from typing import List, Union
from collections.abc import Iterable
from .._grammar import byte_range, ByteRange

def char_range(low, high):
    low_bytes = bytes(low, encoding="utf8")
    high_bytes = bytes(high, encoding="utf8")
    if len(low_bytes) > 1 or len(high_bytes) > 1:
        raise Exception("We don't yet support multi-byte character ranges!")
    return byte_range(low_bytes, high_bytes)


def nice_char_group(chars: Iterable[str]) -> List[Union[bytes, ByteRange]]:
    """
    Condenses a list of characters to a list of "nice" guidance byte ranges,
    e.g. ['A','B','C','D','1','2','3'] -> [b'AB', b'13']

    Code adapted from interegular.fsm.nice_char_group
    """
    bts = [bytes(char, encoding="utf8") for char in chars]
    out: List[Union[bytes, ByteRange]] = []
    current_range: List[bytes] = []
    for b in sorted(bts):
        if len(b) > 1:
            raise NotImplementedError("We don't yet support multi-byte character ranges!")
        if current_range and ord(current_range[-1]) == ord(b):
            continue
        if current_range and ord(current_range[-1]) + 1 == ord(b):
            current_range.append(b)
            continue
        if len(current_range) >= 2:
            out.append(byte_range(current_range[0], current_range[-1]))
        else:
            out.extend(current_range)
        current_range = [b]
    if len(current_range) >= 2:
        out.append(byte_range(current_range[0], current_range[-1]))
    else:
        out.extend(current_range)
    return out
