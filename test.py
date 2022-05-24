from typing import TypeVar, List
# from typeguard import typechecked
from pytypes import typechecked

T = TypeVar('T')

@typechecked
def meth(a: List[T], b: T) -> T:
    return a

if __name__ == '__main__':
    # meth('a', 'b')
    # meth(['a'], 'a')
    # meth(['a', 1], 'a')
    print([T, 1, 23, 4])