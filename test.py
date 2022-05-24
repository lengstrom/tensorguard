from tensorguard.types import TensorShape, Tensor
from typeguard import typechecked

# tests:
a = Tensor([10, 4, 3], 'float32', 'cpu', 'torch')
b = Tensor([5, 4, 3], 'float32', 'cpu', 'torch')
c = Tensor([10, 4, 3], 'uint8', 'cpu', 'torch')
d = Tensor([10, 4, 3], 'float32', 'cuda', 'torch')
e = Tensor([10, 4, 3], 'float32', 'cpu', 'numpy')

f = Tensor(None, 'float32', 'cpu', 'torch')
g = Tensor([10, 4, 3], None, 'cpu', 'torch')
h = Tensor([10, 4, 3], 'float32', None, 'torch')
i = Tensor([10, 4, 3], 'float32', 'cpu', None)

def test_two(a, b, expected, name):
    print(name)
    expected = set(expected)
    diff = a.diff(b)
    assert diff == expected, (diff, expected)
    print(a.rep_diff(b, set()))
    print('*' * 40)
    print(b.rep_diff(a, set()))
    print('-' * 80)

# diff shape
# diff device
# diff dtype
# diff library
test_two(a, b, ['shape'], 'shape diff')
test_two(a, c, ['dtype'], 'dtype diff')
test_two(a, d, ['device'], 'device diff')
test_two(a, e, ['library'], 'library diff')

test_two(a, f, [], 'generic diff')
test_two(a, g, [], 'generic diff')
test_two(a, h, [], 'generic diff')
test_two(a, i, [], 'generic diff')
# generics
# printing
print(Tensor([10, 4, 'bs'], 'float32', 'cpu', None))