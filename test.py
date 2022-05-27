import torch as ch
from tensorguard.types import Tensor as T
from tensorguard.guard import tensorguard

Tensor = T

@tensorguard
def inference(x: T(['bs', 3, 224, 224], 'float16'), y: T(['bs'], 'int64')):
    pass

# make examples with wrong dtype
x = ch.ones(128, 3, 224, 224, dtype=ch.float32)
# make labels with wrong batch size
y = ch.ones(256)

inference(x, y)



def check_bad(f, t):
    if not isinstance(t, tuple):
        t = (t,)
    bad = True
    try:
        f(*t)
    except Exception as e:
        bad = False
        print('no error; msg\n', e)
    if bad:
        raise ValueError(f'no error for {f}, {t}')

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

print(Tensor(['bs', 1, 'sl', 'sl'], 'float32', 'cpu', None))
print(Tensor(['bs', 1, 'sl', 'sl'], 'float32', 'dev2', None))

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

@tensorguard
def f1(a: Tensor([1, 2, 3, 4], 'float32', 'cpu')):
    return 1

t1 = ch.randn(1, 2, 3, 4).to(dtype=ch.float32)
t2 = ch.randn(2, 2, 3, 4).to(dtype=ch.float32)
t3 = ch.randn(1, 2, 3, 4).to(dtype=ch.float64)
f1(t1)

check_bad(f1, t2)
check_bad(f1, t3)

@tensorguard
def f2(a: Tensor([None, 2, 3, 4], 'float32', 'cpu')):
    return 1

f2(t1)
f2(t2)

@tensorguard
def f2(a: Tensor(['a', 'a', 3, 4], 'float32', 'cpu'), b: Tensor(['a', 'a', 1, 2])) -> Tensor([]):
    return b[0, 0, 0, 0]

t1 = ch.randn(4, 4, 3, 4).to(dtype=ch.float32)
t2 = ch.randn(4, 4, 1, 2).to(dtype=ch.float32)
t3 = ch.randn(5, 4, 3, 4).to(dtype=ch.float16)


@tensorguard
def inference(x: T(['bs', 3, 224, 224], 'float16'), y: T(['bs', 'int64'])):
    pass

# make examples with wrong dtype
x = ch.ones(128, 3, 224, 224, dtype=ch.float32)
# make labels with wrong batch size
y = ch.ones(256)

inference(x, y)

#inference()

# this fails because we cant detect scalars??
# f2(t1, t2)
# this fails because of bad generics detection
# check_bad(f2, (t3, t2))

#f2(t3, t2)

# generics
# printing
# print(Tensor([10, 4, 3], 'float32', 'cpu', None))

# @typechecked
# def a(b:int, c:str):
#     return b

# a(1, 'str')
# a(1, 2)
