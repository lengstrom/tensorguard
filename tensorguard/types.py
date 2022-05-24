from typing import TypeVar
import torch as ch

class TensorShape:
    def __init__(self, shape):
        _acceptable_types = [int, TypeVar]
        shape = [(int(k) if type(k) is str else k) for k in shape]
        for k in shape:
            msg = f'Dimension {k} ({type(k)}) should be a int, str, or TypeVar'
            assert type(k) in _acceptable_types, msg

        self.shape = shape

    def __repr__(self):
        return self.shape.__repr__()


DTYPES = [
    ('float8', ch.float8, np.float8),
    ('float16', ch.float16, np.float16),
    ('float32', ch.float32, np.float32),
    ('float64', ch.float64, np.float64),
    ('uint8', ch.uint8, np.uint8),
    ('uint16', ch.uint16, np.uint16),
    ('uint32', ch.uint32, np.uint32),
    ('uint64', ch.uint64, np.uint64),
    ('int8', ch.int8, np.int8),
    ('int16', ch.int16, np.int16),
    ('int32', ch.int32, np.int32),
    ('int64', ch.int64, np.int64, int)
]

NAMES = {
    'long':'int64',
    'int':'int64',
    'half':'float16',
    'float':'float32'
}

NAMES, CH_TYPES, NP_TYPES = zip(*DTYPES)

class Dtype:
    def __init__(self, dtype):
        assert dtype in CH_TYPES
        self.dtype = dtype

    def __repr__():
        return 

    @classmethod
    def make()


class Tensor:
    def __init__(self, shape=None, dtype=None, device=None, library=None):
        self.shape = TensorShape(shape) if shape is not None else shape
        self.dtype = dtype

        assert self.device in ['cuda', 'cpu', None]
        self.device = device

        assert library in [None, 'numpy', 'torch']
        self.library = library

    def __repr__(self):
        ls = [self.shape, self.dtype, self.device]
        filt = [v for v in ls if v is not None]
        spec = ', '.join(filt)

        tensor_class = {
            'torch':'Torch.Tensor',
            'numpy':'numpy.ndarray'
            None:'Tensor'
        }[self.library]

        return f'{tensor_class}({spec})'

# def add_two_tensors(a: Tensor(['bs', 3, 224, 224], 'float32'), b: Tensor(['bs', 3, 224, 224], 'float32'))
# Tensor([shape], 'uint8', 'cuda:0', 'numpy')