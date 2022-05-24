from multiprocessing.sharedctypes import Value
from typing import TypeVar
import torch as ch
import numpy as np
from termcolor import colored
from functools import partial

DTYPES = [
    ('float16', ch.float16, np.float16),
    ('float32', ch.float32, np.float32),
    ('float64', ch.float64, np.float64),
    ('uint8', ch.uint8, np.uint8),
    ('int8', ch.int8, np.int8),
    ('int16', ch.int16, np.int16),
    ('int32', ch.int32, np.int32),
    ('int64', ch.int64, np.int64, int)
]

NAMES = {
    'long':'int64',
    'int':'int64',
    'half':'float16'
}

HUMAN_TYPES, CH_TYPES, NP_TYPES = zip(*DTYPES)

highlight_text = partial(colored, on_color='on_red', attrs=['underline', 'bold'])

def field_ok(a, b, bad_set=set()):
    eq = a == b 
    generic_eq = (type(a) == TypeVar) ^ (type(b) == TypeVar)
    none_eq = a is None or b is None

    either_bad = (a in bad_set or b in bad_set)
    return (eq or generic_eq or none_eq) and not either_bad

class TensorTypeBase:
    def __init__(self, value):
        raise NotImplementedError()

    def __repr__(self):
        raise NotImplementedError()

    def type_matches(self, a):
        raise NotImplementedError()

    def rep_diff(self, a, bad_typevars: set):
        raise NotImplementedError()

class TensorTypeScalar(TensorTypeBase):
    def __init__(self, value):
        self.value = value

    def type_matches(self, a):
        return field_ok(a.value, self.value)

    def rep_diff(self, a, bad_typevars):
        rep = self.__repr__()
        if not self.type_matches(a) or a.value in bad_typevars:
            return highlight_text(rep)

        return rep

class TensorShape(TensorTypeBase):
    def __init__(self, shape):
        assert shape is not None
        _acceptable_types = [int, TypeVar]
        shape = [(int(k) if type(k) is str else k) for k in shape]
        for k in shape:
            msg = f'Dimension {k} ({type(k)}) should be a int, str, or TypeVar'
            assert type(k) in _acceptable_types, msg

        self.shape = shape

    def __repr__(self):
        return self.shape.__repr__()

    def type_matches(self, a):
        if len(a.shape) != len(self.shape):
            return False

        return all(field_ok(a, b) for a, b in zip(a.shape, self.shape))

    def rep_diff(self, a, bad_typevars: set):
        # get rep for diff between this and a, given which typevars are bad
        # if totally diff just highlight the whole thing
        if len(a.shape) != len(self.shape):
            return False, highlight_text(self.__repr__())

        rep = []
        for v1, v2 in zip(self.shape, a.shape):
            this_rep = str(v1)
            # if they match, or if one of them is a typevar
            is_ok = field_ok(v1, v2, bad_typevars)
            this_rep = highlight_text(this_rep) if not is_ok else this_rep
            rep.append(this_rep)

        return '[' + ', '.join(rep) + ']'

class Device(TensorTypeScalar):
    def __init__(self, device):
        msg = f'{device} is not a supported type!'
        assert device in ['cuda', 'cpu'], msg
        super().__init__(device)

    def __repr__(self):
        return str(self.value)

class Library(TensorTypeScalar):
    def __init__(self, library):
        msg = f'{library} is not a supported tensor library!'
        assert library in ['numpy', 'torch'], msg
        super().__init__(library)

    def __repr__(self):
        v = self.value
        v = v[0:1].upper() + v[1:]
        return v

class DType(TensorTypeScalar):
    def __init__(self, dtype):
        msg = f'{dtype} not a supported type!'
        assert dtype in CH_TYPES or type(dtype) == TypeVar, msg

        super().__init__(dtype)

    def __repr__(self):
        return str(self.value).replace('torch.', '')

    @classmethod
    def make(cls, name):
        if name in NAMES:
            name = NAMES[name]
        
        for str_name, ch_name, np_name in DTYPES:
            if name == str_name or name == ch_name or name == np_name:
                return cls(dtype=ch_name)

        return cls(dtype=ch_name)

class Tensor:
    def __init__(self, shape=None, dtype=None, device=None, library='torch'):
        self.shape = TensorShape(shape) if shape is not None else shape
        self.dtype = DType.make(dtype) if dtype is not None else dtype
        self.device = Device(device) if device is not None else device
        self.library = Library(library) if library is not None else library

        names = ['shape', 'dtype', 'device', 'library']
        ls = [getattr(self, n) for n in names]

        self.props = {k:v for k, v in zip(names, ls)}

    @classmethod
    def from_tensor(cls, v):
        raise NotImplementedError()

    def diff(self, a):
        # calculates type differences between this tensortype and another
        # tensortype; returns keys of differences
        diffs = []
        other_keys = set(a.props.keys())

        for k in self.props.keys():
            other_prop = a.props[k]
            if other_prop is not None:
                this_prop = self.props[k]
                if not other_prop.type_matches(this_prop):
                    # print(other_prop, this_prop)
                    # if k != 'shape':
                    #     import pdb; pdb.set_trace()
                    diffs.append(k)

        # return all the fields that theres a difference in
        return set(diffs)

    def rep_diff(self, a, bad_typevars):
        # make a string rep for the diff
        d = {}
        for k, v in self.props.items():
            if v is not None:
                other_v = a.props[k]
                if other_v is not None:
                    d[k] = v.rep_diff(other_v, bad_typevars)
                else:
                    d[k] = str(v)

        rep = Tensor.rep_func(d)
        return rep
 
    def __repr__(self):
        rep = {
            'shape':self.shape,
            'dtype':self.dtype,
            'device':self.device,
            'library':self.library
        }

        d = {k:str(v) for k, v in rep.items()}
        return Tensor.rep_func(d)

    @staticmethod
    def rep_func(rep):
        # highlight: can include shape, dtype, device, library
        names = ['shape', 'dtype', 'device']
        ls = [str(rep[v]) for v in names if v in rep]
        filt = [v for v in ls if v is not None]
        spec = ', '.join(filt)
        library = 'Tensor' if not 'library' in rep else rep['library']
        return f'{library}({spec})'

# two kinds of comparisons:
#  - vertical: does this match the original signature?
#  - horizontal: do the 

# def add_two_tensors(a: Tensor(['bs', 3, 224, 224], 'float32'), b: Tensor(['bs', 3, 224, 224], 'float32'))
# Tensor([shape], 'uint8', 'cuda:0', 'numpy')