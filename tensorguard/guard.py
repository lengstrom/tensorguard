from functools import wraps, partial
from termcolor import colored
from collections import defaultdict
from tensorguard.types import Tensor
from typeguard import _CallMemo

class TensorMismatchError(Exception):
    pass

from .types import _BAD_GENERIC, Tensor

bolder = partial(colored, attrs=['bold'])
underliner = partial(colored, attrs=['underline'])

def error_msg(argnames, generics, hints, realized, conversion_errors,
              ret_hint=None, ret_realized=None):
    ret_issue = ret_hint is not None
    args_issue = not ret_issue or not _generics_ok(generics)

    msg = []
    args_emsg = args_error_msg(argnames, generics, hints, realized, conversion_errors)
    msg.append(bolder('\n'))
    msg.append(args_emsg)
    if not args_issue:
        msg.append(bolder('\n\n'))
        msg.append(return_error_msg(generics, conversion_errors, ret_hint, ret_realized))

    msg = ''.join(msg)
    return msg

from .types import highlight_text
def maybe_message(this_type, other_type, bad_generics):
    if isinstance(this_type, Tensor):
        return this_type.rep_diff(other_type, bad_generics)
    else:
        return highlight_text(str(this_type))

def _error_msg(argnames, generics, hints, realized, conversion_errors, name):
    value_messages = []
    hint_messages = []
    bad_generics = _bad_generics(generics)
    for expected, actual, tname in zip(hints, realized, argnames):
        hint_message = maybe_message(expected, actual, bad_generics)
        value_message = maybe_message(actual, expected, bad_generics)
        value_messages.append(f'{underliner(tname)}: {value_message}')
        hint_messages.append(f'{underliner(tname)}: {hint_message}')

    expected_line = ', '.join(hint_messages)
    actual_line = ', '.join(value_messages)
    errors = [f'- {k} ({v}): \'{ems}\'' for k, (v, ems) in conversion_errors.items()]
    conv = bolder('Type inference errors:\n') + '\n'.join(errors) if errors else ''
    retval = f'{bolder(f"Expected {name}")}: {expected_line}\n{bolder(f"Realized {name}")}: {actual_line}{conv}'
    return retval

def args_error_msg(argnames, generics, hints, realized, conversion_errors):
    n = 'args'
    return _error_msg(argnames, generics, hints, realized, conversion_errors, n)

def return_error_msg(generics, conversion_errors, ret_hint, ret_realized):
    return _error_msg(['return'], generics, [ret_hint], [ret_realized],
                      conversion_errors, 'return')

# format:
# - most errors: shown during expected vs realized comparison
# - some errors (i.e. wrong type?): shown in list form at end

def tensorguard(func):
    def wrapper(*args, **kwargs):
        memo = _CallMemo(func=func, args=args, kwargs=kwargs)
        args_ok, processed = check_argument_types_and_generics(memo)
        argnames, hints, realized, conversion_errors, generics = processed
        error_args = (argnames, generics, hints, realized, conversion_errors)
        if not args_ok:
            msg = error_msg(*error_args)
            raise TensorMismatchError(msg)

        retval = func(*args, **kwargs)
        ret_ok, (ret_hint, ret_realized) = check_return_type(retval, memo,
                                                           conversion_errors,
                                                           generics)
        if not ret_ok:
            error_args = error_args + (ret_hint, ret_realized,)
            msg = error_msg(*error_args)
            raise TensorMismatchError(msg)

        return retval

    return wraps(func)(wrapper)

def _is_bad_generic(s):
    if _BAD_GENERIC in s or len(s) != 1:
        return True

    return False

def add_generics(expected_type, value_type, generics):
    expected = expected_type.props.values() 
    realized = value_type.props.values() 

    for ve, vr in zip(expected, realized):
        if ve is not None:
            vr.add_generics(ve, generics)

def _bad_generics(generics):
    return {k for k, s in generics.items() if _is_bad_generic(s)}

def _generics_ok(generics):
    bad_generics = _bad_generics(generics)
    return len(bad_generics) == 0

def check_types(expected_type, value_type):
    expected = expected_type.props.values() 
    realized = value_type.props.values() 

    is_ok = True
    for ve, vr in zip(expected, realized):
        if ve is not None:
            is_ok = is_ok and vr.type_matches(ve)

    return is_ok

def _process_tensor(value, argname, expected_type, conversion_errors, generics):
    success = False
    try:
        value_type = Tensor.from_tensor(value)
        success = True
    except ValueError as e:
        # add error record to conversion_errors
        conversion_errors[argname] = (e, value)
        value_type = type(value)

    if success:
        add_generics(expected_type, value_type, generics)
        is_ok = check_types(expected_type, value_type)
    else:
        is_ok = False
    
    return is_ok, value_type

def check_argument_types_and_generics(memo):
    # first go through types and...
    # - make types from tensors
    # - check generics
    generics = defaultdict(set)
    hints = []
    realizeds = []
    argnames = []
    conversion_errors = {}
    
    items = memo.type_hints.items()

    is_ok = True
    for argname, expected_type in items:
        if argname != "return" and argname in memo.arguments:
            value = memo.arguments[argname]
            hints.append(expected_type)
            argnames.append(argname)
            # only check the types that are Tensor types
            if isinstance(expected_type, Tensor):
                this_is_ok, value_type = _process_tensor(value,
                                                         argname,
                                                         expected_type,
                                                         conversion_errors,
                                                         generics)
                is_ok = is_ok and this_is_ok

            else:
                value_type = type(value)

            realizeds.append(value_type)

    # now go through again and...
    # - check nongeneric types
    is_ok = is_ok and _generics_ok(generics)

    # if everything typechecks we're good
    processed = (argnames, hints, realizeds, conversion_errors, generics)
    if is_ok:
        return True, processed
    else:
        return False, processed

def check_return_type(retval, memo, conversion_errors, generics) -> bool:
    is_ok = True
    if "return" in memo.type_hints:
        hint = memo.type_hints["return"]
        if isinstance(hint, Tensor):
            is_ok, value_type = _process_tensor(retval, 'return', hint,
                                                conversion_errors, generics)
        else:
            value_type = type(retval)
    else:
        hint = None
        value_type = None

    processed = (hint, value_type)
    return is_ok, processed
