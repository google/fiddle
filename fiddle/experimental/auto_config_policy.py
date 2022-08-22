# coding=utf-8
# Copyright 2022 The Fiddle-Config Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyformat: mode=midnight
"""A policy of functions to skip representing as `fdl.Config`'s.

When running `auto_config` on a function, some Python functions cannot or, in
practice, should never be mapped into `fdl.Config`-space. This file defines
policies (mostly implemented as lists) that enumerates them.

Because either adding or removing a function from this list can cause config
incompatibilities in end-user config, this file is structured around policy
versions.
"""

import copy
import inspect
import itertools
import logging
import types
import typing

from fiddle import building
from fiddle import config
from fiddle.experimental import autobuilders

Policy = typing.Callable[[typing.Type[typing.Any]], bool]

BUILTIN_LIST = [
    ArithmeticError,
    AssertionError,
    AttributeError,
    BaseException,
    BlockingIOError,
    BrokenPipeError,
    BufferError,
    BytesWarning,
    ChildProcessError,
    ConnectionAbortedError,
    ConnectionError,
    ConnectionRefusedError,
    ConnectionResetError,
    DeprecationWarning,
    EOFError,
    Ellipsis,
    EnvironmentError,
    Exception,
    # False,  # Skip False, as it's not callable.
    FileExistsError,
    FileNotFoundError,
    FloatingPointError,
    FutureWarning,
    GeneratorExit,
    IOError,
    ImportError,
    ImportWarning,
    IndentationError,
    IndexError,
    InterruptedError,
    IsADirectoryError,
    KeyError,
    KeyboardInterrupt,
    LookupError,
    MemoryError,
    ModuleNotFoundError,
    NameError,
    # None,  # Skip None, as it's not callable.
    NotADirectoryError,
    NotImplemented,
    NotImplementedError,
    OSError,
    OverflowError,
    PendingDeprecationWarning,
    PermissionError,
    ProcessLookupError,
    RecursionError,
    ReferenceError,
    ResourceWarning,
    RuntimeError,
    RuntimeWarning,
    StopAsyncIteration,
    StopIteration,
    SyntaxError,
    SyntaxWarning,
    SystemError,
    SystemExit,
    TabError,
    TimeoutError,
    # True,  # Skip True, as it's not callable.
    TypeError,
    UnboundLocalError,
    UnicodeDecodeError,
    UnicodeEncodeError,
    UnicodeError,
    UnicodeTranslateError,
    UnicodeWarning,
    UserWarning,
    ValueError,
    Warning,
    ZeroDivisionError,
    abs,
    all,
    any,
    ascii,
    bin,
    bool,
    breakpoint,
    bytearray,
    bytes,
    callable,
    chr,
    classmethod,
    compile,
    complex,
    delattr,
    dict,
    dir,
    divmod,
    enumerate,
    eval,
    exec,
    filter,
    float,
    format,
    frozenset,
    getattr,
    globals,
    hasattr,
    hash,
    hex,
    id,
    input,
    int,
    isinstance,
    issubclass,
    iter,
    len,
    list,
    locals,
    map,
    max,
    memoryview,
    min,
    next,
    object,
    oct,
    open,
    ord,
    pow,
    print,
    property,
    range,
    repr,
    reversed,
    round,
    set,
    setattr,
    slice,
    sorted,
    staticmethod,
    str,
    sum,
    super,
    tuple,
    type,
    vars,
    zip,
]

V1_SKIPLIST = [
    autobuilders.Registry.config,
    building.build,
    config.Config,
    config.Partial,
    copy.copy,
    copy.deepcopy,
    inspect.signature,
    itertools.accumulate,
    itertools.chain,
    itertools.chain.from_iterable,
    itertools.combinations,
    itertools.combinations_with_replacement,
    itertools.compress,
    itertools.count,
    itertools.cycle,
    itertools.dropwhile,
    itertools.groupby,
    itertools.islice,
    # itertools.pairwise,  # Py3.10
    itertools.permutations,
    itertools.product,
    itertools.repeat,
    itertools.starmap,
    itertools.takewhile,
    itertools.tee,
    itertools.zip_longest,
    logging.critical,
    logging.debug,
    logging.error,
    logging.exception,
    logging.fatal,
    logging.info,
    logging.log,
    logging.warn,
    logging.warning,
    logging.Logger.critical,
    logging.Logger.debug,
    logging.Logger.error,
    logging.Logger.exception,
    logging.Logger.fatal,
    logging.Logger.info,
    logging.Logger.warn,
    logging.Logger.warning,
    typing.cast,
]

# TODO: Add Google-specific entries here (e.g. absl logging?)


def _is_builtin_method(fn_or_cls) -> bool:
  """Returns True if `fn_or_cls` is a builtin method.

  Because some builtin method types don't provide access to the underlying
  function (making it impossible to explicitly enumerate them), we need a policy
  to handle them specially.

  Args:
    fn_or_cls: The object to determine whether if it's a builtin method.

  Returns:
    The answer.
  """
  return isinstance(fn_or_cls, types.BuiltinFunctionType) or isinstance(
      fn_or_cls, types.MethodWrapperType
  )


_v1_policy_set = frozenset(
    [
        *V1_SKIPLIST,
        *BUILTIN_LIST,
    ]
)


def v1(fn_or_cls) -> bool:
  """Returns `True` if `fn_or_cls` should not be represented as a fdl.Config."""
  if inspect.ismethod(fn_or_cls):
    fn_or_cls = fn_or_cls.__func__
  return fn_or_cls in _v1_policy_set or _is_builtin_method(fn_or_cls)


latest = v1
