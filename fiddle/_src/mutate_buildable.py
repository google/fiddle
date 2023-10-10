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

"""Library for mutating Buildable instances."""

from typing import Callable, Type, TypeVar, Union

from fiddle._src import config as config_lib
from fiddle._src import signatures


T = TypeVar('T')
TypeOrCallableProducingT = Union[Callable[..., T], Type[T]]


_buildable_internals_keys = (
    '__fn_or_cls__',
    '__arguments__',
    '__argument_tags__',
    '__argument_history__',
    '__signature_info__',
)


def move_buildable_internals(
    *, source: config_lib.Buildable, destination: config_lib.Buildable
):
  """Changes the internals of `destination` to be equivalent to `source`.

  Currently, this results in aliasing behavior, but this should not be relied
  upon. All the internals of `source` are aliased into `destination`.

  Args:
    source: The source configuration to pull internals from.
    destination: One of the buildables to swap.
  """
  if type(source) is not type(destination):
    # TODO(saeta): Relax this constraint such that both types merely need to be
    # buildable's.
    raise TypeError(f'types must match exactly: {type(source)} vs '
                    f'{type(destination)}.')

  # Update in-place using object.__setattr__ to bypass argument checking.
  for attr in _buildable_internals_keys:
    object.__setattr__(destination, attr, getattr(source, attr))


def update_callable(
    buildable: config_lib.Buildable,
    new_callable: TypeOrCallableProducingT,
    drop_invalid_args: bool = False,
) -> None:
  """Updates ``config`` to build ``new_callable`` instead.

  When extending a base configuration, it can often be useful to swap one class
  for another. For example, an experiment may want to swap in a subclass that
  has augmented functionality.

  ``update_callable`` updates ``config`` in-place (preserving argument history).

  Args:
    buildable: A ``Buildable`` (e.g. a ``fdl.Config``) to mutate.
    new_callable: The new callable ``config`` should call when built.
    drop_invalid_args: If True, arguments that don't exist in the new callable
      will be removed from buildable. If False, raise an exception for such
      arguments.

  Raises:
    TypeError: if ``new_callable`` has varargs, or if there are arguments set on
      ``config`` that are invalid to pass to ``new_callable``.
  """
  for key in buildable.__arguments__.keys():
    if isinstance(key, int):
      raise NotImplementedError(
          'Update callable for configs with positional arguments is not'
          ' supported yet.'
      )
  # Note: can't just call config.__init__(new_callable, **config.__arguments__)
  # to preserve history.
  #
  # Note: can't call `setattr` on all the args to validate them, because that
  # will result in duplicate history entries.
  new_signature = signatures.get_signature(new_callable)
  # Update the signature early so that we can set arguments by position.
  # Otherwise, parameter validation logic would complain about argument
  # name not exists.
  new_signature_info = signatures.SignatureInfo(signature=new_signature)
  object.__setattr__(buildable, '__signature__', new_signature)
  object.__setattr__(buildable, '__signature_info__', new_signature_info)
  if not new_signature_info.has_var_keyword:
    invalid_args = [
        arg
        for arg in buildable.__arguments__.keys()
        if arg not in new_signature.parameters and isinstance(arg, str)
    ]
    if invalid_args:
      if drop_invalid_args:
        for arg in invalid_args:
          delattr(buildable, arg)
      else:
        raise TypeError(
            f'Cannot switch to {new_callable} (from '
            f'{buildable.__fn_or_cls__}) because the Buildable would '
            f'have invalid arguments {invalid_args}.'
        )
  object.__setattr__(buildable, '__fn_or_cls__', new_callable)
  buildable.__argument_history__.add_new_value('__fn_or_cls__', new_callable)


def assign(buildable: config_lib.Buildable, **kwargs):
  """Assigns multiple arguments to ``buildable``.

  Although this function does not enable a caller to do something they can't
  already do with other syntax, this helper function can be useful when
  manipulating deeply nested configs. Example::

    cfg = # ...
    fdl.assign(cfg.my.deeply.nested.child.object, arg_a=1, arg_b='b')

  The above code snippet is equivalent to::

    cfg = # ...
    cfg.my.deeply.nested.child.object.arg_a = 1
    cfg.my.deeply.nested.child.object.arg_b = 'b'

  Args:
    buildable: A ``Buildable`` (e.g. a ``fdl.Config``) to set values upon.
    **kwargs: The arguments and values to assign.
  """
  for name, value in kwargs.items():
    setattr(buildable, name, value)
