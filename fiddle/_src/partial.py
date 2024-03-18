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

"""Defines the `Partial` and `ArgFactory` classes and associated functions."""

from __future__ import annotations

import dataclasses
import functools
import itertools
from typing import Any, Callable, Dict, Generic, Tuple, Type, TypeVar, Union

from fiddle._src import arg_factory
from fiddle._src import config
from fiddle._src import daglish


T = TypeVar('T')
TypeOrCallableProducingT = Union[Callable[..., T], Type[T]]


@dataclasses.dataclass(frozen=True)
class _BuiltArgFactory:
  """The result of building an ``ArgFactory``.

  This wrapper is returned by ``ArgFactory.__build__``, and then consumed by the
  ``__build__`` method of the containing ``Partial`` or ``ArgFactory`` object.
  """

  factory: Callable[..., Any]


def _contains_arg_factory(value: Any) -> bool:
  """Returns true if ``value`` contains any ``_BuiltArgFactory`` instances."""

  def visit(node, state: daglish.State):
    if isinstance(node, _BuiltArgFactory):
      return True
    elif state.is_traversable(node):
      return any(state.yield_map_child_values(node))
    else:
      return False

  return daglish.MemoizedTraversal.run(visit, value)


def _invoke_arg_factories(value: Any) -> Any:
  """Makes a copy of value with any _BuiltArgFactory ``f`` replaced by ``f()``.

  The copy is "shallow" in the sense that only containers that (directly or
  indirectly) contain _BuiltArgFactories are replaced.  E.g.:

  >>> x = []
  >>> value = [[], _BuiltArgFactory(list)]
  >>> result = _invoke_arg_factories(value)
  >>> assert value[0] is result[0]  # not replaced

  Args:
    value: The structured value containing _BuiltArgFactories.

  Returns:
    A copy of ``value``.
  """

  def visit(node, state: daglish.State):
    if isinstance(node, _BuiltArgFactory):
      return node.factory()
    elif state.is_traversable(node):
      subtraversal = state.flattened_map_children(node)
      old_vals, _ = subtraversal.node_traverser.flatten(node)
      no_child_changed = all(
          [old is new for (old, new) in zip(old_vals, subtraversal.values)]
      )
      return node if no_child_changed else subtraversal.unflatten()
    else:
      return node

  return daglish.MemoizedTraversal.run(visit, value)


def _promote_arg_factory(arg: Any) -> Any:
  """Converts a structure-of-ArgFactory's to an ArgFactory-of-structure.

  If `arg` is a `_BuiltArgFactory`, or is a nested structure that doesn't
  contain any `_BuiltArgFactory`s, then return it as-is.

  Othewise, return a new `_BuiltArgFactory` whose factory returns a copy of
  `arg` with any `_BuiltArgFactory` `f` replaced by `f()`.

  Args:
    arg: The argument value to convert.

  Returns:
    `arg`; or a `_BuiltArgFactory` whose factory returns a copy of `arg` with
    any `_BuiltArgFactory` `f` replaced by `f()`.
  """
  if isinstance(arg, _BuiltArgFactory) or not _contains_arg_factory(arg):
    return arg
  else:
    return _BuiltArgFactory(functools.partial(_invoke_arg_factories, arg))


def _build_partial(
    fn: Callable[..., Any], args: Tuple[Any], kwargs: Dict[str, Any]
) -> functools.partial:
  """Returns `functools.partial` or `arg_factory.partial` for `fn`.

  If `args` or `kwargs` contain any `ArgFactory` instances, then return
  `arg_factory.partial(fn, *args, **kwargs)`.  Otherwise, return
  `functools.partial(fn, *args, **kwargs)`.  If any `args` or `kwargs`
  are nested structures that contain one or more `ArgFactory`s, then
  convert them to an `ArgFactory` that returns a copy of that structure with
  the nested `ArgFactory`s invoked.

  Args:
    fn: The function argument for the partial.
    args: The positional arguments for the partial.
    kwargs: The keyword arguments for the partial.
  """

  def is_arg_factory(value):
    return isinstance(value, _BuiltArgFactory)

  # If there are nested structures containing _BuiltArgFactory objects,
  # then promote them.
  args = [_promote_arg_factory(arg) for arg in args]
  kwargs = {name: _promote_arg_factory(arg) for name, arg in kwargs.items()}

  # Split the keyword args into those that should be handled by functools vs.
  # arg_factory.
  arg_factory_kwargs = {
      name: arg.factory for name, arg in kwargs.items() if is_arg_factory(arg)
  }
  functool_kwargs = {
      name: arg
      for name, arg in kwargs.items()
      if not isinstance(arg, _BuiltArgFactory)
  }

  # Group positional args by whether they are factories; and add partial
  # wrappers for each group.  Note: this never actually gets exercised by
  # fdl.build, since `fdl.build` calls `__build__(**arguments)`, and doesn't
  # pass in any positional arguments; so `args` will always be empty when
  # this is executed by `fdl.build()`.
  result = fn
  for is_factory, arg_values in itertools.groupby(args, is_arg_factory):
    if is_factory:
      arg_values = [arg.factory for arg in arg_values]
      result = arg_factory.partial(result, *arg_values, **arg_factory_kwargs)
      arg_factory_kwargs = {}
    else:
      result = functools.partial(result, *arg_values, **functool_kwargs)
      functool_kwargs = {}

  if arg_factory_kwargs:
    result = arg_factory.partial(result, **arg_factory_kwargs)
  return functools.partial(result, **functool_kwargs)


class Partial(Generic[T], config.Buildable[T]):
  """A ``Partial`` config creates a partial function or class when built.

  In some cases, it may be desired to leave a function or class uninvoked, and
  instead output a corresponding ``functools.partial`` object. Where the
  ``Config`` base class calls its underlying ``__fn_or_cls__`` when built, this
  ``Partial`` instead results in a partially bound function or class::

      partial_config = Partial(SampleClass)
      partial_config.arg = 1
      partial_config.kwarg = 'kwarg'
      partial_class = build(partial_config)
      instance = partial_class(arg=2)  # Keyword arguments can be overridden.
      assert instance.arg == 2
      assert instance.kwarg == 'kwarg'

  A ``Partial`` can also be created from an existing ``Config``, by using the
  ``fdl.cast()`` function. This results in a shallow copy that is decoupled from
  the ``Config`` used to create it. In the example below, any further changes to
  ``partial_config`` are not reflected by ``class_config`` (and vice versa)::

      partial_config = fdl.cast(Partial, class_config)
      class_config.arg = 'new value'  # Further modification to `class_config`.
      partial_class = build(partial_config)
      instance = partial_class()
      assert instance.arg == 'arg'  # The instance config is still 'arg'.

  ``Partial`` also supports configuring function/class with positional
  arguments. Please see docstring of ``Config`` class for details.
  """

  # NOTE(b/201159339): We currently need to repeat this annotation for pytype.
  __fn_or_cls__: TypeOrCallableProducingT[T]

  def __build__(self, *args, **kwargs):
    """Builds this ``Partial`` for the given ``args`` and ``kwargs``.

    This method is called during ``build`` to get the output for this
    ``Partial``.

    Args:
      *args: Positional arguments to partially bind to ``self.__fn_or_cls__``.
      **kwargs: Keyword arguments to partially bind to ``self.__fn_or_cls__``.

    Returns:
      A partial object (``functools.partial`` or ``arg_factory.partial``) for
      the callable ``self.__fn_or_cls__``, which binds the positional arguments
      ``args`` and the keyword arguments ``kwargs``.
    """
    return _build_partial(self.__fn_or_cls__, args, kwargs)


class ArgFactory(Generic[T], config.Buildable[T]):
  """A configuration that creates an argument factory when built.

  When an ``ArgFactory`` is used as a parameter for a ``fdl.Partial``, the
  partial function built from that ``fdl.Partial`` will construct a new value
  for the parameter each time it is called.  For example:

  >>> def f(x, noise): return x + noise
  >>> cfg = fdl.Partial(f, noise=fdl.ArgFactory(random.random))
  >>> p = fdl.build(cfg)
  >>> p(5) == p(5)  # noise has a different value for each call to `p`.
  False

  In contrast, if we replaced ``fdl.ArgFactory`` with ``fdl.Config`` in the
  above example, then the same noise value would be added each time ``p`` is
  called, since ``random.random`` would be called when ``fdl.build(cfg)`` is
  called.

  ``ArgFactory``'s can also be nested inside containers that are parameter
  values for a ``Partial``.  In this case, the partial function will construct
  the parameter value by copying the containers and replacing any ``ArgFactory``
  with the result of calling its factory.  Only the containers that (directly or
  indirectly) contain ``ArgFactory``'s are copied; any elements of the
  containers that do not contain ``ArgFactory``'s are not copied.

  ``ArgFactory`` can also be used as the parameter for another ``ArgFactory``,
  in which case a new value will be constructed for the child argument
  each time the parent argument is created.

  ``ArgFactory`` should *not* be used as a top-level configuration object, or
  as the argument to a ``fdl.Config``.
  """

  # TODO(b/272077508): Update build to raise an exception if ArgFactory is built
  # in an inappropriate context.

  # NOTE(b/201159339): We currently need to repeat this annotation for pytype.
  __fn_or_cls__: TypeOrCallableProducingT[T]

  def __build__(self, *args, **kwargs):
    if args or kwargs:
      return _BuiltArgFactory(_build_partial(self.__fn_or_cls__, args, kwargs))
    else:
      return _BuiltArgFactory(self.__fn_or_cls__)
