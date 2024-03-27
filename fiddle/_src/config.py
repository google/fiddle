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

"""Defines the `Config` class and associated subclasses and functions."""

from __future__ import annotations

import abc
import collections
import copy
import dataclasses
import functools
import types
from typing import Any, Callable, Collection, Dict, FrozenSet, Generic, Iterable, Mapping, NamedTuple, Optional, Set, Tuple, Type, TypeVar, Union

from fiddle._src import daglish
from fiddle._src import history
from fiddle._src import signatures
from fiddle._src import tag_type


T = TypeVar('T')
TypeOrCallableProducingT = Union[Callable[..., T], Type[T]]

NoValue = signatures.NoValue
# A sentinel object that represents no value is set for the argument.
NO_VALUE = signatures.NO_VALUE

# Unique object instance that should never be used by end-users, and can thus
# be used to differentiate between unset values and user-set values that are
# None or other commonly-used sentinel.
_UNSET_SENTINEL = object()


_defaults_aware_traverser_registry = daglish.NodeTraverserRegistry(
    use_fallback=True
)


class _Placeholder(object):
  """A Placeholder class to represent config arguments."""

  def __init__(self, index):
    self.index = index

  def __eq__(self, other):
    return self.index == other.index


def _buildable_flatten(
    buildable: Buildable, include_defaults: bool = False
) -> Tuple[Tuple[Any, ...], BuildableTraverserMetadata]:
  """Implement Buildable.__flatten__ method."""
  arguments = ordered_arguments(buildable, include_defaults=include_defaults)
  keys = tuple(arguments.keys())
  values = tuple(arguments.values())
  argument_tags = {
      name: frozenset(tags)
      for name, tags in buildable.__argument_tags__.items()
      if tags  # Don't include empty sets.
  }
  argument_history = {
      name: tuple(entries)
      for name, entries in buildable.__argument_history__.items()
  }
  metadata = BuildableTraverserMetadata(
      fn_or_cls=buildable.__fn_or_cls__,
      argument_names=keys,
      argument_tags=argument_tags,
      argument_history=argument_history,
  )
  return values, metadata


def _buildable_path_elements(
    buildable: Buildable, include_defaults: bool = False
) -> Tuple[daglish.PathElement]:
  """Implement Buildable.__path_elements__ method."""
  return tuple(
      daglish.Attr(name) if isinstance(name, str) else daglish.Index(name)
      for name in ordered_arguments(
          buildable, include_defaults=include_defaults
      ).keys()
  )


def _register_buildable_defaults_aware_traversers(cls: Type[Buildable]):
  """Registers defaults aware traversal routines for buildable subclasses."""
  _defaults_aware_traverser_registry.register_node_traverser(
      cls,
      flatten_fn=functools.partial(_buildable_flatten, include_defaults=True),
      unflatten_fn=cls.__unflatten__,
      path_elements_fn=functools.partial(
          _buildable_path_elements, include_defaults=True
      ),
  )


def _compare_buildable(x: Buildable, y: Buildable, check_dag: bool = False):
  """Compare if two Buildables are equal, including DAG structure."""
  assert isinstance(x, Buildable)
  if type(x) is not type(y):
    return False
  if x.__fn_or_cls__ != y.__fn_or_cls__:
    return False
  assert (
      x.__signature_info__.has_var_keyword
      == y.__signature_info__.has_var_keyword
  ), (  # pylint: disable=protected-access
      'Internal invariant violated: has_var_keyword should be the same if '
      "__fn_or_cls__'s are the same."
  )

  missing = object()

  def get_value_or_default(key: Union[str, int], buildable: Buildable) -> Any:
    value = buildable.__arguments__.get(key, missing)
    if value is missing:
      value = buildable.__signature_info__.get_default(key, missing)
    return value

  for key in set(x.__arguments__) | set(y.__arguments__):
    v1 = get_value_or_default(key, x)
    v2 = get_value_or_default(key, y)
    assert not (v1 is missing and v2 is missing)
    if v1 is missing or v2 is missing:
      return False
    if isinstance(v1, Buildable) and isinstance(v2, Buildable):
      if not _compare_buildable(v1, v2, check_dag=False):
        return False
    if v1 != v2:
      return False

  # Compare the DAG structure.
  # The DAG stracture comparison must traverse the whole DAG and sort the
  # result by path, which is expensive. Thus, we compare values first so
  # that most unequal cases will not reach the expensive DAG compare step.
  if check_dag:
    x_elements = list(
        daglish.iterate(
            x,
            memoized=True,
            # Not to memorize internables during traversal, as they might
            # be equal in value but have different object ids.
            memoize_internables=False,
            registry=_defaults_aware_traverser_registry,
        )
    )
    y_elements = list(
        daglish.iterate(
            y,
            memoized=True,
            memoize_internables=False,
            registry=_defaults_aware_traverser_registry,
        )
    )
    x_paths = sorted([elt[1] for elt in x_elements])
    y_paths = sorted([elt[1] for elt in y_elements])

    if len(x_paths) != len(y_paths):
      return False
    for x_path, y_path in zip(x_paths, y_paths):
      if x_path != y_path:
        return False

  return True


class BuildableTraverserMetadata(NamedTuple):
  """Metadata for a Buildable.

  This separate class is used for DAG traversals.
  """

  fn_or_cls: Callable[..., Any]
  argument_names: Tuple[Union[str, int], ...]
  argument_tags: Dict[str, FrozenSet[tag_type.TagType]]
  argument_history: Mapping[str, Tuple[history.HistoryEntry, ...]] = (
      types.MappingProxyType({})
  )

  def without_history(self) -> BuildableTraverserMetadata:
    return self._replace(argument_history={})

  def arguments(self, values: Iterable[Any]) -> Dict[str, Any]:
    """Returns a dictionary combining ``self.argument_names`` with ``values``."""
    return dict(zip(self.argument_names, values))

  def tags(self) -> Dict[str, set[tag_type.TagType]]:
    return collections.defaultdict(
        set, {name: set(tags) for name, tags in self.argument_tags.items()}
    )

  def history(self) -> history.History:
    return history.History(
        {name: list(entries) for name, entries in self.argument_history.items()}
    )


class Buildable(Generic[T], metaclass=abc.ABCMeta):
  """Base class for buildable types (``Config`` and ``Partial``).

  Buildable types implement a ``__build__`` method that is called during
  ``fdl.build()`` with arguments set on the ``Buildable`` to get output for the
  corresponding instance.

  Arguments are stored in the `__arguments__` dict with the following
  "canonical storage format":
  Positional-only and variadic-positional arguments use the position index of
  the argument as a key (of type int); all other arguments use the name of the
  argument as a key (of type str).
  """

  # "Dunder"-style names were chosen here to reduce the possibility of naming
  # conflicts while still making the fields accessible.
  __fn_or_cls__: TypeOrCallableProducingT[T]
  __arguments__: Dict[Union[str, int], Any]
  __argument_history__: history.History
  __argument_tags__: Dict[Union[str, int], Set[tag_type.TagType]]
  __signature_info__: signatures.SignatureInfo

  def __init__(
      self,
      fn_or_cls: Union['Buildable[T]', TypeOrCallableProducingT[T]],
      *args,
      **kwargs,
  ):
    """Initialize for ``fn_or_cls``, optionally specifying parameters.

    Args:
      fn_or_cls: A callable to configure. The signature of this callable will
        determine the parameters this ``Buidlable`` can be used to configure.
      *args: Any positional arguments to configure for ``fn_or_cls``.
      **kwargs: Any keyword arguments to configure for ``fn_or_cls``.
    """
    self.__init_callable__(fn_or_cls)
    arg_history = history.History()
    arg_history.add_new_value('__fn_or_cls__', fn_or_cls)
    super().__setattr__('__argument_history__', arg_history)
    super().__setattr__('__argument_tags__', collections.defaultdict(set))
    arguments = signatures.SignatureInfo.signature_binding(
        fn_or_cls, *args, **kwargs
    )

    for key, value in arguments.items():
      if isinstance(key, (str, int)):
        self._arguments_set_value(key, value)
      else:
        raise ValueError(
            f'Unexpected type received for the argument name: {key!r}'
        )

    for name, tags in tag_type.find_tags_from_annotations(fn_or_cls).items():
      self.__argument_tags__[name].update(tags)
      self.__argument_history__.add_updated_tags(
          name, self.__argument_tags__[name]
      )

  def __init_callable__(
      self, fn_or_cls: Union['Buildable[T]', TypeOrCallableProducingT[T]]
  ) -> None:
    """Save information on `fn_or_cls` to the `Buildable`."""
    if isinstance(fn_or_cls, Buildable):
      raise ValueError(
          'Using the Buildable constructor to convert a buildable to a new '
          'type or to override arguments is forbidden; please use either '
          '`fdl.cast(new_type, buildable)` (for casting) or '
          '`fdl.copy_with(buildable, **kwargs)` (for overriding arguments).'
      )

    # Using `super().__setattr__` here because assigning directly would trigger
    # our `__setattr__` override. Using `super().__setattr__` instead of special
    # casing these attribute names in `__setattr__` also has the effect of
    # making them easily gettable but not as easily settable by user code.
    super().__setattr__('__fn_or_cls__', fn_or_cls)
    super().__setattr__('__arguments__', {})
    signature = signatures.get_signature(fn_or_cls)
    # Several attributes are computed automatically by SignatureInfo during
    # `__post_init__`.
    super().__setattr__(
        '__signature_info__',
        signatures.SignatureInfo(signature=signature),
    )

  def __init_subclass__(cls):
    daglish.register_node_traverser(
        cls,
        flatten_fn=lambda x: x.__flatten__(),
        unflatten_fn=cls.__unflatten__,
        path_elements_fn=lambda x: x.__path_elements__(),
    )
    _register_buildable_defaults_aware_traversers(cls)

  @abc.abstractmethod
  def __build__(self, *args, **kwargs):
    """Builds output for this instance; see subclasses for details."""
    raise NotImplementedError()

  def __flatten__(self) -> Tuple[Tuple[Any, ...], BuildableTraverserMetadata]:
    return _buildable_flatten(self, include_defaults=False)

  @classmethod
  def __unflatten__(
      cls, values: Iterable[Any], metadata: BuildableTraverserMetadata
  ):
    rebuilt = cls.__new__(cls)
    rebuilt.__init_callable__(metadata.fn_or_cls)
    object.__setattr__(rebuilt, '__argument_tags__', metadata.tags())
    object.__setattr__(rebuilt, '__argument_history__', metadata.history())
    object.__setattr__(rebuilt, '__arguments__', metadata.arguments(values))
    return rebuilt

  def __path_elements__(self) -> Tuple[daglish.PathElement]:
    return _buildable_path_elements(self, include_defaults=False)

  def __getattr__(self, name: str):
    """Get parameter with given ``name``."""
    value = self.__arguments__.get(name, _UNSET_SENTINEL)
    # Check that positional-only arguments cannot be accessed by keywords.
    param = self.__signature_info__.parameters.get(name)
    if param is not None and (
        param.kind in (param.POSITIONAL_ONLY, param.VAR_POSITIONAL)
    ):
      raise AttributeError(
          'Cannot access positional-only or variadic positional arguments '
          f'{name} on {self!r} using attributes.'
      )

    if value is not _UNSET_SENTINEL:
      return value
    if dataclasses.is_dataclass(
        self.__fn_or_cls__
    ) and _field_uses_default_factory(self.__fn_or_cls__, name):
      raise ValueError(
          "Can't get default value for dataclass field "
          + f'{self.__fn_or_cls__.__qualname__}.{name} '
          + 'since it uses a default_factory.'
      )
    if param is not None and param.default is not param.empty:
      return param.default
    msg = f"No parameter '{name}' has been set on {self!r}."
    # TODO(b/219988937): Implement an edit distance function and display valid
    # attributes that are close to `name`.
    if hasattr(self.__fn_or_cls__, name):
      msg += (
          f' Note: {self.__fn_or_cls__.__module__}.'
          f'{self.__fn_or_cls__.__qualname__} has an attribute/method with '
          'this name, so this could be caused by using a '
          f'fdl.{self.__class__.__qualname__} in '
          'place of the actual function or class being configured. Did you '
          'forget to call `fdl.build(...)`?'
      )
    raise AttributeError(msg)

  def _arguments_set_value(self, key: Union[int, str], value: Any):
    """Set `self.__arguments__` values directly."""
    if isinstance(value, TaggedValueCls):
      tags = value.__argument_tags__.get('value', ())
      if tags:
        self.__argument_tags__[key].update(tags)
        self.__argument_history__.add_updated_tags(
            key, self.__argument_tags__[key]
        )
      if 'value' in value.__arguments__:
        value = value.__arguments__['value']
      else:
        return

    self.__arguments__[key] = value
    self.__argument_history__.add_new_value(key, value)

  def _arguments_del_value(self, key: Union[int, str]):
    """Delete `self.__arguments__` values directly."""
    del self.__arguments__[key]
    self.__argument_history__.add_deleted_value(key)

  def __setattr__(self, name: str, value: Any):
    """Sets parameter ``name`` to ``value``."""
    self.__signature_info__.validate_param_name(name, self.__fn_or_cls__)
    self._arguments_set_value(name, value)

  def __delattr__(self, name):
    """Unsets parameter ``name``."""
    try:
      self._arguments_del_value(name)
    except KeyError:
      err = AttributeError(f"No parameter '{name}' has been set on {self!r}")
      raise err from None

  def __getitem__(self, key: Any):
    """Get positional arguments by index."""
    key = self.__signature_info__.replace_varargs_handle(key)
    all_positional_args, _ = self.__signature_info__.transform_to_args_kwargs(
        self.__arguments__,
        include_pos_or_kw_in_args=True,
        include_no_value=True,
    )
    params = list(self.__signature_info__.parameters.values())
    for index, value in enumerate(all_positional_args):
      if value is NO_VALUE and index < len(params):
        param = params[index]
        if param.default is not param.empty:
          all_positional_args[index] = param.default
    return all_positional_args[key]

  def __delitem__(self, key: Any):
    """Delete positional arguments by index."""
    key = self.__signature_info__.replace_varargs_handle(key)
    all_positional_args, _ = self.__signature_info__.transform_to_args_kwargs(
        self.__arguments__,
        include_pos_or_kw_in_args=True,
        include_no_value=True,
    )
    var_positional_start = self.__signature_info__.var_positional_start
    if isinstance(key, slice):
      key = key.indices(len(all_positional_args))
      indices = list(range(*key))
    else:
      if key < 0:
        key += len(all_positional_args)
      indices = [key]

    old_placeholders = [
        _Placeholder(index) for index in range(len(all_positional_args))
    ]
    new_placeholders = old_placeholders.copy()
    # Traverse from largest index to maintain order of undeleted indices.
    for index in indices[::-1]:
      if index < var_positional_start:
        k = self.__signature_info__.index_to_key(index, self.__arguments__)
        if k in self.__arguments__:
          self._arguments_del_value(k)
      else:
        del new_placeholders[index]

    # Delete var-positional args and compact the *args list.
    for index in range(var_positional_start, len(old_placeholders)):
      if index < len(new_placeholders):
        if new_placeholders[index] != old_placeholders[index]:
          new_value = self.__arguments__[new_placeholders[index].index]
          self._arguments_set_value(index, new_value)
      else:
        self._arguments_del_value(index)

  def _set_item_by_index(self, key: int, value: Any):
    """Set positional arguments by index."""
    key = self.__signature_info__.index_to_key(key, self.__arguments__)
    positional_num = self.__signature_info__.var_positional_start
    if positional_num is None:
      # *args does not exist
      positional_num = len(self.__signature_info__.parameters)
      if self.__signature_info__.var_keyword_name:
        # Exclude **kwargs
        positional_num -= 1

    # Cannot set item when index is beyond current positional args list length.
    # Only index that points to *args can be out of range.
    if (
        isinstance(key, int)
        and key >= positional_num
        and key not in self.__arguments__
    ):
      raise IndexError(
          f'Cannot set positional argument with index {key}'
          ' (index out of range).'
      )
    self._arguments_set_value(key, value)

  def _set_item_by_slice(self, slice_key: slice, value: Any):
    """Set positional arguments by slice."""
    all_positional_args, _ = self.__signature_info__.transform_to_args_kwargs(
        self.__arguments__,
        include_pos_or_kw_in_args=True,
        include_no_value=True,
    )
    var_positional_start = self.__signature_info__.var_positional_start
    index_range = slice_key.indices(len(all_positional_args))
    if var_positional_start is None or index_range[0] < var_positional_start:
      # The slice key spans on non-variadic positional arguments, this set item
      # operation cannot modify the total length of full positiona args list.
      indices = range(*index_range)
      if len(indices) != len(value):
        raise ValueError(
            'Cannot modify the total length of full positional arguments list'
            ' with __setitem__ when the slice key spans over non-variadic'
            ' positional arguments. To remove values, use `del config[key]`.'
            ' To append values to variadic positional arguments, you can use '
            ' config[fdl.VARARGS:] = value.'
        )
      for index, v in zip(indices, value):
        self._set_item_by_index(index, v)
    else:
      # The slice key only spans on variadic positional arguments
      old_placeholders = [
          _Placeholder(index) for index in range(len(all_positional_args))
      ]
      new_placeholders = old_placeholders.copy()
      new_placeholders[slice_key] = value
      for index in range(var_positional_start, len(old_placeholders)):
        if index < len(new_placeholders):
          new_value = new_placeholders[index]
          if isinstance(new_value, _Placeholder):
            if new_value == old_placeholders[index]:
              continue
            else:
              new_value = self.__arguments__[new_value.index]
          self._arguments_set_value(index, new_value)
        else:
          self._arguments_del_value(index)
      len_old = len(old_placeholders)
      len_new = len(new_placeholders)
      for index in range(len_old, len_new):
        new_value = new_placeholders[index]
        if isinstance(new_value, _Placeholder):
          new_value = self.__arguments__[new_value.index]
        self._arguments_set_value(index, new_value)

  def __setitem__(self, key: Any, value: Any):
    """Set positional arguments by index."""
    key = self.__signature_info__.replace_varargs_handle(key)
    if isinstance(key, int):
      self._set_item_by_index(key, value)
    elif isinstance(key, slice):
      self._set_item_by_slice(key, value)

  def __dir__(self) -> Collection[str]:
    """Provide a useful list of attribute names, optimized for Jupyter/Colab.

    ``__dir__`` is often implicitly called by tooling such as Jupyter/Colab to
    provide autocomplete suggestions. This implementation of ``__dir__`` makes
    it easy to see what are valid attributes to get, set, or delete.

    Returns:
      A list of useful attribute names corresponding to set or unset parameters.
    """
    set_argument_names = self.__arguments__.keys()
    valid_param_names = set(self.__signature_info__.valid_param_names)
    all_names = valid_param_names.union(set_argument_names)
    return all_names

  def __repr__(self):
    if hasattr(self.__fn_or_cls__, '__qualname__'):
      formatted_fn_or_cls = self.__fn_or_cls__.__qualname__
    else:
      formatted_fn_or_cls = str(self.__fn_or_cls__)
    formatted_params = []
    # Show parameters from signature first (in signature order) followed by
    # **varkwarg parameters (in the order they were set).
    param_names = list(self.__signature_info__.parameters) + [
        name
        for name in self.__arguments__
        if name not in self.__signature_info__.parameters
    ]

    for name in param_names:
      tags = self.__argument_tags__.get(name, ())
      value = self.__arguments__.get(name, NO_VALUE)
      if tags or (value is not NO_VALUE):
        param_str = str(name)
        if tags:
          param_str += f"[{', '.join(sorted(str(tag) for tag in tags))}]"
        if value is not NO_VALUE:
          param_str += f'={value!r}'
        formatted_params.append(param_str)

    name = type(self).__name__

    formatted_params_no_linebreak = ', '.join(formatted_params)
    # An arbitrary threshold to determine whether to add linebreaks to args.
    if (
        len(formatted_params_no_linebreak)
        + len(name)
        + len(formatted_fn_or_cls)
        > 80
    ):

      def indent(s):
        return '\n'.join(['  ' + x for x in s.split('\n')])

      formatted_params = ','.join(['\n' + indent(x) for x in formatted_params])
    else:
      formatted_params = formatted_params_no_linebreak

    return f'<{name}[{formatted_fn_or_cls}({formatted_params})]>'

  def __copy__(self):
    """Shallowly copies this ``Buildable`` instance.

    This copy implementation ensures that setting parameters on a copy of a
    ``Buildable`` won't affect the original instance. However, the copy is
    shallow, so parameter values will still refer to the same instances of
    objects after the copy.

    Returns:
      A shallow copy of this ``Buildable``.
    """
    # TODO(b/231368256): Preserve argument history...
    return self.__unflatten__(*self.__flatten__())

  def __deepcopy__(self, memo: Dict[int, Any]):
    """Deepcopies this Buildable, skipping copying of ``__signature_info__``."""
    # Skipping copying inspect.Signature objects, which are generally immutable,
    # is about 2x faster on artificial benchmarks.
    memo[id(self.__signature_info__.signature)] = (
        self.__signature_info__.signature
    )
    result = object.__new__(type(self))
    result.__dict__.update(copy.deepcopy(self.__dict__, memo))
    return result

  def __eq__(self, other):
    """Returns true iff self and other contain the same argument values.

    This compares the specific types of ``self`` and ``other``, the function or
    class being configured, and then checks for equality in the configured
    arguments.

    Default argument values are considered in this comparison: If one
    ``Buildable`` has an argument explicitly set to its default value while
    another does not, the two will still be considered equal (motivated by the
    fact that calls to the function or class being configured will be the same).

    Argument history is not compared (i.e., it doesn't matter how the
    ``Buildable``'s being compared reached their current state).

    Args:
      other: The other value to compare ``self`` to.

    Returns:
      ``True`` if ``self`` equals ``other``, ``False`` if not.
    """
    return _compare_buildable(self, other, check_dag=True)

  def __getstate__(self):
    """Gets pickle serialization state, removing some fields.

    For now, we discard the ``__signature_info__.signature`` (which can be
    recalculated), because these tend to contain values which cannot be
    serialized.

    Returns:
      Dict of serialized state.
    """
    result = dict(self.__dict__)
    # During serialization, set `__signature_info__` as None to avoid signature
    # serialization issues.
    result['__signature_info__'] = None
    return result

  def __setstate__(self, state) -> None:
    """Loads pickle serialization state.

    This re-derives the signature if not present.

    Args:
      state: State dictionary, from ``__getstate__``.
    """
    self.__dict__.update(state)  # Support unpickle.
    if self.__signature_info__ is None:
      signature = signatures.get_signature(self.__fn_or_cls__)
      super().__setattr__(
          '__signature_info__',
          signatures.SignatureInfo(signature=signature),
      )


class Config(Generic[T], Buildable[T]):
  """A mutable representation of a function or class's parameters.

  This class represents the configuration for a given function or class,
  exposing configured parameters as mutable attributes. For example, given a
  class::

      class SampleClass:
        '''Example class for demonstration purposes.'''

        def __init__(self, arg, kwarg=None):
          self.arg = arg
          self.kwarg = kwarg

  a configuration may (for instance) be accomplished via::

      class_config = Config(SampleClass, 1, kwarg='kwarg')

  or via::

      class_config = Config(SampleClass)
      class_config.arg = 1
      class_config.kwarg = 'kwarg'

  A function can be configured in the same ways::

      def test_function(arg, kwarg=None):
        return arg, kwarg

      fn_config = Config(test_function, 1)
      fn_config.kwarg = 'kwarg'

  If the class/function has positional arguments, they can be accessed through
  the `[]` syntax::

      def test_function(a, b, /, c, *args):
        return locals()

      fn_config = Config(test_function, 1, 2, 3, 4, 5)

      # Read
      assert fn_config[0] == 1
      assert fn_config[:] == [1, 2, 3, 4, 5]

      # Modify
      fn_config[0] = 'a'
      fn_config.c = 'c'

      # `fdl.VARARGS` represents the start of variadic positional args (*args)
      fn_config[fdl.VARARGS:] = ['x', 'y']
      assert fn_config[:] == [1, 2, 3, 'x', 'y']

      # Delete
      del fn_config[0]
      del fn_config[fdl.VARARGS:]
      assert fn_config[:] == [fdl.NO_VALUE, 2, 3]

  NOTE: Directly calling `list` methods like `append` and `extend` is not
  supported, and will not mutate the config. Like with Python lists, slice
  operations on Configs effectively create a copy of the underlying sequence.

  NOTE: If using `slice` as key for modifying the config, and the `slice` spans
  over positional-only or positional-or-keyword arguments, the provided value
  must have the same length as that of the slice range.

  fn_config[2:4] = ['a', 'b'] # OK
  fn_config[2:4] = ['m']      # Not OK. Will raise an error!

  A ``Config`` instance may be transformed into instances and function outputs
  by passing it to the ``build`` function. The ``build`` function invokes each
  function or class in the configuration tree (appropriately propagating the
  built outputs from nested ``Config``'s). For example, using the
  ``SampleClass`` config from above::

      instance = build(class_config)
      assert instance.arg == 1
      assert instance.kwarg == 'kwarg'

  If the same ``Config`` instance is used in multiple places within the
  configuration tree, its function or class is invoked only once during
  ``build``, and the result shared across all occurrences of the ``Config``
  instance. (See ``build`` documentation for further details.) To create a new
  instance of a ``Config`` with the same parameter settings that will yield a
  separate instance during ``build``, ``copy.copy()`` or ``copy.deepcopy()``
  may be used.
  """

  # NOTE(b/201159339): We currently need to repeat these annotations for pytype.
  __fn_or_cls__: TypeOrCallableProducingT[T]
  __signature_info__: signatures.SignatureInfo

  def __build__(self, *args, **kwargs):
    """Builds this ``Config`` for the given ``args`` and ``kwargs``.

    This method is called during `build` to get the output for this `Config`.

    Args:
      *args: Positional arguments to pass to ``self.__fn_or_cls__``.
      **kwargs: Keyword arguments to pass to ``self.__fn_or_cls__`.

    Returns:
      The result of calling ``self.__fn_or_cls__`` with the given ``args`` and
      ``kwargs``.
    """
    return self.__fn_or_cls__(*args, **kwargs)


def tagged_value_fn(
    value: Union[T, NoValue], tags: Optional[Set[tag_type.TagType]] = None
) -> T:
  """Identity function to return value if set, and raise an error if not.

  Args:
    value: The value to return.
    tags: The tags associated with the value. (Used in generating error messages
      if `value` is not set.)

  Returns:
    The value `value` passed to it.
  """
  if value is NO_VALUE:
    msg = (
        'Expected all `TaggedValue`s to be replaced via fdl.set_tagged() '
        'calls, but one was not set.'
    )
    if tags:
      msg += ' Unset tags: ' + str(tags)
    raise tag_type.TaggedValueNotFilledError(msg)
  return value


class TaggedValueCls(Generic[T], Config[T]):
  """Placeholder class for TaggedValue instances.

  Instances of this class are generally transitory; when passed as an argument
  of a Fiddle Buildable, will be expanded into that argument's values and tags.
  However, they may survive as stand-alone objects within tuples, lists, and
  dictionaries.
  """

  # NOTE(b/201159339): We currently need to repeat these annotations for pytype.
  __fn_or_cls__: TypeOrCallableProducingT[T]
  __signature_info__: signatures.SignatureInfo

  @property
  def tags(self):
    return self.__argument_tags__['value']

  def __build__(self, *args: Any, **kwargs: Any) -> T:
    if self.__fn_or_cls__ is not tagged_value_fn:
      raise RuntimeError(
          'Unexpected __fn_or_cls__ in TaggedValueCls; found:'
          f'{self.__fn_or_cls__}'
      )
    return self.__fn_or_cls__(tags=self.tags, *args, **kwargs)


def _field_uses_default_factory(dataclass_type: Type[Any], field_name: str):
  """Returns true if <dataclass_type>.<field_name> uses a default_factory."""
  for field in dataclasses.fields(dataclass_type):
    if field.name == field_name:
      return field.default_factory != dataclasses.MISSING
  return False


BuildableT = TypeVar('BuildableT', bound=Buildable)


def get_callable(buildable: Buildable[T]) -> Union[Callable[..., T], Type[T]]:
  """Returns the callable of a Buildable."""
  return buildable.__fn_or_cls__


def ordered_arguments(
    buildable: Buildable,
    *,
    include_var_keyword: bool = True,
    include_defaults: bool = False,
    include_unset: bool = False,
    include_positional: bool = True,
    include_equal_to_default: bool = True,
) -> Dict[Union[int, str], Any]:
  """Returns arguments of a Buildable, ordered by the signature.

  Args:
    buildable: The buildable whose arguments should be returned.
    include_var_keyword: If True, then include arguments that will be consumed
      by the buildable's callable's `VAR_KEYWORD` parameter (e.g. `**kwargs`).
    include_defaults: If True, then include arguments that have not been
      explicitly set, but that have a default value.  Can not be combined with
      `include_equal_to_default=False`.
    include_unset: If True, then include arguments that have not been explicitly
      set, that don't have a default value.  The value for these parameters will
      be `fdl.NO_VALUE`.
    include_positional: If False, positional-only and variadic positional
      arguments will be excluded from the output.
    include_equal_to_default: If False, then exclude arguments that are equal to
      their default value (using `==`).  Can not be combined with
      `include_defaults=True`.

  Returns:
    A dictionary mapping argument keys to values or `fdl.NO_VALUE`. Argument
    keys are either positional indices or names.
  """
  if not include_equal_to_default and include_defaults:
    raise ValueError(
        'Exclude_equal_to_default and include_defaults are mutually exclusive.'
    )
  result = {}
  unset = object()
  for index, (name, param) in enumerate(
      buildable.__signature_info__.parameters.items()
  ):
    if param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
      value = unset
      if name in buildable.__arguments__ or (
          index in buildable.__arguments__
          and param.kind == param.POSITIONAL_ONLY
      ):
        if name in buildable.__arguments__:
          value = buildable.__arguments__[name]
        else:
          value = buildable.__arguments__[index]
      elif param.default is not param.empty:
        if include_defaults:
          value = param.default
      elif include_unset:
        value = NO_VALUE
      if value is not unset:
        if include_equal_to_default or (value != param.default):
          if param.kind == param.POSITIONAL_ONLY:
            result[index] = value
          else:
            result[name] = value

    if param.kind == param.VAR_POSITIONAL:
      # Cannot add defaults for *args in Python, so ignore defaults related
      # flags and `include_unset` flag here.
      while index in buildable.__arguments__:
        result[index] = buildable.__arguments__[index]
        index += 1

  if include_var_keyword:
    for name, value in buildable.__arguments__.items():
      param = buildable.__signature_info__.parameters.get(name)
      if param is None or param.kind == param.VAR_KEYWORD:
        result[name] = value

  if not include_positional:
    result = {k: v for k, v in result.items() if isinstance(k, str)}

  return result
