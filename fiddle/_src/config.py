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


class NoValue:
  """Sentinel class for arguments with no value."""

  def __repr__(self):
    return 'fdl.NO_VALUE'

  def __deepcopy__(self, memo):
    """Override for deepcopy that does not copy this sentinel object."""
    del memo
    return self

  def __copy__(self):
    """Override for `copy.copy()` that does not copy this sentinel object."""
    return self


NO_VALUE = NoValue()

# Unique object instance that should never be used by end-users, and can thus
# be used to differentiate between unset values and user-set values that are
# None or other commonly-used sentinel.
_UNSET_SENTINEL = object()


_defaults_aware_traverser_registry = daglish.NodeTraverserRegistry(
    use_fallback=True
)


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
) -> Tuple[daglish.Attr]:
  """Implement Buildable.__path_elements__ method."""
  return tuple(
      daglish.Attr(name)
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
  for key in set(x.__arguments__) | set(y.__arguments__):
    v1 = getattr(x, key, missing)
    v2 = getattr(y, key, missing)
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
  argument_names: Tuple[str, ...]
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
  """

  # "Dunder"-style names were chosen here to reduce the possibility of naming
  # conflicts while still making the fields accessible.
  __fn_or_cls__: TypeOrCallableProducingT[T]
  __arguments__: Dict[str, Any]
  __argument_history__: history.History
  __argument_tags__: Dict[str, Set[tag_type.TagType]]
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

    for name, value in arguments.items():
      setattr(self, name, value)

    for name, tags in tag_type.find_tags_from_annotations(fn_or_cls).items():
      self.__argument_tags__[name].update(tags)
      self.__argument_history__.add_updated_tags(
          name, self.__argument_tags__[name]
      )

  def __init_callable__(
      self, fn_or_cls: Union['Buildable[T]', TypeOrCallableProducingT[T]]
  ) -> None:
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
    super().__setattr__(
        '__signature_info__',
        signatures.SignatureInfo(signature),
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

  def __path_elements__(self) -> Tuple[daglish.Attr]:
    return _buildable_path_elements(self, include_defaults=False)

  def __getattr__(self, name: str):
    """Get parameter with given ``name``."""
    value = self.__arguments__.get(name, _UNSET_SENTINEL)

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
    param = self.__signature_info__.parameters.get(name)
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

  def __setattr__(self, name: str, value: Any):
    """Sets parameter ``name`` to ``value``."""

    self.__signature_info__.validate_param_name(name, self.__fn_or_cls__)

    if isinstance(value, TaggedValueCls):
      tags = value.__argument_tags__.get('value', ())
      if tags:
        self.__argument_tags__[name].update(tags)
        self.__argument_history__.add_updated_tags(
            name, self.__argument_tags__[name]
        )
      if 'value' in value.__arguments__:
        value = value.__arguments__['value']
      else:
        return

    self.__arguments__[name] = value
    self.__argument_history__.add_new_value(name, value)

  def __delattr__(self, name):
    """Unsets parameter ``name``."""
    try:
      del self.__arguments__[name]
      self.__argument_history__.add_deleted_value(name)
    except KeyError:
      err = AttributeError(f"No parameter '{name}' has been set on {self!r}")
      raise err from None

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
      value = self.__arguments__.get(name, _UNSET_SENTINEL)
      if tags or (value is not _UNSET_SENTINEL):
        param_str = name
        if tags:
          param_str += f"[{', '.join(sorted(str(tag) for tag in tags))}]"
        if value is not _UNSET_SENTINEL:
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
    result['__signature_info__'] = signatures.SignatureInfo(  # pytype: disable=wrong-arg-types
        None, result['__signature_info__'].has_var_keyword
    )
    return result

  def __setstate__(self, state) -> None:
    """Loads pickle serialization state.

    This re-derives the signature if not present.

    Args:
      state: State dictionary, from ``__getstate__``.
    """
    self.__dict__.update(state)  # Support unpickle.
    if self.__signature_info__.signature is None:
      self.__signature_info__.signature = signatures.get_signature(
          self.__fn_or_cls__
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
  ``build``, and the result shared across all occurences of the ``Config``
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


def update_callable(
    buildable: Buildable,
    new_callable: TypeOrCallableProducingT,
    drop_invalid_args: bool = False,
):
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
  # Note: can't just call config.__init__(new_callable, **config.__arguments__)
  # to preserve history.
  #
  # Note: can't call `setattr` on all the args to validate them, because that
  # will result in duplicate history entries.
  original_args = buildable.__arguments__
  signature = signatures.get_signature(new_callable)
  if any(
      param.kind == param.VAR_POSITIONAL
      for param in signature.parameters.values()
  ):
    raise NotImplementedError(
        'Variable positional arguments (aka `*args`) not supported.'
    )
  signature_info = signatures.SignatureInfo(signature)
  object.__setattr__(
      buildable,
      '__signature_info__',
      signature_info,
  )
  if not signature_info.has_var_keyword:
    invalid_args = [
        arg for arg in original_args.keys() if arg not in signature.parameters
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


def assign(buildable: Buildable, **kwargs):
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


BuildableT = TypeVar('BuildableT', bound=Buildable)


def copy_with(buildable: BuildableT, **kwargs) -> BuildableT:
  """Returns a shallow copy of ``buildable`` with updates to arguments.

  Args:
    buildable: A ``Buildable`` (e.g. a ``fdl.Config``) to copy and mutate.
    **kwargs: The arguments and values to assign.
  """
  buildable = copy.copy(buildable)
  assign(buildable, **kwargs)
  return buildable


def deepcopy_with(buildable: Buildable, **kwargs):
  """Returns a deep copy of ``buildable`` with updates to arguments.

  Note: if any ``Config``'s inside ``buildable`` are shared with ``Config``'s
  outside of ``buildable``, then they will no longer be shared in the returned
  value. E.g., if ``cfg1.x.y is cfg2``, then
  ``fdl.deepcopy_with(cfg1, ...).x.y is cfg2`` will be ``False``.

  Args:
    buildable: A ``Buildable`` (e.g. a ``fdl.Config``) to copy and mutate.
    **kwargs: The arguments and values to assign.
  """
  buildable = copy.deepcopy(buildable)
  assign(buildable, **kwargs)
  return buildable


def get_callable(buildable: Buildable[T]) -> Union[Callable[..., T], Type[T]]:
  """Returns the callable of a Buildable."""
  return buildable.__fn_or_cls__


def ordered_arguments(
    buildable: Buildable,
    *,
    include_var_keyword: bool = True,
    include_defaults: bool = False,
    include_unset: bool = False,
    exclude_equal_to_default: bool = False,
) -> Dict[str, Any]:
  """Returns arguments of a Buildable, ordered by the signature.

  Args:
    buildable: The buildable whose arguments should be returned.
    include_var_keyword: If True, then include arguments that will be consumed
      by the buildable's callable's `VAR_KEYWORD` parameter (e.g. `**kwargs`).
    include_defaults: If True, then include arguments that have not been
      explicitly set, but that have a default value.  Can not be combined with
      `exclude_equal_to_default=True`.
    include_unset: If True, then include arguments that have not been explicitly
      set, that don't have a default value.  The value for these parameters will
      be `fdl.NO_VALUE`.
    exclude_equal_to_default: If True, then exclude arguments that are equal to
      their default value (using `==`).  Can not be combined with
      `include_defaults=True`.

  Returns:
    A dictionary mapping argument names to values or `fdl.NO_VALUE`.
  """
  if exclude_equal_to_default and include_defaults:
    raise ValueError(
        'exclude_equal_to_default and include_defaults are mutually exclusive.'
    )
  result = {}
  for name, param in buildable.__signature_info__.parameters.items():
    if param.kind != param.VAR_KEYWORD:
      if name in buildable.__arguments__:
        value = buildable.__arguments__[name]
        if (not exclude_equal_to_default) or (value != param.default):
          result[name] = value
      elif param.default is not param.empty:
        if include_defaults:
          result[name] = param.default
      elif include_unset:
        result[name] = NO_VALUE
  if include_var_keyword:
    for name, value in buildable.__arguments__.items():
      param = buildable.__signature_info__.parameters.get(name)
      if param is None or param.kind == param.VAR_KEYWORD:
        result[name] = value
  return result
