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

"""Tags attach metadata to arguments & values for ergonomic fiddling.

When defining shared parameters across a project that later could be changed,
for example dtype or activation function, we encourage the following coding
pattern: a tiny shared library file should declare tags for an
entire project, like

  class ActivationDType(fdl.Tag):
    "Data type (e.g. jnp.float32, or jnp.bfloat16) for intermediate values."
  class KernelInitFn(fdl.Tag):
    "A function from RNG & hyperparameters to initial weights."

Then, in library code which configures shared Fiddle fixtures, these tags are
used,

  def layer_norm_fixture() -> fdl.Config[LayerNorm]:
    cfg = fdl.Config(LayerNorm)
    cfg.dtype = ActivationDType.new(default=jnp.float32)

And in experiment code stitching everything together, all of these tagged values
can be set at once,

  def encoder_decoder_fixture() -> fdl.Config[EncoderDecoder]:
    cfg = fdl.Config(EncoderDecoder)
    ...
    cfg.encoder.encoder_norm = layer_norm_fixture()

    # Set all activation dtypes.
    fdl.set_tagged(cfg, ActivationDType, jnp.bfloat16)
    return cfg

  model = fdl.build(encoder_decoder_fixture())

Tags can be defined in a module (file) or inside classes, but cannot be defined
within a function, as there would not be a way to refer to it without
re-invoking the function, defining a new unique Tag.

Tags can inherit from each other. Use this power wisely.

While the normal mechanism for creating a TaggedValue is via $TAG.new(), you can
also create a TaggedValue explicitly, allowing multiple tags to be associated
with the value.
"""

from __future__ import annotations

import copy
import inspect
from typing import Any, Collection, FrozenSet, Generic, Iterable, Set, TypeVar, Union

from fiddle import config
from fiddle.experimental import serialization
import tree


class TaggedValueNotFilledError(ValueError):
  """A TaggedValue was not filled when build() was called."""


class _NoValue:
  """Sentinel class (used in place of object for more precise errors)."""

  def __deepcopy__(self, memo):
    """Override for deepcopy that does not copy this sentinel object."""
    del memo
    return self


NO_VALUE = _NoValue()

serialization.register_node_traverser(
    _NoValue,
    flatten_fn=lambda _: ((), None),
    unflatten_fn=lambda values, metadata: NO_VALUE,
    path_elements_fn=lambda _: ())


class TagType(type):
  """All Fiddle tags are instances of this class.

  For defining Tags, we leverage Python's class definition and documentation
  syntax.

  See the documentation on `fdl.Tag` for instructions on how to use.
  """

  def __init__(cls, name, bases, dct):
    if '__doc__' not in dct:
      raise TypeError('You must provide a tag description with a docstring.')
    if '__qualname__' not in dct:
      raise TypeError('No `__qualname__` property found.')
    if '<' in dct['__qualname__']:
      raise TypeError('You cannot define a tag within a function or lambda.')
    super().__init__(name, bases, dct)

  def __call__(cls, *args, **kwds):
    raise TypeError('You cannot instantiate Fiddle tags (trying to instantiate '
                    f'{cls.name}); just use the type itself (no parenthesis) '
                    'when specifying tags on a TaggedValue(...) call, or call '
                    '`.new(default=)` to create a new `TaggedValue`.')

  @property
  def description(cls) -> str:
    """A string describing the semantics and intended usecases for this tag."""
    return cls.__doc__

  @property
  def name(cls) -> str:
    """A unique name for this tag."""
    return f'{cls.__module__}.{cls.__qualname__}'

  def __str__(cls) -> str:
    return f'#{cls.name}'

  def __repr__(cls) -> str:  # pylint: disable=invalid-repr-returned
    return cls.name


class Tag(metaclass=TagType):
  """Metadata associated with a Fiddle configurable value.

  To see a usage example, please see the documentation on the tagging module.

  Note: Tags cannot be instantiated.
  """

  @classmethod
  def new(cls, default: Any = NO_VALUE) -> TaggedValue:
    """Creates a new `TaggedValue` with `cls` as the only tag.

    If you would like to create a `TaggedValue` with multiple tags attached,
    directly call the `TaggedValue` constructor:

    ```py
    tagged_value = TaggedValue(tags=(Tag1, Tag2, Tag3), default=Foo)
    ```

    Args:
      default: Optional. The default value for this TaggedValue.

    Returns:
      A TaggedValue tagged with the tag `cls`.
    """
    return TaggedValue(tags=(cls,), default=default)


def tagvalue_fn(tags: Set[TagType], value: Any = NO_VALUE) -> Any:
  if value is NO_VALUE:
    raise TaggedValueNotFilledError(
        'Expected all `TaggedValue`s to be replaced via fdl.set_tagged() '
        f'calls, but one with tags {tags} was not set.')
  else:
    return value


T = TypeVar('T')


class TaggedValue(Generic[T], config.Config[T]):
  """Declares a value annotated with a set of `Tag`s."""

  def __init__(
      self,
      tags: Collection[TagType],
      default: Union[_NoValue, T] = NO_VALUE,
  ):
    """Initializes the TaggedValue.

    Args:
      tags: A non-empty collection of tags to associated with this value.
      default: Default value of the TaggedValue. By default this is a sentinel
        which will cause the configuration to fail to build when the
        placeholders are not set.
    """
    if isinstance(tags, TaggedValue):
      # Handle copy.copy call (`type(self)(self)`).
      other = tags
      super().__init__(tagvalue_fn, tags=other.tags, value=other.value)
      return
    if not tags:
      raise ValueError('At least one tag must be provided.')
    super().__init__(tagvalue_fn, tags=set(tags), value=default)

  def __deepcopy__(self, memo) -> config.Buildable[T]:
    """Implements the deepcopy API."""
    return TaggedValue(tags=self.tags, default=copy.deepcopy(self.value, memo))

  @classmethod
  def __unflatten__(
      cls,
      values: Iterable[Any],
      metadata: config.BuildableTraverserMetadata,
  ) -> TaggedValue:
    tags, default = values
    return cls(tags, default)


# TODO: Migrate users of this API to a `select`-based API.
def set_tagged(root: config.Buildable, *, tag: TagType, value: Any) -> None:
  """Sets all parameters in `root` tagged with `tag` to `value`.

  Args:
    root: The root of a DAG of Buildables.
    tag: The tag to search for.
    value: Value to set for all parameters tagged with `tag`.
  """
  if isinstance(root, TaggedValue):
    if any(issubclass(t, tag) for t in root.tags):
      root.value = value

  def map_fn(leaf):
    if isinstance(leaf, config.Buildable):
      set_tagged(root=leaf, tag=tag, value=value)
    return leaf

  # TODO: Use something other than map to avoid creating unused garbage.
  tree.map_structure(map_fn, root.__arguments__)


def list_tags(
    root: config.Buildable,
    add_superclasses=False,
) -> FrozenSet[TagType]:
  """Lists all tags in a buildable.

  Args:
    root: The root of a DAG of `Buildable`s.
    add_superclasses: For tags that inherit from other tags, add the
      superclasses as well.

  Returns:
    Set of tags used in this buildable.
  """
  tags = set()

  def _inner(node: config.Buildable):
    if isinstance(node, TaggedValue):
      tags.update(node.tags)

    def map_fn(leaf):
      if isinstance(leaf, config.Buildable):
        _inner(leaf)
      return leaf

    # TODO: Use something other than map for efficiency.
    tree.map_structure(map_fn, node.__arguments__)

  _inner(root)

  # Add superclasses if desired.
  if add_superclasses:
    for tag in list(tags):
      for base in inspect.getmro(tag):
        if base not in tags and base is not Tag and issubclass(base, Tag):
          tags.add(base)

  return frozenset(tags)
