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

import collections
import inspect
import typing
from typing import Any, Collection, FrozenSet, Generic, Optional, TypeVar, Union

from fiddle import config
from fiddle import daglish
from fiddle import tag_type
from fiddle.experimental import auto_config
from fiddle.experimental import serialization

TagType = tag_type.TagType
TaggedValueNotFilledError = tag_type.TaggedValueNotFilledError


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


class Tag(metaclass=TagType):
  """Metadata associated with a Fiddle configurable value.

  To see a usage example, please see the documentation on the tagging module.

  Note: Tags cannot be instantiated.
  """

  @classmethod
  def new(cls, default: Any = NO_VALUE) -> TaggedValueCls:
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

  if not typing.TYPE_CHECKING:
    new = classmethod(
        auto_config.AutoConfig(
            func=new.__func__, buildable_func=new.__func__, always_inline=True))


T = TypeVar('T')


def tagged_value_identity_fn(value: Union[T, _NoValue] = NO_VALUE) -> T:
  if value is NO_VALUE:
    raise TaggedValueNotFilledError(
        'Expected all `TaggedValue`s to be replaced via fdl.set_tagged() '
        'calls, but one was not set.')
  else:
    return value


class TaggedValueCls(Generic[T], config.Config[T]):
  """Placeholder class for TaggedValue instances."""

  @property
  def tags(self):
    return self.__argument_tags__['value']


def TaggedValue(  # pylint: disable=invalid-name
    tags: Collection[TagType],
    default: Union[_NoValue, T] = NO_VALUE,
) -> TaggedValueCls[T]:
  """Declares a value annotated with a set of `Tag`s.

  This is now basically a fdl.Config(lambda value: value) configuration, since
  tags can be set on any field.

  Args:
    tags: Set of tags to apply.
    default: Default value to the identity function.

  Returns:
    Tagged value configuration object.

  Raises:
    ValueError: If `tags` is empty.
  """
  result = TaggedValueCls(tagged_value_identity_fn, value=default)
  if not tags:
    raise ValueError('At least one tag must be provided.')
  for tag in tags:
    config.add_tag(result, 'value', tag)
  return result


def set_tagged(root: config.Buildable, *, tag: TagType, value: Any) -> None:
  """Sets all parameters in `root` tagged with `tag` to `value`.

  Args:
    root: The root of a DAG of Buildables.
    tag: The tag to search for.
    value: Value to set for all parameters tagged with `tag`.
  """

  def traverse(node_value, state: daglish.State):
    if isinstance(node_value, config.Buildable):
      for key, tags in node_value.__argument_tags__.items():
        if any(issubclass(t, tag) for t in tags):
          setattr(node_value, key, value)
    if state.is_traversable(node_value):
      state.flattened_map_children(node_value)

  daglish.MemoizedTraversal.run(traverse, root)


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

  def _inner(value, state: daglish.State):
    if isinstance(value, config.Buildable):
      for node_tags in value.__argument_tags__.values():
        tags.update(node_tags)
    if state.is_traversable(value):
      state.flattened_map_children(value)

  daglish.MemoizedTraversal.run(_inner, root)

  # Add superclasses if desired.
  if add_superclasses:
    for tag in list(tags):
      for base in inspect.getmro(tag):
        if base not in tags and base is not Tag and issubclass(base, Tag):
          tags.add(base)

  return frozenset(tags)


# Any subclass of buildable.
AnyBuildable = TypeVar('AnyBuildable', bound=config.Buildable)


def materialize_tags(
    buildable: AnyBuildable,
    tags: Optional[collections.abc.Set[TagType]] = None) -> AnyBuildable:
  """Materialize tagged fields with assigned values or default values.

  TODO: Consider supporting tags directly on Config objects, e.g.
  by removing those tags.

  Converts:
  ```foo.bar.baz = MyCustomTag.new(4096)```

  Into:
  ```foo.bar.baz = 4096```

  Args:
    buildable: A `fdl.Buildable` to materialize tags in. This will not be
      mutated.
    tags: An optional set of `Tags` to replace. If this is not specified, all
      tagged fields with a value assigned or with a default tag value will be
      materialized. Note, if you would like to exclude a set of tags from being
      materialized, you can pass `tagging.list_tags(buildable) - excluded_tags`
      as the `tag` parameter.

  Returns:
    A new `fdl.Buildable` with its tags replaced by their values.
  """

  def transform(value, state: daglish.State):
    value = state.map_children(value)
    if isinstance(value, TaggedValueCls) and value.value != NO_VALUE and (
        tags is None or set(value.tags) & tags):
      return value.value
    else:
      return value

  return daglish.MemoizedTraversal.run(transform, buildable)
