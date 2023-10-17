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
from typing import Any, Collection, FrozenSet, Optional, TypeVar, Union

from fiddle._src import config
from fiddle._src import daglish
from fiddle._src import tag_type
from fiddle._src.experimental import auto_config

TagType = tag_type.TagType
TaggedValueNotFilledError = tag_type.TaggedValueNotFilledError

NO_VALUE = config.NO_VALUE
NoValue = config.NoValue
_NoValue = config.NoValue
tagged_value_fn = config.tagged_value_fn
TaggedValueCls = config.TaggedValueCls


class Tag(metaclass=TagType):
  """Metadata associated with a Fiddle configurable value.

  To see a usage example, please see the documentation on the tagging module.

  Note: Tags cannot be instantiated.
  """

  @classmethod
  def new(cls, default: Any = NO_VALUE) -> Any:
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


def TaggedValue(  # pylint: disable=invalid-name
    tags: Collection[TagType],
    default: Union[NoValue, T] = NO_VALUE,
) -> TaggedValueCls[T]:
  """Declares a value annotated with a set of ``Tag``'s.

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
  if not tags:
    raise ValueError('At least one tag must be provided.')
  result = TaggedValueCls(tagged_value_fn)
  if default is not NO_VALUE:
    result.value = default
  for tag in tags:
    add_tag(result, 'value', tag)
  return result


def set_tagged(root: config.Buildable, *, tag: TagType, value: Any) -> None:
  """Sets all parameters in `root` tagged with `tag` to `value`.

  Args:
    root: The root of a DAG of Buildables.
    tag: The tag to search for.
    value: Value to set for all parameters tagged with `tag`.
  """

  for node, _ in daglish.iterate(root):
    if isinstance(node, config.Buildable):
      for key, tags in node.__argument_tags__.items():
        if any(issubclass(t, tag) for t in tags):
          setattr(node, key, value)


def list_tags(
    root: config.Buildable,
    add_superclasses=False,
) -> FrozenSet[TagType]:
  """Lists all tags in a buildable.

  Args:
    root: The root of a DAG of ``Buildable``'s.
    add_superclasses: For tags that inherit from other tags, add the
      superclasses as well.

  Returns:
    Set of tags used in this buildable.
  """
  tags = set()

  for value, _ in daglish.iterate(root):
    if isinstance(value, config.Buildable):
      for node_tags in value.__argument_tags__.values():
        tags.update(node_tags)

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
    tags: Optional[collections.abc.Set[TagType]] = None,
    clear_field_tags: bool = False,
) -> AnyBuildable:
  """Materialize tagged fields with assigned values or default values.

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
    clear_field_tags: Whether to remove all field tags.

  Returns:
    A new `fdl.Buildable` with its tags replaced by their values.
  """

  def transform(value, state: daglish.State):
    value = state.map_children(value)
    if isinstance(value, TaggedValueCls) and value.value != NO_VALUE and (
        tags is None or set(value.tags) & tags):
      return value.value
    elif isinstance(value, config.Buildable):
      if tags:
        for tag_set in value.__argument_tags__.values():
          for tag in tags:
            tag_set.discard(tag)
      if clear_field_tags:
        value.__argument_tags__.clear()
      return value
    else:
      return value

  return daglish.MemoizedTraversal.run(transform, buildable)


def _validate_param_index(buildable: config.Buildable, index: int) -> None:
  """Validate param index for tagging APIs."""
  if index < 0:
    raise IndexError(
        'Cannot use negative index to specify positional arguments for tagging.'
    )
  # If *args exists, any non-negative index is valid because setting tags for
  # arguments that are not set is allowed.
  if buildable.__signature_info__.var_positional_start is not None:
    return
  params = list(buildable.__signature_info__.parameters.values())
  err_msg = f'Index {index} is out of range.'
  if index >= len(params):
    raise IndexError(err_msg)
  param = params[index]
  if param.kind not in (param.POSITIONAL_ONLY, param.POSITIONAL_OR_KEYWORD):
    raise IndexError(err_msg)


def _validate_argument_name(
    buildable: config.Buildable, argument: Union[str, int]
) -> None:
  """Validate argument name for tagging APIs."""
  if isinstance(argument, str):
    buildable.__signature_info__.validate_param_name(
        argument, buildable.__fn_or_cls__
    )
  elif isinstance(argument, int):
    _validate_param_index(buildable, argument)


def add_tag(
    buildable: config.Buildable,
    argument: Union[str, int],
    tag: tag_type.TagType,
) -> None:
  """Tags `name` with `tag` in `buildable`."""
  _validate_argument_name(buildable, argument)
  if isinstance(argument, int):
    argument = buildable.__signature_info__.index_to_key(
        argument, buildable.__arguments__
    )
  buildable.__argument_tags__[argument].add(tag)
  buildable.__argument_history__.add_updated_tags(
      argument, buildable.__argument_tags__[argument]
  )


def set_tags(
    buildable: config.Buildable,
    argument: Union[str, int],
    tags: Collection[tag_type.TagType],
) -> None:
  """Sets tags for a parameter in `buildable`, overriding existing tags."""
  # Skip the validation here as this API depends on other tagging APIs.
  clear_tags(buildable, argument)
  for tag in tags:
    add_tag(buildable, argument, tag)
  buildable.__argument_history__.add_updated_tags(
      argument, buildable.__argument_tags__[argument]
  )


def remove_tag(
    buildable: config.Buildable,
    argument: Union[str, int],
    tag: tag_type.TagType,
) -> None:
  """Removes a given tag from a named argument of a Buildable."""
  _validate_argument_name(buildable, argument)
  if isinstance(argument, int):
    argument = buildable.__signature_info__.index_to_key(
        argument, buildable.__arguments__
    )

  field_tag_set = buildable.__argument_tags__[argument]
  if tag not in field_tag_set:
    raise ValueError(
        f'{tag} not set on {argument}; current tags: {field_tag_set}.'
    )
  field_tag_set.remove(tag)
  buildable.__argument_history__.add_updated_tags(
      argument, buildable.__argument_tags__[argument]
  )


def clear_tags(buildable: config.Buildable, argument: Union[str, int]) -> None:
  """Removes all tags from a named argument of a Buildable."""
  _validate_argument_name(buildable, argument)
  if isinstance(argument, int):
    argument = buildable.__signature_info__.index_to_key(
        argument, buildable.__arguments__
    )
  buildable.__argument_tags__[argument].clear()
  buildable.__argument_history__.add_updated_tags(
      argument, buildable.__argument_tags__[argument]
  )


def get_tags(
    buildable: config.Buildable,
    argument: Union[str, int],
) -> FrozenSet[tag_type.TagType]:
  _validate_argument_name(buildable, argument)
  if isinstance(argument, int):
    argument = buildable.__signature_info__.index_to_key(
        argument, buildable.__arguments__
    )
  return frozenset(buildable.__argument_tags__[argument])
