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

"""Library for expressing placeholders.

When defining shared parameters across a project that later could be changed,
for example dtype or activation function, we encourage the following coding
pattern: a tiny shared library file should declare placeholder keys for an
entire project, like

  activation_dtype = fdl.PlaceholderKey("activation_dtype")
  kernel_init_fn = fdl.PlaceholderKey("kernel_init_fn")

Then, in library code which configures shared Fiddle fixtures, these keys are
used,

  def layer_norm_fixture() -> fdl.Config[LayerNorm]:
    cfg = fdl.Config(LayerNorm)
    cfg.dtype = fdl.Placeholder(activation_dtype, jnp.float32)

And in experiment code stitching everything together, all of these placeholders
can be set at once,

  def encoder_decoder_fixture() -> fdl.Config[EncoderDecoder]:
    cfg = fdl.Config(EncoderDecoder)
    ...
    cfg.encoder.encoder_norm = layer_norm_fixture()

    # Set all activation dtypes.
    fdl.set_placeholder(cfg, activation_dtype, jnp.bfloat16)
    return cfg

  model = fdl.build(encoder_decoder_fixture())
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Any, Generic, TypeVar, Union

from fiddle import config
import tree


class _NoValue:
  """Sentinel class (used in place of object for more precise errors)."""

  def __deepcopy__(self, memo):
    """Override for deepcopy that does not copy this sentinel object."""
    del memo
    return self


NO_VALUE = _NoValue()


class PlaceholderNotFilledError(ValueError):
  """A placeholder was not filled when build() was called."""


@dataclasses.dataclass(frozen=True)
class PlaceholderKey:
  """Represents a key identifying a type of placeholder.

  Instances of this class are typically compared on object identity, so please
  share them via a common library file (see module docstring).
  """
  name: str

  def __deepcopy__(self, memo):
    """Override for deepcopy that does not copy this immutable object.

    This is important because in `set_placeholder` we use object identity on
    keys, so we should not unnecessarily copy them.

    Args:
      memo: Unused memoization object from deepcopy API.

    Returns:
      `self`.
    """
    del memo
    return self


def placeholder_fn(key: PlaceholderKey, value: Any = NO_VALUE) -> Any:
  if value is NO_VALUE:
    raise PlaceholderNotFilledError(
        "Expected all placeholders to be replaced via fdl.set_placeholder() "
        f"calls, but one with name {key.name!r} was not set.")
  else:
    return value


T = TypeVar("T")


class Placeholder(Generic[T], config.Config[T]):
  """Declares a placeholder in a configuration."""

  def __init__(self,
               key: Union[PlaceholderKey, Placeholder[T]],
               default: Union[_NoValue, T] = NO_VALUE):
    """Initializes the placeholder.

    Args:
      key: Normally a key identifying the type of placeholder, so that all
        placeholders of a given type can be set at once. Alternately an existing
        Placeholder can be specified, and it will be copied.
      default: Default value of the placeholder. This is normally a sentinel
        which will cause the configuration to fail to build when the
        placeholders are not set.
    """
    if isinstance(key, Placeholder):
      assert default is NO_VALUE
      super().__init__(key)
    else:
      if not isinstance(key, PlaceholderKey):
        raise TypeError(f"Expected key to be a PlaceholderKey, got {key}")
      super().__init__(placeholder_fn, key=key, value=default)

  def __deepcopy__(self, memo) -> config.Buildable[T]:
    """Implements the deepcopy API."""
    return Placeholder(self.key, copy.deepcopy(self.value, memo))


def set_placeholder(cfg: config.Buildable, key: PlaceholderKey,
                    value: Any) -> None:
  """Replaces placeholders of a given key with a given value.

  Specifically, placeholders are config.Config sub-nodes with keys and values.
  Calling this method will preserve the placeholder nodes, in case experiment
  configs would like to override placeholder values from another config/fixture.

  This implementation does not currently track nodes with shared parents, so for
  some pathological Buildable DAGs, e.g. lattices, it could take exponential
  time. We think these are edge cases, and could be easily fixed later.

  Args:
    cfg: A tree of buildable elements. This tree is mutated.
    key: Key identifying which placeholders' values should be set.
    value: Value to set for these placeholders.
  """
  if isinstance(cfg, Placeholder):
    if key is cfg.key:
      cfg.value = value

  def map_fn(leaf):
    if isinstance(leaf, config.Buildable):
      set_placeholder(leaf, key, value)
    return leaf

  tree.map_structure(map_fn, cfg.__arguments__)
