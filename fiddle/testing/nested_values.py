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

"""Provides utilities to generate and test nested values with Fiddle objects."""
import random
from typing import Any, List, Union

from fiddle import config
from fiddle import daglish
from fiddle import tagging


def kwargs_fn(**kwargs):
  return kwargs


class SampleTag(tagging.Tag):
  """`fdl.Tag` that can be used for testing purposes."""


def calculate_nested_value_depth(value):
  traverser = daglish.find_node_traverser(type(value))
  children, _ = traverser.flatten(value) if traverser else ([], None)
  depths = [calculate_nested_value_depth(child) for child in children]
  return 1 + max(depths) if depths else 0


def generate_nested_value(
    rng: random.Random,
    max_depth: int = 8,
    max_container_size: int = 5,
    share_objects: Union[bool, List[Any]] = True,
):
  """Generates a (possibly) nested value that (may) contain `fdl.Buildable`s.

  The resulting value can contain: integers, floats, booleans, `None`, strings,
  lists, tuples, dicts, and `fdl.Buildable`s (`fdl.Config`s and `fdl.Partial`s)
  that may have tagged arguments.

  Args:
    rng: The RNG (`random.Random`) instance to use for generating random values.
    max_depth: A guideline controlling the maximum nesting depth of the
      resulting value. The actual nesting depth may be less than this, or if
      `share_objects` is `True`, it may be larger.
    max_container_size: The maximum size of any container in the resulting
      nested_value.
    share_objects: Whether to allow aliasing of generated objects. This can be a
      `bool`, or a list of existing objects to share. If provided as a list, the
      list will be mutated to include aliasable values that are generated during
      nested value creation. If `False`, no aliasing will take place. Note that
      allowing aliasing may result in the final nested value having nesting
      depth greater than `max_depth`.

  Returns:
    A randomly generated nested value.
  """
  if isinstance(share_objects, bool) and share_objects:
    share_objects = []

  def generate_value():
    return generate_nested_value(
        rng,
        max_depth=max_depth - 1,
        max_container_size=max_container_size,
        share_objects=share_objects)

  def generate_int():
    return rng.randint(-10000, 10000)

  def generate_float():
    return rng.random() * 10000 - 5000

  def generate_bool():
    return rng.random() > 0.5

  def generate_none():
    return None

  def generate_string():
    length = rng.randint(1, 10)
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
    return ''.join([rng.choice(alphabet) for _ in range(length)])

  def generate_list():
    length = rng.randint(0, max_container_size)
    return [generate_value() for _ in range(length)]

  def generate_tuple():
    return tuple(generate_list())

  def generate_dict(key_generator=None):
    length = rng.randint(0, max_container_size)
    key_generator = key_generator or rng.choice(
        [generate_int, generate_float, generate_string])
    return {key_generator(): generate_value() for _ in range(length)}

  def generate_buildable():
    buildable_type = rng.choice([config.Config, config.Partial])
    arguments = generate_dict(key_generator=generate_string)
    buildable = buildable_type(kwargs_fn, **arguments)
    if generate_bool() and arguments:
      argument = rng.choice(list(arguments))
      config.set_tags(buildable, argument, {SampleTag})
    return buildable

  def generate_alias():
    for value in enumerate(share_objects):
      if calculate_nested_value_depth(value) < max_depth:
        return value
    return generate_value()

  def generate_leaf():
    generator = rng.choice([
        generate_string,
        generate_int,
        generate_float,
        generate_bool,
        generate_none,
    ])
    return generator()

  def generate_collection():
    generator = rng.choice([
        generate_list,
        generate_tuple,
        generate_dict,
        generate_buildable,
    ])
    return generator()

  generators = [generate_leaf, generate_collection, generate_alias]
  # Weights here are chosen somewhat arbitrarily to limit the rate of aliasing
  # and to yield more collections near the root of the nested value, with no
  # collections (or aliases) once the maximum depth has been reached.
  weights = [3, max_depth, 1 if share_objects and max_depth else 0]

  generator = rng.choices(generators, weights=weights)[0]
  value = generator()
  if (isinstance(share_objects, list) and daglish.is_memoizable(value) and
      generator is not generate_alias):
    share_objects.append(value)
  return value
