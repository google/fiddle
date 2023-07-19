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

"""Sample configs to use to validate the testing infrastructure."""

import dataclasses
from typing import Any, Tuple
from fiddle._src import config as config_lib
from fiddle._src.experimental import auto_config


@dataclasses.dataclass
class Simple:
  x: Any
  y: Any = 'why?'
  z: Any = None


def base() -> config_lib.Config:
  return config_lib.Config(Simple, x=1)


def simple_fiddler(cfg: config_lib.Config):
  if isinstance(cfg.x, int):
    cfg.x += 1
  if isinstance(cfg.y, str):
    cfg.y *= 3  # Lots of why's!


@auto_config.auto_config
def nested_node_sharing_config():
  shared = Simple('shared-x', 'shared-y')
  f1 = Simple('f1-x', shared)
  f2 = Simple('f2-x', shared)
  f3 = Simple('f3-x', shared)
  a = Simple(f1, f2, f3)
  b = Simple('b-x', 'b-y')
  return Simple(a, b)


@auto_config.auto_config
def linear_nested_config():
  shared = Simple('shared-x', 'shared-y')
  x3 = Simple(shared, 'y3')
  x2 = Simple(x3, 'y2')
  x1 = Simple(x2, 'y1')
  return Simple(x1, 'y')


def get_equal_but_not_object_identical_string_configs() -> (
    Tuple[config_lib.Config, config_lib.Config]
):
  """Generate configs containing equal strings but different object ids."""
  s1 = 'abcd!'
  # Create a string with same content but different object id
  s2 = ''.join(['a', 'b', 'c', 'd', '!'])
  assert s1 == s2
  assert id(s1) != id(s2)

  x = config_lib.Config(Simple, config_lib.Config(Simple, 1, s1), s1)
  y = config_lib.Config(Simple, config_lib.Config(Simple, 1, s1), s2)

  return x, y


def get_equal_but_different_order_dict_configs() -> (
    Tuple[config_lib.Config, config_lib.Config]
):
  """Generate configs containing equal dict but different orders."""
  d1 = {'a': 1, 'b': 2}
  d2 = {'b': 2}
  d2['a'] = 1
  assert list(d1.keys()) != list(d2.keys())

  x = config_lib.Config(Simple, d1)
  y = config_lib.Config(Simple, d2)

  return x, y
