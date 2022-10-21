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

"""Small example graph with some semantic names for classes.

This creates a "diamond" structure with a chain of descendants.
"""

import dataclasses
from typing import Any

from fiddle.experimental import auto_config


@dataclasses.dataclass
class Person:
  name: str
  child: Any = None
  friend: Any = None
  toy: Any = None


@dataclasses.dataclass
class Toy:
  name: str


@auto_config.auto_config
def build_example():
  sue = Person('Sue', toy=Toy('Robot'))
  bob = Person('Bob', friend=sue)
  joe = Person('Joe', friend=bob)
  nat = Person('Nat', friend=bob)
  mat = Person('Mat', friend=nat, child=joe)
  return mat
