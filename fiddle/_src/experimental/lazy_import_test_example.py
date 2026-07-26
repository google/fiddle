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

"""Examples used by lazy_import_test.py."""

import dataclasses
from typing import List


def my_function(x: int, y: float = 1.0) -> List[float]:
  return [y] * x


class MyClass:

  def __init__(self, name: str):
    self.name = name

  def reversed_name(self):
    return self.name[::-1]

  def __repr__(self):
    return f'MyClass({self.name!r})'


@dataclasses.dataclass
class MyDataclass:
  a: int
  b: float


def another_function(a: MyClass, b: List[MyDataclass], **kwargs):
  """Test function with a kwargs and annotations involving types."""
  return (a, b, kwargs)
