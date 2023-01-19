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

"""A module using `__future__.annotations` to test autofill."""

from __future__ import annotations

import dataclasses

from fiddle.experimental import autofill
from typing_extensions import Annotated


@dataclasses.dataclass
class TopLevel:
  child: Annotated[Child, autofill.Autofill]
  explicit: Annotated[ExplicitInit, autofill.Autofill]


class ExplicitInit:

  def __init__(self, child: Annotated[Child, autofill.Autofill]):
    self.child = child


@dataclasses.dataclass
class Child:
  a: int
  b: str = 'hello'
