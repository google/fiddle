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
import fiddle as fdl


@dataclasses.dataclass
class Simple:
  x: int
  y: str = 'why?'


def base() -> fdl.Config:
  return fdl.Config(Simple, x=1)


def simple_fiddler(cfg: fdl.Config):
  cfg.x += 1
  cfg.y *= 3  # Lots of why's!
