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

"""Defines a config to check that we can resolve inside submodules."""

import dataclasses

import fiddle as fdl


@dataclasses.dataclass
class Bar:
  b: int


def increment_b(config: fdl.Config[Bar]) -> fdl.Config[Bar]:
  config.b += 1
  return config


def config_bar() -> fdl.Config[Bar]:
  return fdl.Config(Bar, b=1)
