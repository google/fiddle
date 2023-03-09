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

"""A test module used in `module_reflection_test`."""

import fiddle as fdl


# Note: normally, this function should be defined in a different module!
def function_to_configure(x: int, y: str):
  return x, y


def simple_base():
  return fdl.Config(function_to_configure, x=2)


def alternate_base():
  return fdl.Config(function_to_configure, x=0)


def base_with_defaults(x: int = 5, negative: bool = False):
  if negative:
    x = -x
  return fdl.Config(function_to_configure, x=x)


def arbitrary_other_function(x, y):  # pylint: disable=unused-argument
  pass


def fiddler1(cfg: fdl.Config):  # pylint: disable=unused-argument
  pass


def fiddler2(cfg: fdl.Config):  # pylint: disable=unused-argument
  pass


def another_fiddler(cfg: fdl.Config, defaulted_arg=3):  # pylint: disable=unused-argument
  pass
