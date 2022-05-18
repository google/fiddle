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

"""Module used by flags_test to test automatic imports.."""

import types

from fiddle.experimental import dict_config

base1 = dict_config.DictConfig
base2 = lambda: dict_config.DictConfig(x=1, y='2')


def a_fiddler(cfg):
  cfg.z = int(cfg.y)


def b_fiddler(cfg):
  cfg.b = 9


namespace = types.SimpleNamespace()
namespace.base3 = lambda: dict_config.DictConfig(x=1, y=2.0)
namespace.b_fiddler = b_fiddler
