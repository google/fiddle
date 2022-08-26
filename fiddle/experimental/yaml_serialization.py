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

"""Serialization helper that writes YAML output.

Note: This API is highly experimental, and primarily intended for dumping
objects in a medium-easy-to-read format, using indentation/spaces instead of
`printing.py`'s dot-separated paths. Please use `serialization.py` whenever
you need to serialize Fiddle objects in a robust manner.
"""

import collections
import inspect
from typing import Any

from fiddle import config
from fiddle import tagging
from fiddle.experimental import fixture_node
import yaml


def _defaultdict_representer(dumper, data):
  return dumper.represent_dict(data)


yaml.SafeDumper.add_representer(collections.defaultdict,
                                _defaultdict_representer)


def _config_representer(dumper, data, type_name="fdl.Config"):
  """Returns a YAML representation of `data`."""
  value = dict(data.__arguments__)

  # We put __fn_or_cls__ into __arguments__, so "__fn_or_cls__" must be a new
  # key that doesn't exist in __arguments__. It would be pretty rare for this
  # to be an issue.
  if "__fn_or_cls__" in value:
    raise ValueError("It is not supported to dump objects of functions/classes "
                     "that have a __fn_or_cls__ parameter.")

  value["__fn_or_cls__"] = {
      "module": inspect.getmodule(data.__fn_or_cls__).__name__,
      "name": data.__fn_or_cls__.__qualname__,
  }
  return dumper.represent_mapping(f"!{type_name}", value)


def _partial_representer(dumper, data):
  return _config_representer(dumper, data, type_name="fdl.Partial")


def _fixture_representer(dumper, data):
  return _config_representer(
      dumper, data, type_name="fiddle.experimental.Fixture")


def _taggedvalue_representer(dumper, data):
  return dumper.represent_mapping("!fdl.TaggedValue", {
      "tags": [tag.name for tag in data.tags],
      "value": data.value,
  })


yaml.SafeDumper.add_representer(config.Config, _config_representer)
yaml.SafeDumper.add_representer(config.Partial, _partial_representer)
yaml.SafeDumper.add_representer(fixture_node.FixtureNode, _fixture_representer)
yaml.SafeDumper.add_representer(tagging.TaggedValueCls,
                                _taggedvalue_representer)


def dump_yaml(value: Any) -> str:
  """Returns the YAML serialization of `value`.

  Args:
    value: The value to serialize.

  Raises:
    PyrefError: If an error is encountered while serializing a Python reference.
  """
  return yaml.safe_dump(value)
