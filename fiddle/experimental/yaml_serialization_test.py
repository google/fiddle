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

"""Tests for yaml_serialization."""
import dataclasses
import pathlib
from typing import Any

from absl.testing import absltest
import fiddle as fdl
from fiddle.experimental import fixture_node
from fiddle.experimental import serialization
from fiddle.experimental import yaml_serialization
from fiddle.testing import test_util
import yaml


def _testdata_dir():
  return (pathlib.Path(absltest.get_default_test_srcdir()) /
          "fiddle/experimental/testdata")


def _config_constructor(loader, node):
  arguments = loader.construct_mapping(node, deep=True)
  fn_or_cls_pyref = arguments.pop("__fn_or_cls__")
  fn_or_cls = serialization.import_symbol(
      serialization.DefaultPyrefPolicy(),
      fn_or_cls_pyref["module"],
      fn_or_cls_pyref["name"],
  )
  return fdl.Config(fn_or_cls, **arguments)


def _partial_constructor(loader, node):
  return fdl.Partial(_config_constructor(loader, node))


def _fixture_constructor(loader, node):
  return fixture_node.FixtureNode(_config_constructor(loader, node))


class SemiSafeLoader(yaml.SafeLoader):
  """Intermediate class that can load Fiddle configs."""


def load_yaml_test_only(serialized: str) -> Any:
  """Test-only method that returns a Fiddle configuration from YAML.

  As mentioned by the yaml_serialization module docstring, YAML serialization
  is primarily provided for debug printing. However, sometimes reversing that
  transformation (loading) can be useful for testing that all values were
  serialized.

  Args:
    serialized: Serialized configuration.

  Returns:
    Loaded object.
  """
  SemiSafeLoader.add_constructor("!fdl.Config", _config_constructor)
  SemiSafeLoader.add_constructor("!fdl.Partial", _partial_constructor)
  SemiSafeLoader.add_constructor("!fiddle.experimental.Fixture",
                                 _fixture_constructor)
  return yaml.load(serialized, Loader=SemiSafeLoader)


@dataclasses.dataclass
class Foo:
  a: int
  b: str
  c: Any


def my_fixture(template):
  return template


class FakeTag(fdl.Tag):
  """Fake/test tag class."""


def _normalize_expected_config(config_str: str):
  return config_str.replace("fiddle.experimental.yaml_serialization_test",
                            "__main__")


class YamlSerializationTest(test_util.TestCase):

  def test_dump_yaml_config(self):
    config = fdl.Config(
        Foo, a=1, b="hi", c=fdl.Config(Foo, a=2, b="bye", c=None))
    serialized = yaml_serialization.dump_yaml(value=config)
    loaded = load_yaml_test_only(serialized)
    self.assertEqual(loaded, config)

  def test_various_python_collections(self):
    # Note: Tuples are currently not maintained by default, which is probably
    # fine for printing, so we haven't added them to this test case.
    config = fdl.Config(
        Foo,
        a=1,
        b="hi",
        c={
            "subconfig": fdl.Config(Foo, a=2, b="bye", c=None),
            "list": [1, 2, [3], None],
        },
    )
    serialized = yaml_serialization.dump_yaml(value=config)
    loaded = load_yaml_test_only(serialized)
    self.assertEqual(loaded, config)

  def test_dump_yaml_partial(self):
    config = fdl.Partial(
        Foo, a=1, b="hi", c=fdl.Config(Foo, a=2, b="bye", c=None))
    serialized = yaml_serialization.dump_yaml(value=config)
    loaded = load_yaml_test_only(serialized)
    self.assertEqual(loaded, config)

  def test_dump_fixture(self):
    config = fixture_node.FixtureNode(my_fixture,
                                      fdl.Config(Foo, a=1, b="hi", c=None))
    serialized = yaml_serialization.dump_yaml(value=config)
    loaded = load_yaml_test_only(serialized)
    self.assertIsInstance(loaded, fixture_node.FixtureNode)
    self.assertEqual(loaded, config)

  def test_dump_tagged_value(self):
    self.assertRegex(
        yaml_serialization.dump_yaml(value=FakeTag.new(2)).strip(),
        r"""!fdl\.TaggedValue
tags:
- [\w\d_\.]+\.FakeTag
value: 2""")

  def test_dump_diamond(self):
    shared = fdl.Config(Foo, a=1, b="shared", c=None)
    config = fdl.Config(Foo, a=2, b="root", c=[shared, shared])
    serialized = yaml_serialization.dump_yaml(value=config)
    with open(_testdata_dir() / "yaml_serialization_diamond.yaml") as f:
      expected = f.read()
    self.assertEqual(
        _normalize_expected_config(serialized),
        _normalize_expected_config(expected))
    loaded = load_yaml_test_only(serialized)
    self.assertDagEqual(config, loaded)


if __name__ == "__main__":
  absltest.main()
