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

"""Tests for serialization."""

import pickle
import textwrap
from typing import Any, NamedTuple

from absl.testing import absltest
from absl.testing import parameterized

import fiddle as fdl
from fiddle.experimental import serialization


class Unserializable:

  def __getstate__(self):
    raise NotImplementedError()


def identity_fn(arg):
  return arg


class TestTag(fdl.Tag):
  """`fdl.Tag` that can be used for testing purposes."""


class TestNamedTuple(NamedTuple):
  foo: int
  bar: str


class PickleSerializationTest(absltest.TestCase):

  def test_pickling_non_serializable_history_deepcopy_default(self):
    cfg = fdl.Config(identity_fn, arg=Unserializable())
    cfg.arg = 4
    serialization.clear_argument_history(cfg)
    with self.assertRaises(NotImplementedError):
      pickle.dumps(cfg)

  def test_pickling_non_serializable_history(self):
    cfg = fdl.Config(identity_fn, arg=Unserializable())
    cfg.arg = 4
    cfg = serialization.clear_argument_history(cfg)
    pickle.dumps(cfg)

  def test_pickling_non_serializable_history_mutation(self):
    cfg = fdl.Config(identity_fn, arg=Unserializable())
    cfg.arg = 4
    serialization.clear_argument_history(cfg, deepcopy=False)
    pickle.dumps(cfg)


class DisallowEverythingPyrefPolicy(serialization.PyrefPolicy):

  def allows_import(self, module: str, symbol: str) -> bool:
    return False

  def allows_value(self, value: Any) -> bool:
    return True  # This should never be reached!


class JsonSerializationTest(parameterized.TestCase):

  def serialize_deserialize(self, value):
    return serialization.load_json(serialization.dump_json(value))

  # pyformat: disable
  @parameterized.parameters([
      # Leaf types
      1,  # int
      'str',  # str
      3.12,  # float
      True,  # bool
      None,  # NoneType
      # Basic containers
      ([1, 2, 3, 4],),  # list
      ((1, 2, 3),),  # tuple
      ({'a': 3},),  # dict
      ({'set', 'values'},),  # set
      (frozenset({'frozenset', 'values'}),),  # frozenset
      # Custom serializers
      TestTag.new(1),  # With value set.
      TestTag.new(),  # Without value set.
      b'\x00\xaa\xff',  # A non-UTF-8 byte string.
  ])
  # pyformat: enable
  def test_simple_values(self, value):
    self.assertEqual(value, self.serialize_deserialize(value))

  def test_simple_int_value(self):
    value = 1234
    json = serialization.dump_json(value, indent=4)

    with self.subTest('produces_expected_json'):
      expected_json = textwrap.dedent("""
      {
          "root": {
              "type": "leaf",
              "value": 1234,
              "paths": [
                  "<root>"
              ]
          },
          "objects": {},
          "refcounts": {},
          "version": "0.0.1"
      }
      """).strip('\n')
      self.assertEqual(expected_json, json)

    with self.subTest('deserializes_to_original_value'):
      deserialized_value = serialization.load_json(json)
      self.assertEqual(value, deserialized_value)

  def test_simple_tuple_value(self):
    value = (1234,)
    json = serialization.dump_json(value, indent=4)

    with self.subTest('produces_expected_json'):
      expected_json = textwrap.dedent("""
      {
          "root": {
              "type": "ref",
              "key": "tuple_1"
          },
          "objects": {
              "tuple_1": {
                  "type": {
                      "type": "pyref",
                      "module": "builtins",
                      "name": "tuple"
                  },
                  "items": [
                      [
                          "Index(index=0)",
                          {
                              "type": "leaf",
                              "value": 1234,
                              "paths": [
                                  "<root>[0]"
                              ]
                          }
                      ]
                  ],
                  "metadata": null,
                  "paths": [
                      "<root>"
                  ]
              }
          },
          "refcounts": {
              "tuple_1": 1
          },
          "version": "0.0.1"
      }
      """).strip('\n')
      self.assertEqual(expected_json, json)

    with self.subTest('deserializes_to_original_value'):
      deserialized_value = serialization.load_json(json)
      self.assertEqual(value, deserialized_value)

  def test_simple_config_value(self):
    cfg = fdl.Config(identity_fn, arg=3.14)
    json = serialization.dump_json(cfg, indent=4)

    with self.subTest('produces_expected_json'):
      expected_json = textwrap.dedent("""
      {
          "root": {
              "type": "ref",
              "key": "identity_fn_1"
          },
          "objects": {
              "tuple_1": {
                  "type": {
                      "type": "pyref",
                      "module": "builtins",
                      "name": "tuple"
                  },
                  "items": [
                      [
                          "Index(index=0)",
                          "arg"
                      ]
                  ],
                  "metadata": null
              },
              "buildable_traverser_metadata_1": {
                  "type": {
                      "type": "pyref",
                      "module": "fiddle.config",
                      "name": "BuildableTraverserMetadata"
                  },
                  "items": [
                      [
                          "Attr(name='fn_or_cls')",
                          {
                              "type": "pyref",
                              "module": "__main__",
                              "name": "identity_fn"
                          }
                      ],
                      [
                          "Attr(name='argument_names')",
                          {
                              "type": "ref",
                              "key": "tuple_1"
                          }
                      ]
                  ],
                  "metadata": {
                      "type": "pyref",
                      "module": "fiddle.config",
                      "name": "BuildableTraverserMetadata"
                  }
              },
              "identity_fn_1": {
                  "type": {
                      "type": "pyref",
                      "module": "fiddle.config",
                      "name": "Config"
                  },
                  "items": [
                      [
                          "Attr(name='arg')",
                          {
                              "type": "leaf",
                              "value": 3.14,
                              "paths": [
                                  "<root>.arg"
                              ]
                          }
                      ]
                  ],
                  "metadata": {
                      "type": "ref",
                      "key": "buildable_traverser_metadata_1"
                  },
                  "paths": [
                      "<root>"
                  ]
              }
          },
          "refcounts": {
              "tuple_1": 1,
              "buildable_traverser_metadata_1": 1,
              "identity_fn_1": 1
          },
          "version": "0.0.1"
      }
      """).strip('\n')
      self.assertEqual(expected_json, json)

    with self.subTest('deserializes_to_original_value'):
      deserialized_cfg = serialization.load_json(json)
      self.assertEqual(cfg, deserialized_cfg)

  def test_simple_shared_value(self):
    shared_value = []
    value = [shared_value, shared_value]
    json = serialization.dump_json(value, indent=4)

    with self.subTest('produces_expected_json'):
      expected_json = textwrap.dedent("""
      {
          "root": {
              "type": "ref",
              "key": "list_2"
          },
          "objects": {
              "list_1": {
                  "type": {
                      "type": "pyref",
                      "module": "builtins",
                      "name": "list"
                  },
                  "items": [],
                  "metadata": null,
                  "paths": [
                      "<root>[0]",
                      "<root>[1]"
                  ]
              },
              "list_2": {
                  "type": {
                      "type": "pyref",
                      "module": "builtins",
                      "name": "list"
                  },
                  "items": [
                      [
                          "Index(index=0)",
                          {
                              "type": "ref",
                              "key": "list_1"
                          }
                      ],
                      [
                          "Index(index=1)",
                          {
                              "type": "ref",
                              "key": "list_1"
                          }
                      ]
                  ],
                  "metadata": null,
                  "paths": [
                      "<root>"
                  ]
              }
          },
          "refcounts": {
              "list_1": 2,
              "list_2": 1
          },
          "version": "0.0.1"
      }
      """).strip('\n')
      self.assertEqual(expected_json, json)

    with self.subTest('deserializes_properly'):
      deserialized_value = serialization.load_json(json)
      self.assertIs(deserialized_value[0], deserialized_value[1])

  def test_shared_references_are_preserved(self):
    nested_cfg = fdl.Config(identity_fn, arg=(3, identity_fn))
    cfg = fdl.Config(identity_fn, arg=[1, 2, {'key': nested_cfg}, nested_cfg])
    json = serialization.dump_json(cfg)
    deserialized_cfg = serialization.load_json(json)
    self.assertEqual(cfg, deserialized_cfg)
    self.assertIs(deserialized_cfg.arg[2]['key'], deserialized_cfg.arg[3])

  def test_shared_references_are_preserved_in_metadata(self):
    instance = TestNamedTuple(1, 'one')
    cfg = fdl.Config(
        identity_fn, arg=[{
            instance: instance
        }, {
            instance: instance
        }])
    json = serialization.dump_json(cfg)
    deserialized_cfg = serialization.load_json(json)
    self.assertEqual(cfg, deserialized_cfg)
    deserialized_instance, = deserialized_cfg.arg[0].keys()
    self.assertIsNot(instance, deserialized_instance)
    self.assertIs(deserialized_instance,
                  deserialized_cfg.arg[0][deserialized_instance])
    self.assertIs(deserialized_instance,
                  deserialized_cfg.arg[1][deserialized_instance])

  def test_unserializble_value_error(self):
    with self.assertRaises(serialization.UnserializableValueError):
      serialization.dump_json(fdl.Config(identity_fn, Unserializable()))

  def test_default_pyref_policy_error_during_serialization(self):
    with self.assertRaises(serialization.PyrefPolicyError):
      serialization.dump_json(fdl.Config(identity_fn, range))

  def test_default_pyref_policy_error_during_deserialization(self):
    with self.assertRaises(serialization.PyrefPolicyError):
      json = textwrap.dedent("""
      {
          "root": {
              "type": "pyref",
              "module": "builtins",
              "name": "range"
          },
          "objects": {},
          "refcounts": {},
          "version": "0.0.1"
      }
      """).strip('\n')
      serialization.load_json(json)

  def test_pyref_module_not_found_error_during_deserialization(self):
    with self.assertRaises(serialization.PyrefError) as e:
      json = textwrap.dedent("""
      {
          "root": {
              "type": "pyref",
              "module": "some.invalid.module",
              "name": "foo"
          },
          "objects": {},
          "refcounts": {},
          "version": "0.0.1"
      }
      """).strip('\n')
      serialization.load_json(json)
    self.assertIsInstance(e.exception.__cause__, ModuleNotFoundError)

  def test_pyref_attribute_error_during_deserialization(self):
    with self.assertRaises(serialization.PyrefError) as e:
      json = textwrap.dedent("""
      {
          "root": {
              "type": "pyref",
              "module": "collections",
              "name": "invalid_symbol"
          },
          "objects": {},
          "refcounts": {},
          "version": "0.0.1"
      }
      """).strip('\n')
      serialization.load_json(json)
    self.assertIsInstance(e.exception.__cause__, AttributeError)

  def test_non_default_pyref_policy_error_during_serialization(self):
    pyref_policy = DisallowEverythingPyrefPolicy()
    with self.assertRaises(serialization.PyrefPolicyError):
      serialization.dump_json(fdl.Config(identity_fn), pyref_policy)

  def test_non_default_pyref_policy_error_during_deserialization(self):
    serialized_cfg = serialization.dump_json(fdl.Config(identity_fn))
    pyref_policy = DisallowEverythingPyrefPolicy()
    with self.assertRaises(serialization.PyrefPolicyError):
      serialization.load_json(serialized_cfg, pyref_policy)


if __name__ == '__main__':
  absltest.main()
