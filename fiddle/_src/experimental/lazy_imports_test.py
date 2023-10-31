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

"""Tests for history."""

import dataclasses
import inspect
from typing import Any
import unittest

from absl.testing import absltest
from fiddle._src import building
from fiddle._src import config as config_lib
from fiddle._src import partial
from fiddle._src.experimental import lazy_imports
from fiddle._src.experimental import serialization


@dataclasses.dataclass
class SampleSumClass:
  arg1: Any
  arg2: Any

  def sum(self) -> Any:
    a = self.arg1.sum() if hasattr(self.arg1, 'sum') else self.arg1
    b = self.arg2.sum() if hasattr(self.arg2, 'sum') else self.arg2
    return a + b


class LazyImportsInspectTest(absltest.TestCase, unittest.TestCase):

  def assert_module(self, m: lazy_imports.ProxyObject, name: str) -> None:
    self.assertIsInstance(m, lazy_imports.ProxyObject)
    self.assertEqual(m.qualname, name)

  def assert_signature(self, m: lazy_imports.ProxyObject, kw_only=True) -> None:
    params = inspect.signature(m).parameters
    self.assertIn('kwargs', params)
    if kw_only:
      self.assertLen(params, 1)
    else:
      self.assertLen(params, 2)
      self.assertIn('args', params)

  def test_proxy_objects(self):
    # TODO(b/291129675): Add tests for `kw_only=False`` after supporting args.
    with lazy_imports.lazy_imports(kw_only=True):
      # pylint: disable=g-import-not-at-top,g-multiple-import
      # pytype: disable=import-error
      import a0
      import a1.b.c
      import a2.b.c as c00
      import a2.b.c as c01  # Importing object twice is the same instance
      from a2.b import c as c02

      from a3 import c2, c3
      from a3.b.c import c4
      # pytype: enable=import-error
      # pylint: enable=g-import-not-at-top,g-multiple-import

    with self.subTest('qualname'):
      self.assert_module(a0, 'a0')
      self.assert_module(a1.b.c, 'a1.b.c')
      self.assert_module(a1.non_module.c, 'a1:non_module.c')
      self.assert_module(c00, 'a2.b.c')
      self.assert_module(c01, 'a2.b.c')
      self.assert_module(c02, 'a2.b.c')
      self.assert_module(c02.non_module.c, 'a2.b.c:non_module.c')
      self.assert_module(c2, 'a3.c2')
      self.assert_module(c3, 'a3.c3')
      self.assert_module(c3.non_module.c, 'a3.c3:non_module.c')
      self.assert_module(c4, 'a3.b.c.c4')
      self.assert_module(c4.non_module.c, 'a3.b.c.c4:non_module.c')

    with self.subTest('same_instance'):
      self.assertIs(c01, c00)
      self.assertIs(c02, c00)

    with self.subTest('inspect_signature'):
      self.assert_signature(a0, kw_only=True)
      self.assert_signature(a1.b.c, kw_only=True)
      self.assert_signature(c00, kw_only=True)
      self.assert_signature(c01, kw_only=True)
      self.assert_signature(c02, kw_only=True)

    with self.subTest('inspect_getmodule'):
      self.assertEqual(
          inspect.getmodule(a0).__name__,
          'fiddle._src.experimental.lazy_imports',
      )
      self.assertEqual(
          inspect.getmodule(a1.b.c).__name__,
          'fiddle._src.experimental.lazy_imports',
      )
      self.assertEqual(
          inspect.getmodule(c00).__name__,
          'fiddle._src.experimental.lazy_imports',
      )
      self.assertEqual(
          inspect.getmodule(c01).__name__,
          'fiddle._src.experimental.lazy_imports',
      )
      self.assertEqual(
          inspect.getmodule(c02).__name__,
          'fiddle._src.experimental.lazy_imports',
      )


class BuildLazyImportsTest(absltest.TestCase, unittest.TestCase):

  def test_import_as(self):
    with lazy_imports.lazy_imports(kw_only=True):
      from fiddle._src.experimental import lazy_imports_test_example as example  # pylint: disable=g-import-not-at-top
      from fiddle._src.experimental.lazy_imports_test_example import my_function as my_fn  # pylint: disable=g-import-not-at-top, g-importing-member

    with self.subTest('dataclass'):
      cfg = config_lib.Config(example.MyDataClass, x=1, y=2)
      obj = building.build(cfg)
      self.assertEqual(obj.sum(), 3)

    with self.subTest('function'):
      cfg = config_lib.Config(my_fn, x=5)
      obj = building.build(cfg)
      self.assertEqual(obj, 128)

  def test_wrong_imports_raise_errors(self):
    with lazy_imports.lazy_imports(kw_only=True):
      from fiddle._src.experimental import not_exist_file  # pylint: disable=g-import-not-at-top # pytype: disable=import-error

    cfg = config_lib.Config(not_exist_file.DummyClass, x=2, y=3)
    with self.assertRaises(ModuleNotFoundError):
      _ = building.build(cfg)

  def test_config_mutation(self):
    with lazy_imports.lazy_imports(kw_only=True):
      from fiddle._src.experimental import lazy_imports_test_example  # pylint: disable=g-import-not-at-top

    cfg = config_lib.Config(lazy_imports_test_example.MyDataClass, x=1, y=2)
    cfg.x = 100
    cfg.y = 20
    obj = building.build(cfg)
    self.assertEqual(obj.sum(), 120)

  def test_nested_attribute(self):
    with lazy_imports.lazy_imports(kw_only=True):
      from fiddle._src.experimental import lazy_imports_test_example  # pylint: disable=g-import-not-at-top, g-importing-member
    cls_or_fn = lazy_imports_test_example.MyDataClass.create
    cfg = config_lib.Config(cls_or_fn, a=10, b=2)
    obj = building.build(cfg)
    self.assertEqual(obj.sum(), 12)

  def test_nested_config(self):
    with lazy_imports.lazy_imports(kw_only=True):
      from fiddle._src.experimental import lazy_imports_test_example  # pylint: disable=g-import-not-at-top

    x = config_lib.Config(lazy_imports_test_example.MyDataClass, x=1, y=2)
    y = config_lib.Config(lazy_imports_test_example.MyDataClass, x=20, y=30)
    xy = config_lib.Config(SampleSumClass, x, y)
    z = config_lib.Config(SampleSumClass, arg1=300, arg2=400)
    cfg = config_lib.Config(SampleSumClass, arg1=xy, arg2=z)

    foo = building.build(cfg)
    self.assertEqual(foo.sum(), 753)

  def test_partial(self):
    with lazy_imports.lazy_imports(kw_only=True):
      from fiddle._src.experimental import lazy_imports_test_example  # pylint: disable=g-import-not-at-top
    cfg = partial.Partial(lazy_imports_test_example.my_function)
    obj = building.build(cfg)
    # Note that the lazy imported module is not loaded at build time but delayed
    # until call time when working with fdl.Partial.
    self.assertIsInstance(obj.func, lazy_imports.ProxyObject)  # pytype: disable=attribute-error
    self.assertEqual(obj(x=5), 128)

  def test_positional_args(self):
    with lazy_imports.lazy_imports(kw_only=False):
      from fiddle._src.experimental import lazy_imports_test_example  # pylint: disable=g-import-not-at-top
    cfg = config_lib.Config(lazy_imports_test_example.MyDataClass, 1, 2)
    obj = building.build(cfg)
    self.assertEqual(obj.sum(), 3)

  def test_partial_w_positional_args(self):
    with lazy_imports.lazy_imports(kw_only=False):
      from fiddle._src.experimental import lazy_imports_test_example  # pylint: disable=g-import-not-at-top
    cfg = partial.Partial(lazy_imports_test_example.my_function, 5)
    obj = building.build(cfg)
    self.assertIsInstance(obj.func, lazy_imports.ProxyObject)  # pytype: disable=attribute-error
    self.assertEqual(obj(2), 130)

  def test_kw_only_check(self):
    with lazy_imports.lazy_imports(kw_only=True):
      from fiddle._src.experimental import lazy_imports_test_example  # pylint: disable=g-import-not-at-top
    with self.assertRaisesRegex(TypeError, 'too many positional arguments'):
      _ = config_lib.Config(lazy_imports_test_example.MyDataClass, 1, 2)


class SerializationTest(absltest.TestCase, unittest.TestCase):

  def test_regular_import(self):
    from fiddle._src.experimental import lazy_imports_test_example  # pylint: disable=g-import-not-at-top

    x = config_lib.Config(lazy_imports_test_example.MyDataClass, x=1, y=2)
    original_cfg = config_lib.Config(SampleSumClass, arg1=300, arg2=x)
    del lazy_imports_test_example

    with lazy_imports.lazy_imports(kw_only=True):
      from fiddle._src.experimental import lazy_imports_test_example  # pylint: disable=g-import-not-at-top, reimported

    x = config_lib.Config(lazy_imports_test_example.MyDataClass, x=1, y=2)
    lazy_cfg = config_lib.Config(SampleSumClass, arg1=300, arg2=x)

    original_serialized = serialization.dump_json(original_cfg)
    lazy_serialized = serialization.dump_json(lazy_cfg)
    self.assertEqual(original_serialized, lazy_serialized)

  def test_import_as(self):
    from fiddle._src.experimental import lazy_imports_test_example as test_example  # pylint: disable=g-import-not-at-top

    x = config_lib.Config(test_example.MyDataClass, x=1, y=2)
    original_cfg = config_lib.Config(SampleSumClass, arg1=300, arg2=x)
    del test_example

    with lazy_imports.lazy_imports(kw_only=True):
      from fiddle._src.experimental import lazy_imports_test_example as test_example  # pylint: disable=g-import-not-at-top, reimported

    x = config_lib.Config(test_example.MyDataClass, x=1, y=2)
    lazy_cfg = config_lib.Config(SampleSumClass, arg1=300, arg2=x)

    original_serialized = serialization.dump_json(original_cfg)
    lazy_serialized = serialization.dump_json(lazy_cfg)
    self.assertEqual(original_serialized, lazy_serialized)


if __name__ == '__main__':
  unittest.main()
