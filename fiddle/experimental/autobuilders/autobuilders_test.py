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

"""Tests for autobuilders."""

import dataclasses

from absl.testing import absltest
import fiddle as fdl

from fiddle.experimental.autobuilders import autobuilders as ab
from fiddle.experimental.autobuilders import autobuilders_test_util


@dataclasses.dataclass(frozen=True)
class FakeDense:
  in_dim: int
  out_dim: int


@dataclasses.dataclass(frozen=True)
class FakeDenseSubclass(FakeDense):
  pass


@dataclasses.dataclass(frozen=True)
class FakeMlp:
  first_dense: FakeDense
  second_dense: FakeDense


class RegistryTest(absltest.TestCase):

  def setUp(self):
    autobuilders_test_util.clear_registry()
    super().setUp()

  def test_skeleton_and_get_config(self):

    class Foo:

      def __init__(self, x):
        self.x = x

    @ab.skeleton(Foo)
    def foo_skeleton(config: fdl.Config):  # pylint: disable=unused-variable
      # Note: Setting constants is generally not the purpose of skeletons; we're
      # just doing that here for testing purposes.
      config.x = 3

    foo_builder = ab.config(Foo)
    self.assertIsInstance(foo_builder, fdl.Config)
    foo = fdl.build(foo_builder)
    self.assertIsInstance(foo, Foo)
    self.assertEqual(foo.x, 3)

  def test_get_config_not_present(self):

    class Foo:
      pass

    with self.assertRaisesRegex(KeyError, r".*\bFoo\b"):
      ab.Registry().config(Foo)
    with self.assertRaisesRegex(KeyError, r".*\bFoo\b"):
      ab.config(Foo)

    # Still raises informative error even when table entry is present
    # (for a validator).
    ab.validator(Foo)(lambda config: None)
    with self.assertRaisesRegex(KeyError, r".*\bFoo\b"):
      ab.config(Foo)

  def test_skeleton_registers(self):
    registry = ab.Registry()
    fn = lambda config: None
    registry.skeleton(FakeDense)(fn)
    self.assertDictEqual(registry.table, {
        FakeDense: ab.TableEntry(skeleton_fn=fn, validators=[]),
    })

  def test_skeleton_duplicate_class_registration_error(self):
    registry = ab.Registry()
    registry.skeleton(FakeDense)(lambda config: None)
    with self.assertRaisesRegex(ab.DuplicateSkeletonError, r".*FakeDense.*"):
      registry.skeleton(FakeDense)(lambda config: None)

  def test_skeleton_duplicate_function_registration_error(self):
    # Because the fancy error message includes source lines, make sure that
    # logic doesn't crash with functions too.
    def fake_fn():
      pass

    registry = ab.Registry()
    registry.skeleton(fake_fn)(lambda config: None)
    with self.assertRaisesRegex(ab.DuplicateSkeletonError, r".*fake_fn.*"):
      registry.skeleton(fake_fn)(lambda config: None)

  def test_recursive_skeleton(self):

    @ab.skeleton(FakeDense)
    def dense_skeleton(config: fdl.Config) -> None:  # pylint: disable=unused-variable
      config.in_dim = 4
      config.out_dim = 4

    @ab.skeleton(FakeMlp)
    def mlp_skeleton(config: fdl.Config) -> None:  # pylint: disable=unused-variable
      config.first_dense = ab.config(FakeDense)
      config.first_dense.in_dim = 5
      config.second_dense = ab.config(FakeDense)

    fake_mlp = fdl.build(ab.config(FakeMlp))
    self.assertEqual(
        fake_mlp,
        FakeMlp(
            first_dense=FakeDense(in_dim=5, out_dim=4),
            second_dense=FakeDense(in_dim=4, out_dim=4),
        ))

  def test_auto_skeleton_basic(self):
    ab.auto_skeleton(FakeMlp)
    self.assertEqual(
        ab.config(FakeMlp),
        fdl.Config(FakeMlp, fdl.Config(FakeDense), fdl.Config(FakeDense)))

  def test_auto_skeleton_subclasses_and_existing_skeletons(self):

    @ab.skeleton(FakeDense)
    def dense_skeleton(config: fdl.Config) -> None:  # pylint: disable=unused-variable
      config.in_dim = 4
      config.out_dim = 4

    @ab.skeleton(FakeDenseSubclass)
    def dense_subclass_skeleton(config: fdl.Config) -> None:  # pylint: disable=unused-variable
      config.in_dim = 7
      config.out_dim = 9

    ab.auto_skeleton(FakeMlp, second_dense=FakeDenseSubclass)
    fake_mlp = fdl.build(ab.config(FakeMlp))
    self.assertEqual(
        fake_mlp,
        FakeMlp(
            first_dense=FakeDense(in_dim=4, out_dim=4),
            second_dense=FakeDenseSubclass(in_dim=7, out_dim=9),
        ))

  def test_auto_skeleton_allows_none_as_unconfigured_sentinel(self):
    ab.auto_skeleton(FakeMlp, second_dense=None)
    self.assertEqual(
        ab.config(FakeMlp),
        fdl.Config(
            FakeMlp,
            first_dense=fdl.Config(FakeDense),
        ))

  def test_auto_skeleton_unannotated_raises(self):

    def unannotated(x, y):
      return x + y

    with self.assertRaises(ValueError):
      ab.auto_skeleton(unannotated)

    # But it's fine if we specify defaults, or None
    ab.auto_skeleton(unannotated, x=FakeDense, y=None)

  def test_auto_skeleton_doesnt_configure_primitives(self):

    @dataclasses.dataclass
    class Foo:
      x: int = 4

    with self.assertRaisesRegex(TypeError,
                                r"Parameter 'x'.*cannot be configured"):
      ab.auto_skeleton(Foo)

    # Fine to do by explicitly not configuring the parameter.
    ab.auto_skeleton(Foo, x=None)
    foo = fdl.build(ab.config(Foo))
    self.assertEqual(foo.x, 4)

  def test_auto_skeleton_attribute_skeletons_required(self):
    ab.auto_skeleton(FakeMlp, attributes_require_skeletons=True)

    with self.assertRaisesRegex(KeyError, "No skeleton.*FakeDense"):
      ab.config(FakeMlp)

    def bar(z):
      return z + 1

    def foo(x):
      return x

    ab.auto_skeleton(foo, x=bar, attributes_require_skeletons=True)

    with self.assertRaisesRegex(KeyError, "No skeleton.*bar"):
      ab.config(foo)

  def test_validator_registers(self):
    registry = ab.Registry()
    fn = lambda config: None
    registry.validator(FakeDense)(fn)
    self.assertDictEqual(registry.table, {
        FakeDense: ab.TableEntry(skeleton_fn=None, validators=[fn]),
    })


if __name__ == "__main__":
  absltest.main()
