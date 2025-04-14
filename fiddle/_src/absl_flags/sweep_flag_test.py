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

import copy
import dataclasses
import sys

from absl.testing import absltest
from fiddle._src import config
from fiddle._src.absl_flags import sweep_flag
from fiddle._src.experimental import auto_config


# We'll just test the _parse method, the rest is relatively trivial wrapper.
FLAG = sweep_flag.DEFINE_fiddle_sweep(
    "_dummy", default_module=sys.modules[__name__], allow_imports=False
)


@dataclasses.dataclass
class Foo:
  x: str
  y: int
  z: int = 0


def get_config(arg: str = "arg", kwarg: int = 1):
  return config.Config(Foo, x=arg, y=kwarg)


@auto_config.auto_config
def get_config_auto(kwarg: int = 1):
  return Foo(x="foo", y=kwarg)


def fiddler(cfg):
  cfg.y *= 2


def arg_kwarg_sweep():
  return [{"arg:0": "foo"}, {"kwarg:kwarg": 3}]


def override_sweep(y=3):
  return [{"x": "foo"}, {"y": y}]


def x_sweep():
  return [{"x": "foo"}, {"x": "bar"}]


def y_sweep():
  return [{"y": 2}, {"y": 3}]


def kwarg_sweep():
  return [{"kwarg:kwarg": 2}, {"kwarg:kwarg": 3}]


class FiddleSweepFlagTest(absltest.TestCase):

  def assert_single_config_sweep(self, sweep, expected):
    self.assertEqual(
        sweep, [sweep_flag.SweepItem(config=expected, overrides_applied={})]
    )

  def test_plain_config(self):
    sweep = FLAG._parse(["config:get_config"])
    expected = get_config()
    self.assert_single_config_sweep(sweep, expected)

  def test_config_with_args(self):
    sweep = FLAG._parse(["config:get_config('foo', kwarg=2)"])
    expected = get_config("foo", kwarg=2)
    self.assert_single_config_sweep(sweep, expected)

  def test_auto_config_with_args(self):
    sweep = FLAG._parse(["config:get_config_auto(kwarg=2)"])
    expected = get_config_auto.as_buildable(kwarg=2)
    self.assert_single_config_sweep(sweep, expected)

  def test_set(self):
    sweep = FLAG._parse(["config:get_config", "set:x='bar'"])
    expected = get_config()
    expected.x = "bar"
    self.assert_single_config_sweep(sweep, expected)

  def test_fiddler(self):
    sweep = FLAG._parse(["config:get_config", "fiddler:fiddler"])
    expected = get_config()
    fiddler(expected)
    self.assert_single_config_sweep(sweep, expected)

  def test_arg_kwarg_sweep(self):
    sweep = FLAG._parse(["config:get_config", "sweep:arg_kwarg_sweep"])
    expected = [
        sweep_flag.SweepItem(
            config=get_config("foo"),
            overrides_applied={"arg:0": "foo"},
        ),
        sweep_flag.SweepItem(
            config=get_config(kwarg=3),
            overrides_applied={"kwarg:kwarg": 3},
        ),
    ]
    self.assertEqual(sweep, expected)

  def test_arg_kwarg_sweep_overriding_existing(self):
    sweep = FLAG._parse(
        ["config:get_config('bar', kwarg=2)", "sweep:arg_kwarg_sweep"]
    )
    expected = [
        sweep_flag.SweepItem(
            config=get_config("foo", kwarg=2),
            overrides_applied={"arg:0": "foo"},
        ),
        sweep_flag.SweepItem(
            config=get_config("bar", kwarg=3),
            overrides_applied={"kwarg:kwarg": 3},
        ),
    ]
    self.assertEqual(sweep, expected)

  def test_override_sweep_and_sweep_arg(self):
    sweep = FLAG._parse(["config:get_config", "sweep:override_sweep(y=4)"])
    parent_config = get_config()
    config1 = copy.copy(parent_config)
    config1.x = "foo"
    config2 = copy.copy(parent_config)
    config2.y = 4
    expected = [
        sweep_flag.SweepItem(
            config=config1,
            overrides_applied={"x": "foo"},
        ),
        sweep_flag.SweepItem(
            config=config2,
            overrides_applied={"y": 4},
        ),
    ]
    self.assertEqual(sweep, expected)

  def test_product_sweep(self):
    sweep = FLAG._parse(["config:get_config", "sweep:x_sweep", "sweep:y_sweep"])
    expected = [
        sweep_flag.SweepItem(
            config=config.Config(Foo, x="foo", y=2),
            overrides_applied={"x": "foo", "y": 2},
        ),
        sweep_flag.SweepItem(
            config=config.Config(Foo, x="foo", y=3),
            overrides_applied={"x": "foo", "y": 3},
        ),
        sweep_flag.SweepItem(
            config=config.Config(Foo, x="bar", y=2),
            overrides_applied={"x": "bar", "y": 2},
        ),
        sweep_flag.SweepItem(
            config=config.Config(Foo, x="bar", y=3),
            overrides_applied={"x": "bar", "y": 3},
        ),
    ]
    self.assertEqual(sweep, expected)

  def test_set_and_override_sweep(self):
    sweep = FLAG._parse([
        "config:get_config",
        "set:y=5",
        "sweep:x_sweep",
    ])
    parent_config = get_config()
    parent_config.y = 5
    expected = [
        sweep_flag.SweepItem(
            config=config.Config(Foo, x="foo", y=5),
            overrides_applied={"x": "foo"},
        ),
        sweep_flag.SweepItem(
            config=config.Config(Foo, x="bar", y=5),
            overrides_applied={"x": "bar"},
        ),
    ]
    self.assertEqual(sweep, expected)

  def test_kwarg_sweep_and_set(self):
    sweep = FLAG._parse([
        "config:get_config",
        "sweep:kwarg_sweep",
        "set:x='foo'",
    ])
    config1 = get_config(kwarg=2)
    config1.x = "foo"
    config2 = get_config(kwarg=3)
    config2.x = "foo"
    expected = [
        sweep_flag.SweepItem(
            config=config1,
            overrides_applied={"kwarg:kwarg": 2},
        ),
        sweep_flag.SweepItem(
            config=config2,
            overrides_applied={"kwarg:kwarg": 3},
        ),
    ]
    self.assertEqual(sweep, expected)

  def test_shebang(self):
    sweep = FLAG._parse([
        "config:get_config",
        "sweep:x_sweep",
        "sweep:kwarg_sweep",
        "set:z=10",
    ])
    expected = [
        sweep_flag.SweepItem(
            config=config.Config(Foo, x="foo", y=2, z=10),
            overrides_applied={"x": "foo", "kwarg:kwarg": 2},
        ),
        sweep_flag.SweepItem(
            config=config.Config(Foo, x="foo", y=3, z=10),
            overrides_applied={"x": "foo", "kwarg:kwarg": 3},
        ),
        sweep_flag.SweepItem(
            config=config.Config(Foo, x="bar", y=2, z=10),
            overrides_applied={"x": "bar", "kwarg:kwarg": 2},
        ),
        sweep_flag.SweepItem(
            config=config.Config(Foo, x="bar", y=3, z=10),
            overrides_applied={"x": "bar", "kwarg:kwarg": 3},
        ),
    ]
    self.assertEqual(sweep, expected)


if __name__ == "__main__":
  absltest.main()
