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

"""Tests for codegen."""
import dataclasses
import functools
from typing import List

from absl.testing import absltest
import fiddle as fdl
from fiddle.codegen import codegen
from fiddle.codegen import test_util
from fiddle.codegen.test_submodule import test_util as submodule_test_util


def tokens(code: str) -> List[str]:
  return code.strip().split()


@dataclasses.dataclass
class Foo:
  a: int
  leaves: List["Foo"] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Bar:
  foo1: Foo
  foo2: Foo


@dataclasses.dataclass
class Baz:
  foo: Foo
  bar: Bar


def identity(x):
  return x


def simple_tree() -> fdl.Config[Foo]:
  return fdl.Config(
      Foo, a=1, leaves=[
          fdl.Config(Foo, a=2),
          fdl.Config(Foo, a=3),
      ])


def shared_config() -> fdl.Config[Baz]:
  shared_foo = fdl.Config(Foo, a=1)

  baz = fdl.Config(Baz)
  baz.foo = shared_foo

  baz.bar = fdl.Config(Bar)
  baz.bar.foo1 = fdl.Config(Foo, a=2)
  baz.bar.foo2 = shared_foo

  return baz


def multi_shared_config() -> fdl.Config[Foo]:
  foo = functools.partial(fdl.Config, Foo)

  shared1 = foo(1)
  shared2 = foo(2, leaves=[shared1])

  return foo(
      3, leaves=[
          shared2,
          foo(4),
          foo(5, leaves=[shared1, shared2]),
      ])


def unshared_child_of_shared() -> fdl.Config[Foo]:
  foo = functools.partial(fdl.Config, Foo)

  shared = foo(1, leaves=[foo(2)])
  return foo(3, leaves=[foo(4, leaves=[shared]), foo(5, leaves=[shared])])


def partial_config() -> fdl.Partial[Baz]:
  # Not sure what it means to share a partial, but let's make sure the code
  # works.
  x = fdl.Partial(Foo, a=1)
  return fdl.Partial(Baz, foo=x, bar=fdl.Partial(Bar, foo1=x))


# Depending on how the test harness invokes this test, the expected values may
# or may not include a module name.
if __name__ == "__main__":
  this_module_import = ""
  this_module_prefix = ""
else:
  this_module_import = "from fiddle.codegen import codegen_test"
  this_module_prefix = "codegen_test."


class CodegenTest(absltest.TestCase):

  def test_codegen_dot_syntax_shared(self):
    cfg = shared_config()
    result = codegen.codegen_dot_syntax(cfg)
    expected = f"""
import fiddle as fdl
{this_module_import}


def build_config():
  shared_foo = fdl.Config({this_module_prefix}Foo)
  shared_foo.a = 1

  root = fdl.Config({this_module_prefix}Baz)
  root.foo = shared_foo

  root.bar = fdl.Config({this_module_prefix}Bar)
  root.bar.foo2 = shared_foo

  root.bar.foo1 = fdl.Config({this_module_prefix}Foo)
  root.bar.foo1.a = 2

  return root
    """
    actual_tokens = tokens("\n".join(result.lines()))
    self.assertSequenceEqual(actual_tokens, tokens(expected),
                             "\n".join(result.lines()))

  def test_codegen_multi_shared(self):
    cfg = multi_shared_config()
    result = codegen.codegen_dot_syntax(cfg)
    expected = f"""
import fiddle as fdl
{this_module_import}


def build_config():
  shared_foo = fdl.Config({this_module_prefix}Foo)
  shared_foo.a = 1

  shared_foo_2 = fdl.Config({this_module_prefix}Foo)
  shared_foo_2.a = 2
  shared_foo_2.leaves = [shared_foo]

  root = fdl.Config({this_module_prefix}Foo)
  root.a = 3
  root.leaves = [shared_foo_2, NotImplemented, NotImplemented]  # fdl.Config sub-nodes will replace NotImplemented

  root.leaves[1] = fdl.Config({this_module_prefix}Foo)
  root.leaves[1].a = 4

  root.leaves[2] = fdl.Config({this_module_prefix}Foo)
  root.leaves[2].a = 5
  root.leaves[2].leaves = [shared_foo, shared_foo_2]

  return root
    """
    code = "\n".join(result.lines())
    self.assertSequenceEqual(tokens(code), tokens(expected), code)

  def test_codegen_import_and_exec(self):
    cfg = fdl.Config(
        test_util.Foo, a=1, leaves=[fdl.Config(test_util.Foo, a=2)])
    result = codegen.codegen_dot_syntax(cfg)
    expected = """
import fiddle as fdl
from fiddle.codegen import test_util


def build_config():
  root = fdl.Config(test_util.Foo)
  root.a = 1
  root.leaves = [NotImplemented]  # fdl.Config sub-nodes will replace NotImplemented

  root.leaves[0] = fdl.Config(test_util.Foo)
  root.leaves[0].a = 2

  return root
    """
    code = "\n".join(result.lines())
    self.assertSequenceEqual(tokens(code), tokens(expected), code)

    # Actually run the new builder.
    exec_globals = {}
    exec(code, exec_globals)  # pylint: disable=exec-used
    config = exec_globals["build_config"]()
    foo = fdl.build(config)

    # For now, check the built config values. We might add equality operators to
    # Buildable in the future.
    self.assertEqual(foo.a, 1)
    self.assertLen(foo.leaves, 1)
    self.assertEqual(foo.leaves[0].a, 2)
    self.assertEmpty(foo.leaves[0].leaves)

  def test_codegen_unshared_child_of_shared(self):
    cfg = unshared_child_of_shared()
    result = codegen.codegen_dot_syntax(cfg)
    expected = f"""
import fiddle as fdl
{this_module_import}


def build_config():
  shared_foo = fdl.Config({this_module_prefix}Foo)
  shared_foo.a = 2

  shared_foo_2 = fdl.Config({this_module_prefix}Foo)
  shared_foo_2.a = 1
  shared_foo_2.leaves = [shared_foo]

  root = fdl.Config({this_module_prefix}Foo)
  root.a = 3
  root.leaves = [NotImplemented, NotImplemented]  # fdl.Config sub-nodes will replace NotImplemented

  root.leaves[0] = fdl.Config({this_module_prefix}Foo)
  root.leaves[0].a = 4
  root.leaves[0].leaves = [shared_foo_2]

  root.leaves[1] = fdl.Config({this_module_prefix}Foo)
  root.leaves[1].a = 5
  root.leaves[1].leaves = [shared_foo_2]

  return root
    """
    actual_tokens = tokens("\n".join(result.lines()))
    self.assertSequenceEqual(actual_tokens, tokens(expected),
                             "\n".join(result.lines()))

  def test_codegen_partial(self):
    cfg = partial_config()
    result = codegen.codegen_dot_syntax(cfg)
    expected = f"""
import fiddle as fdl
{this_module_import}


def build_config():
  shared_foo = fdl.Partial({this_module_prefix}Foo)
  shared_foo.a = 1

  root = fdl.Partial({this_module_prefix}Baz)
  root.foo = shared_foo

  root.bar = fdl.Partial({this_module_prefix}Bar)
  root.bar.foo1 = shared_foo

  return root
    """
    actual_tokens = tokens("\n".join(result.lines()))
    self.assertSequenceEqual(actual_tokens, tokens(expected),
                             "\n".join(result.lines()))

  def test_codegen_inner_class_name(self):
    cfg = fdl.Config(test_util.NestedParent.Inner, a=4)
    code = "\n".join(codegen.codegen_dot_syntax(cfg).lines())
    expected = """
import fiddle as fdl
from fiddle.codegen import test_util

def build_config():
  root = fdl.Config(test_util.NestedParent.Inner)
  root.a = 4

  return root
    """
    self.assertSequenceEqual(tokens(code), tokens(expected), code)

  def test_dict_value(self):
    cfg = fdl.Config(identity, x={"foo": fdl.Config(Foo, a=1)})
    code = "\n".join(codegen.codegen_dot_syntax(cfg).lines())
    expected = f"""
import fiddle as fdl
{this_module_import}


def build_config():
  root = fdl.Config({this_module_prefix}identity)
  root.x = {{'foo': NotImplemented}}  # fdl.Config sub-nodes will replace NotImplemented

  root.x['foo'] = fdl.Config({this_module_prefix}Foo)
  root.x['foo'].a = 1

  return root
    """
    self.assertSequenceEqual(tokens(code), tokens(expected), code)

  def test_deeply_nested_constant(self):
    cfg = fdl.Config(identity, x={"bar": [3, 4], "foo": [1, 2]})
    code = "\n".join(codegen.codegen_dot_syntax(cfg).lines())
    expected = f"""
import fiddle as fdl
{this_module_import}


def build_config():
  root = fdl.Config({this_module_prefix}identity)
  root.x = {{'bar': [3, 4], 'foo': [1, 2]}}

  return root
    """
    self.assertSequenceEqual(tokens(code), tokens(expected), code)

  def test_codegen_submodule(self):
    cfg = fdl.Config(
        submodule_test_util.Foo, a=1, leaves=[fdl.Config(test_util.Foo, a=4)])
    code = "\n".join(codegen.codegen_dot_syntax(cfg).lines())
    expected = """
import fiddle as fdl
from fiddle.codegen.test_submodule import test_util
from fiddle.codegen import test_util as test_util_2


def build_config():
  root = fdl.Config(test_util.Foo)
  root.a = 1
  root.leaves = [NotImplemented]  # fdl.Config sub-nodes will replace NotImplemented

  root.leaves[0] = fdl.Config(test_util_2.Foo)
  root.leaves[0].a = 4

  return root
    """
    self.assertSequenceEqual(tokens(code), tokens(expected), code)


if __name__ == "__main__":
  absltest.main()
