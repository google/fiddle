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

"""Tests for fiddle.codegen.codegen_diff."""

import ast
import copy
import dataclasses
import re
import textwrap
from typing import Any, NamedTuple

from absl.testing import absltest
from absl.testing import parameterized
import fiddle as fdl
from fiddle.codegen import codegen_diff
from fiddle.experimental import daglish
from fiddle.experimental import diff as fdl_diff
from fiddle.experimental import selectors


class TestNamedTuple(NamedTuple):
  x: Any
  y: Any
  z: Any


@dataclasses.dataclass
class TestClass:
  a: Any
  b: Any


@dataclasses.dataclass
class TestSubclass(TestClass):
  c: Any


def test_func(a, b):
  return (a, b)


class TestTag(fdl.Tag):
  """Fiddle tag for testing."""


class AnotherTag(fdl.Tag):
  """Fiddle tag for testing."""


@dataclasses.dataclass(frozen=True)
class UnknownPathElement(daglish.PathElement):
  code = property(lambda self: '<test>')
  follow = lambda self, container: None


class FiddlerFromDiffTest(parameterized.TestCase):

  def assertSameAst(self, actual_ast, expected_pysrc):
    expected_pysrc = textwrap.dedent(expected_pysrc)

    # First check the pysrc as string (ignoring whitespace).  If they differ,
    # then we can give a better error message by displaying the pysrc diffs.
    if hasattr(ast, 'unparse'):
      actual_pysrc = ast.unparse(actual_ast)
      if ' '.join(actual_pysrc.split()) != ' '.join(expected_pysrc.split()):
        self.assertEqual(actual_pysrc.strip(), expected_pysrc.strip())

    # Next, check the ast actually matches what we get if we parse
    # expected_pysrc.  This will catch missing fields in the ast (such as
    # ctx) which aren't necessarily reflected in the unparsed pysrc.
    expected_ast = ast.parse(expected_pysrc)
    self.assertEqual(
        ast.dump(actual_ast, indent=2), ast.dump(expected_ast, indent=2))

  def check_fiddler_from_diff(self, old, new, expected):
    alignment = fdl_diff.align_heuristically(old, new)
    diff = fdl_diff.build_diff_from_alignment(alignment)
    fiddler_ast = codegen_diff.fiddler_from_diff(diff, old=old)
    self.assertSameAst(fiddler_ast, expected)

  def test_simple_changes(self):
    old = fdl.Config(TestNamedTuple, 1, 2)
    new = copy.deepcopy(old)
    del new.x
    new.y *= 10
    new.z = 'new_value'
    self.check_fiddler_from_diff(
        old, new, """
        def fiddler(cfg):
            del cfg.x
            cfg.y = 20
            cfg.z = 'new_value'
        """)

  def test_swap_children(self):
    old = fdl.Config(TestNamedTuple, [1], [2, 3])
    new = copy.deepcopy(old)
    new.x, new.y = new.y, new.x
    self.check_fiddler_from_diff(
        old, new, """
        def fiddler(cfg):
            moved_cfg_x = cfg.x
            moved_cfg_y = cfg.y
            cfg.x = moved_cfg_y
            cfg.y = moved_cfg_x
        """)

  def test_duplicate_child(self):
    old = fdl.Config(TestNamedTuple, [1])
    new = copy.deepcopy(old)
    new.y = copy.deepcopy(new.x)
    self.check_fiddler_from_diff(
        old, new, """
        def fiddler(cfg):
            cfg.y = [1]
        """)

  def test_new_shared_value_list(self):
    old = fdl.Config(TestNamedTuple, 1, 2, 3)
    new = copy.deepcopy(old)
    new.x = [1]
    new.y = [2, new.x]
    self.check_fiddler_from_diff(
        old, new, """
        def fiddler(cfg):
            shared_list = [1]
            cfg.x = shared_list
            cfg.y = [2, shared_list]
        """)

  def test_new_shared_value_config(self):
    old = fdl.Config(TestNamedTuple, 1, 2, 3)
    new = copy.deepcopy(old)
    new.x = fdl.Config(test_func, 3)
    new.y = [2, new.x]
    self.check_fiddler_from_diff(
        old, new, """
        import fiddle as fdl
        def fiddler(cfg):
            shared_test_func = fdl.Config(test_func, a=3)
            cfg.x = shared_test_func
            cfg.y = [2, shared_test_func]
        """)

  def test_add_config(self):
    old = fdl.Config(TestNamedTuple, 1, 2, 3)
    new = copy.deepcopy(old)
    new.x = fdl.Config(TestNamedTuple, 4)
    # Note: since we reference `fdl.Config` in the fiddler, an import statement
    # gets added to the output.
    self.check_fiddler_from_diff(
        old, new, """
        import fiddle as fdl
        def fiddler(cfg):
            cfg.x = fdl.Config(TestNamedTuple, x=4)
        """)

  def test_add_namedtuple(self):
    old = fdl.Config(TestNamedTuple, 1, 2, 3)
    new = copy.deepcopy(old)
    new.x = TestNamedTuple(4, 5, 6)
    self.check_fiddler_from_diff(
        old, new, """
        def fiddler(cfg):
            cfg.x = TestNamedTuple(x=4, y=5, z=6)
        """)

  def test_change_tagged_value(self):
    # We don't currently do any special handling for tagged values.  So if
    # you use `selectors.select` to change all values of a tag at once, this
    # will show up in the codegen'ed fiddler as separate changes for each
    # place where the tag is used.  (We may add special handling for this in
    # the future, in which case the fiddler could contain `select` calls.)
    old = fdl.Config(TestNamedTuple, 1, TestTag.new(2),
                     fdl.Config(test_func, TestTag.new(2)))
    new = copy.deepcopy(old)
    selectors.select(new, tag=TestTag).set(value=22)
    self.check_fiddler_from_diff(
        old, new, """
        def fiddler(cfg):
            cfg.y.value = 22
            cfg.z.a.value = 22
        """)

  def test_change_callable(self):
    old = fdl.Config(TestClass, 1, 2)
    new = copy.deepcopy(old)
    fdl.update_callable(new, TestSubclass)
    new.c = 3
    self.check_fiddler_from_diff(
        old, new, """
        import fiddle as fdl
        def fiddler(cfg):
            fdl.update_callable(cfg, TestSubclass)
            cfg.c = 3
        """)

  def test_change_config_to_partial(self):
    old = fdl.Config(TestClass, fdl.Config(TestNamedTuple, [1], 2))
    new = copy.deepcopy(old)
    new.a = fdl.Partial(new.a)
    self.check_fiddler_from_diff(
        old, new, """
        import fiddle as fdl
        def fiddler(cfg):
            moved_cfg_a_x = cfg.a.x
            cfg.a = fdl.Partial(TestNamedTuple, x=moved_cfg_a_x, y=2)
        """)

  def test_invalidate_path_via_shared_value(self):
    diff = fdl_diff.Diff(
        changes={
            (daglish.Index(0), daglish.Index(0)):
                fdl_diff.ModifyValue(None),
            (daglish.Index(1), daglish.Index(0), daglish.Index(0)):
                fdl_diff.ModifyValue(1),
        })

    shared_a = [0]
    shared_b = [shared_a]
    shared_old = [shared_b, shared_b, shared_a]

    with self.subTest('unshared'):
      # Because cfg[0][0] and cfg[1][0] point to the same object, and we are
      # modifying that object, we store it in moved_cfg_0_0 before making any
      # mutations.
      fiddler_ast = codegen_diff.fiddler_from_diff(diff, old=shared_old)
      self.assertSameAst(
          fiddler_ast, """
          def fiddler(cfg):
              moved_cfg_1_0 = cfg[1][0]
              cfg[0][0] = None
              moved_cfg_1_0[0] = 1
          """)

    with self.subTest('unshared'):
      # But if we have the same values but no sharing, then we don't need to
      # bother to store it.
      unshared_old = [[[0]], [[0]], [0]]
      fiddler_ast = codegen_diff.fiddler_from_diff(diff, old=unshared_old)
      self.assertSameAst(
          fiddler_ast, """
          def fiddler(cfg):
              cfg[0][0] = None
              cfg[1][0][0] = 1
          """)

    with self.subTest('no_old_value'):
      # If we don't know what `old` is, then we pessimistically record aliases
      # for everything that we use.
      fiddler_ast = codegen_diff.fiddler_from_diff(diff, old=None)
      self.assertSameAst(
          fiddler_ast, """
          def fiddler(cfg):
              original_cfg_0 = cfg[0]
              original_cfg_1_0 = cfg[1][0]
              original_cfg_0[0] = None
              original_cfg_1_0[0] = 1
          """)

  def test_using_imported_values(self):
    old = fdl.Config(TestClass, fdl.Config(TestNamedTuple, [1], 2))
    new = copy.deepcopy(old)
    new.a = fdl.Config(re.sub, 'a|b')
    new.b = fdl.Config(daglish.path_str)
    self.check_fiddler_from_diff(
        old, new, """
        import fiddle as fdl
        from fiddle.experimental import daglish
        import re

        def fiddler(cfg):
            cfg.b = fdl.Config(daglish.path_str)
            del cfg.a.x
            del cfg.a.y
            fdl.update_callable(cfg.a, re.sub)
            cfg.a.pattern = 'a|b'
        """)

  def test_import_name_conflicts(self):
    # If the names we use for `func_name` or `param_name` conflict with
    # the names of other variables we create (e.g., for imports), then
    # we use names to avoid those conflicts.
    old = [1, 2, 3, 4, 5]
    new = copy.deepcopy(old)
    new[0] = fdl.Config(re.sub)
    alignment = fdl_diff.align_heuristically(old, new)
    diff = fdl_diff.build_diff_from_alignment(alignment)
    fiddler_ast = codegen_diff.fiddler_from_diff(
        diff, old=old, func_name='re', param_name='fdl')
    self.assertSameAst(
        fiddler_ast, """
        import fiddle as fdl_2
        import re as re_2

        def re(fdl):
            fdl[0] = fdl_2.Config(re_2.sub)
        """)

  def test_param_name_conflict(self):
    # The param_name and func_name are reserved, so they won't be used when
    # generating new variables for shared values (or moved values).
    old = [1, 1]
    new = copy.deepcopy(old)
    new[0] = [1]
    new[1] = new[0]  # shared
    alignment = fdl_diff.align_heuristically(old, new)
    diff = fdl_diff.build_diff_from_alignment(alignment)
    fiddler_ast = codegen_diff.fiddler_from_diff(
        diff, old=old, param_name='shared_list')
    self.assertSameAst(
        fiddler_ast, """
        def fiddler(shared_list):
            shared_list_2 = [1]
            shared_list[0] = shared_list_2
            shared_list[1] = shared_list_2
        """)

  def test_multiple_changes(self):
    old: Any = [1, {'foo': [2]}, [3, 4, [5, 6, 7]]]
    new = copy.deepcopy(old)
    new[0] = 3
    moved_foo = new[1]['foo']
    new_shared_list = [55]
    new[1]['foo'] = [2, moved_foo]
    new[1]['bar'] = fdl.Config(TestNamedTuple, new[2][2], new_shared_list,
                               new_shared_list)

    self.check_fiddler_from_diff(
        old, new, """
      import fiddle as fdl

      def fiddler(cfg):
          shared_list = [55]
          moved_cfg_1_foo = cfg[1]['foo']
          cfg[0] = 3
          cfg[1]['foo'] = [2, moved_cfg_1_foo]
          cfg[1]['bar'] = fdl.Config(TestNamedTuple, x=cfg[2][2],
                                     y=shared_list, z=shared_list)
        """)

  def test_error_cant_change_root_object(self):
    diff = fdl_diff.Diff(changes={(): fdl_diff.ModifyValue(1)})
    with self.assertRaisesRegex(ValueError,
                                'Changing the root object is not supported'):
      codegen_diff.fiddler_from_diff(diff)

  def test_error_unsupported_path_elt(self):
    diff = fdl_diff.Diff(
        changes={(UnknownPathElement(),): fdl_diff.ModifyValue(1)})
    with self.assertRaisesRegex(ValueError, 'Unsupported PathElement'):
      codegen_diff.fiddler_from_diff(diff)


if __name__ == '__main__':
  absltest.main()
