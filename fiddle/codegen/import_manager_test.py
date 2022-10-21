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

"""Tests for import_manager."""

import textwrap

from absl.testing import absltest
from absl.testing import parameterized
from fiddle.codegen import import_manager
from fiddle.codegen import namespace
import libcst as cst


class ImportManagerTest(parameterized.TestCase):

  @parameterized.parameters(
      ("from qux import foo as name", "name"),
      ("from qux.fandangle import foo as name", "name"),
      ("import foo as name", "name"),
      ("import name", "name"),
      ("import name1.name2", "name1.name2"),
      ("import foo.bar as name", "name"),
  )
  def test_get_import_name(self, import_stmt, expected):
    parsed = import_manager.parse_import(import_stmt)
    actual = import_manager.get_import_name(parsed)
    self.assertEqual(actual, expected)

  @parameterized.parameters(
      ("from qux import foo as name", "from qux import foo as new_name"),
      ("from qux.fandangle import foo as name",
       "from qux.fandangle import foo as new_name"),
      ("import foo as name", "import foo as new_name"),
      ("import name", "import name as new_name"),
      ("import name1.name2", "import name1.name2 as new_name"),
      ("import foo.bar as name", "import foo.bar as new_name"),
  )
  def test_change_alias(self, import_stmt, expected):
    parsed = import_manager.parse_import(import_stmt)
    renamed = import_manager.change_alias(parsed, "new_name")
    as_str = import_manager._dummy_module_for_formatting.code_for_node(renamed)
    self.assertEqual(as_str, expected)

  def test_import_manager_basic_use(self):
    manager = import_manager.ImportManager(namespace.Namespace())
    self.assertEqual(manager.add_by_name("fiddle.config"), "fdl")
    self.assertEqual(manager.add_by_name("fiddle"), "fdl")
    self.assertEqual(manager.add_by_name("foo"), "foo")
    self.assertEqual(manager.add_by_name("foo.bar"), "bar")
    self.assertEqual(manager.add_by_name("baz.bar"), "bar_2")
    self.assertEqual(manager.add_by_name("foo.bar"), "bar")
    self.assertEqual(manager.add_by_name("baz.bar"), "bar_2")

    module = cst.Module(body=manager.sorted_import_lines())
    self.assertEqual(
        module.code.strip(),
        textwrap.dedent("""
            from baz import bar as bar_2
            import fiddle as fdl
            import foo
            from foo import bar""".strip("\n")))

  def test_import_manager_fiddle_and_fiddle_config(self):
    manager = import_manager.ImportManager(namespace.Namespace())
    self.assertEqual(manager.add_by_name("fiddle"), "fdl")
    self.assertEqual(manager.add_by_name("fiddle.config"), "fdl")
    self.assertEqual(manager.add_by_name("fiddle"), "fdl")
    self.assertEqual(manager.add_by_name("fiddle.config"), "fdl")
    module = cst.Module(body=manager.sorted_import_lines())
    self.assertEqual(module.code.strip(), "import fiddle as fdl")

  def test_import_manager_aliasing_builtin(self):
    manager = import_manager.ImportManager(namespace.Namespace({"fdl"}))
    self.assertEqual(manager.add_by_name("fiddle.config"), "fdl_2")
    module = cst.Module(body=manager.sorted_import_lines())
    self.assertEqual(module.code.strip(), "import fiddle as fdl_2")

  def test_dotted_special_import_okay(self):
    """Tests cases where dotted imports are used."""
    import_manager.register_import_alias("lace.bar", "import lace.bar")
    import_manager.register_import_alias("lace.lace", "import lace.lace")
    manager = import_manager.ImportManager(namespace.Namespace())
    self.assertEqual(manager.add_by_name("lace.bar"), "lace.bar")
    self.assertEqual(manager.add_by_name("lace.lace"), "lace.lace")
    module = cst.Module(body=manager.sorted_import_lines())
    self.assertEqual(
        module.code.strip(),
        textwrap.dedent("""
            import lace.bar
            import lace.lace""".strip("\n")))

  def test_dotted_special_import_conflicts(self):
    """Tests cases where dotted imports are used."""
    import_manager.register_import_alias("lace.bar", "import lace.bar")
    manager = import_manager.ImportManager(namespace.Namespace())
    self.assertEqual(manager.add_by_name("lace.bar"), "lace.bar")
    self.assertEqual(manager.add_by_name("lace.qux.lace"), "lace_2")
    module = cst.Module(body=manager.sorted_import_lines())
    self.assertEqual(
        module.code.strip(),
        textwrap.dedent("""
            import lace.bar
            from lace.qux import lace as lace_2""".strip("\n")))


if __name__ == "__main__":
  absltest.main()
