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

"""Tests for code_ir."""

from absl.testing import absltest
from absl.testing import parameterized
import fiddle as fdl
from fiddle import daglish
from fiddle._src.codegen.auto_config import code_ir
from fiddle._src.codegen.auto_config import test_fixtures


def foo(x=1):
  return x


class NameTest(absltest.TestCase):

  def test_replace(self):
    name = code_ir.Name("foo_1")
    name.replace("foo_2")
    self.assertEqual(name.value, "foo_2")
    self.assertIsNotNone(name.previous)
    self.assertEqual(name.previous.value, "foo_1")
    self.assertTrue(name.is_generated)
    self.assertTrue(name.previous.is_generated)


class CodegenNodeTest(parameterized.TestCase):

  def test_symbol_expression_string_raises(self):
    with self.assertRaisesRegex(TypeError, "Strings are no longer allowed.*"):
      code_ir.FunctoolsPartialCall(
          symbol_expression="functools.partial",
          positional_arg_expressions=[],
          arg_expressions={},
      )

  def test_daglish_iteration(self):
    fn = code_ir.FixtureFunction(
        name=code_ir.Name("foo"),
        parameters=[],
        variables=[],
        output_value=fdl.Config(foo, x=2),
    )
    iterate_results = [
        (daglish.path_str(path), value) for value, path in daglish.iterate(fn)
    ]
    self.assertEqual(
        iterate_results,
        [
            ("", fn),
            (".name", fn.name),
            (".parameters", []),
            (".variables", []),
            (".output_value", fn.output_value),
            (".output_value.x", 2),
            (".return_type_annotation", None),
        ],
    )

  @parameterized.named_parameters(test_fixtures.parameters_for_testcases())
  def test_smoke_traverse_fixtures(self, task: code_ir.CodegenTask):
    functions = task.top_level_call.all_fixture_functions()
    self.assertNotEmpty(functions)
    for fn in functions:
      list(daglish.iterate(fn))


if __name__ == "__main__":
  absltest.main()
