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

"""Tests for add_type_signatures."""

from typing import List

from absl.testing import absltest
from absl.testing import parameterized
import fiddle as fdl
from fiddle._src.codegen import import_manager as import_manager_lib
from fiddle._src.codegen import namespace as namespace_lib
from fiddle._src.codegen.auto_config import add_type_signatures
from fiddle._src.codegen.auto_config import ir_printer
from fiddle._src.codegen.auto_config import test_fixtures
from fiddle._src.testing.example import fake_encoder_decoder


def foo(x):
  return x


def bar(x: int) -> int:
  return x


def baz() -> List[int]:
  return [1]


def qux() -> list:  # pylint: disable=g-bare-generic
  return [1]


class AddTypeSignaturesTest(parameterized.TestCase):

  @parameterized.parameters(
      {
          "value": True,
          "expected": "bool",
      },
      {
          "value": [1, 2, 3],
          "expected": "list[int]",
      },
      {
          "value": [1, 2, "a"],
          "expected": "list",
      },
      {
          "value": {"hi": 3, "bye": 4},
          "expected": "dict[str, int]",
      },
      {
          "value": {},
          "expected": "dict[Any, Any]",
      },
      {
          # Custom types are replaced with Any.
          # (Rationale: Don't put custom objects in Fiddle configs.)
          "value": namespace_lib.Namespace(set()),
          "expected": "Any",
      },
      {
          "value": fdl.Config(foo, x=1),
          "expected": "fdl.Config",
      },
      {
          "value": fdl.Config(bar, x=1),
          "expected": "fdl.Config[int]",
      },
      {
          # TODO(b/293509806): Handle more types, especially from function
          # return signatures.
          "value": fdl.Config(baz),
          "expected": "fdl.Config",
      },
      {
          # TODO(b/293509806): Handle more types, especially from function
          # return signatures.
          "value": fdl.Config(qux),
          "expected": "fdl.Config[Any]",
      },
      {
          "value": fdl.Config(fake_encoder_decoder.FakeEncoderDecoder),
          "expected": "fdl.Config[fake_encoder_decoder.FakeEncoderDecoder]",
      },
      {
          "value": fdl.Partial(foo, x=1),
          "expected": "fdl.Partial",
      },
      {
          "value": fdl.Partial(bar, x=1),
          "expected": "fdl.Partial[int]",
      },
  )
  def test_get_type_annotation(self, value, expected):
    import_manager = import_manager_lib.ImportManager(namespace_lib.Namespace())
    expression = add_type_signatures.get_type_annotation(
        value=value, import_manager=import_manager
    )
    formatted = ir_printer.format_expr(expression)
    self.assertEqual(formatted, expected)

  @parameterized.named_parameters(*test_fixtures.parameters_for_testcases())
  def test_smoke_add_return_types(self, task):
    add_type_signatures.add_return_types(task)


if __name__ == "__main__":
  absltest.main()
