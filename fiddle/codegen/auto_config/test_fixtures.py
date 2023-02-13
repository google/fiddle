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

"""Some manually-generated code intermediate representations for testing."""

import fiddle as fdl
from fiddle.codegen.auto_config import code_ir


def foo(x):
  return x


def simple_ir() -> code_ir.CodegenTask:
  """Returns a single fixture bound to a config.

  Code:

  def simple_ir_fixture():
    return fdl.Config(foo, x=4)
  """
  config = fdl.Config(foo, x=4)
  fn = code_ir.FixtureFunction(
      name=code_ir.Name("simple_ir_fixture"),
      parameters=[],
      variables=[],
      output_value=config,
  )
  call_instance = code_ir.CallInstance(
      fn, parent=None, children={}, parameter_values={}, output_value=config
  )
  return code_ir.CodegenTask(config, call_instance)


def simple_shared_variable_ir() -> code_ir.CodegenTask:
  """Returns a single fixture bound to a config.

  Code:

  def simple_shared_variable_ir_fixture():
    shared = {"a": 7}
    return [shared, shared]
  """
  shared = {"a": 7}
  config = [shared, shared]

  shared_name = code_ir.Name("shared")
  fn = code_ir.FixtureFunction(
      name=code_ir.Name("simple_shared_variable_ir_fixture"),
      parameters=[],
      variables=[code_ir.VariableDeclaration(shared_name, shared)],
      output_value=[
          code_ir.VariableReference(shared_name),
          code_ir.VariableReference(shared_name),
      ],
  )
  call_instance = code_ir.CallInstance(
      fn, parent=None, children={}, parameter_values={}, output_value=config
  )
  return code_ir.CodegenTask(config, call_instance)


def parameters_for_testcases():
  """Returns parameters for absl's parameterized test cases."""
  return [
      {
          "testcase_name": "simple_ir",
          "task": simple_ir(),
      },
      {
          "testcase_name": "simple_shared_variable_ir",
          "task": simple_shared_variable_ir(),
      },
  ]
