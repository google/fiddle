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

"""Initializes a codegen task for a config.

This is a low-level API, usually the first stage in generating an auto_config
fixture.
"""
from typing import Any

from fiddle._src.codegen.auto_config import code_ir


def init_task(
    config: Any, *, top_level_fixture_name: str = "config_fixture"
) -> code_ir.CodegenTask:
  """Initializes a codegen task for a config.

  Args:
    config: Fiddle buildable, or other type to generate an auto_config function
      for.
    top_level_fixture_name: Name of the top-level function.

  Returns:
    CodegenTask, representing code generation.
  """
  fn = code_ir.FixtureFunction(
      name=code_ir.Name(
          top_level_fixture_name,
          is_generated=top_level_fixture_name == "config_fixture",
      ),
      parameters=[],
      variables=[],
      output_value=config,
  )
  call_instance = code_ir.CallInstance(
      fn, parent=None, children=[], parameter_values={}
  )
  return code_ir.CodegenTask(config, top_level_call=call_instance)
