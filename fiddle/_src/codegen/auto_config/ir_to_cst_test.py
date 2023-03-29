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

"""Tests for ir_to_cst."""

from absl.testing import absltest

from fiddle._src.codegen.auto_config import init_task
from fiddle._src.codegen.auto_config import ir_to_cst
from fiddle._src.codegen.auto_config import make_symbolic_references
from fiddle._src.codegen.auto_config import shared_to_variables
from fiddle._src.codegen.auto_config import test_fixtures


class IrToCstTest(absltest.TestCase):

  def test_basic_ir(self):
    task = test_fixtures.simple_ir()
    make_symbolic_references.import_symbols(task)
    make_symbolic_references.replace_callables_and_configs_with_symbols(task)
    code = ir_to_cst.code_for_task(task).code
    expected = """
    from fiddle._src.codegen.auto_config import test_fixtures
    from fiddle.experimental import auto_config


    @auto_config.auto_config
    def simple_ir_fixture():
        return test_fixtures.foo(x=4)
    """
    self.assertEqual(code.split(), expected.split(), msg=code)

  def test_two_shared_config(self):
    task = test_fixtures.unprocessed_two_shared_config()
    make_symbolic_references.import_symbols(task)
    shared_to_variables.move_shared_nodes_to_variables(task)
    make_symbolic_references.replace_callables_and_configs_with_symbols(task)
    code = ir_to_cst.code_for_task(task).code
    expected = """
    from fiddle._src.codegen.auto_config import test_fixtures
    from fiddle.experimental import auto_config


    @auto_config.auto_config
    def unprocessed_two_shared_fixture():
        foo = test_fixtures.foo(x=3)
        shared_type = test_fixtures.SharedType(x=foo, z=7.0)
        return [shared_type, shared_type, foo]
    """
    self.assertEqual(code.split(), expected.split(), msg=code)

  def test_complex_dict_node_generation(self):
    # These kinds of dictionaries aren't really supported by other passes, so
    # please don't actually put them in your configs.
    #
    # Also sets are not supported by daglish by default, so they are only
    # generated for primitive values here thanks to py_val_to_cst_converter.
    config = {7.2: ("hi", {3, 4})}
    task = init_task.init_task(config)
    code = ir_to_cst.code_for_task(task).code
    expected = """
    from fiddle.experimental import auto_config


    @auto_config.auto_config
    def config_fixture():
        return {7.2: ('hi', {3, 4})}
    """
    self.assertEqual(code.split(), expected.split(), msg=code)


if __name__ == "__main__":
  absltest.main()
