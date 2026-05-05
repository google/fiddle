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

import dataclasses
import sys
from typing import Any

from absl.testing import absltest
import fiddle as fdl
from fiddle._src import config
from fiddle._src.absl_flags import submodule_for_flags_test
from fiddle._src.absl_flags import utils
from fiddle._src.testing.example import fake_encoder_decoder

_IRRELEVANT_MODE = utils.ImportDottedNameDebugContext.BASE_CONFIG


def base_experiment() -> fdl.Config:
  return fake_encoder_decoder.fixture.as_buildable()


@dataclasses.dataclass
class TestConfig:
  a: int


def another_base_config() -> fdl.Config:
  return fdl.Config(TestConfig, a=2)


class ResolveFunctionReferenceTest(absltest.TestCase):

  def test_module_relative_resolution(self):
    import fiddle._src.absl_flags as parent  # pylint: disable=g-import-not-at-top

    self.assertIs(
        utils.resolve_function_reference(
            function_name='config_bar',
            mode=_IRRELEVANT_MODE,
            module=submodule_for_flags_test,
            allow_imports=True,
            failure_msg_prefix='',
        ),
        submodule_for_flags_test.config_bar,
    )
    self.assertIs(
        utils.resolve_function_reference(
            function_name='submodule_for_flags_test.config_bar',
            mode=_IRRELEVANT_MODE,
            module=parent,
            allow_imports=True,
            failure_msg_prefix='',
        ),
        submodule_for_flags_test.config_bar,
    )

  def test_module_relative_resolution_falls_back_to_absolute(self):
    self.assertIs(
        utils.resolve_function_reference(
            function_name=(
                'fiddle._src.absl_flags.submodule_for_flags_test.config_bar'
            ),
            mode=_IRRELEVANT_MODE,
            module=utils,
            allow_imports=True,
            failure_msg_prefix='',
        ),
        submodule_for_flags_test.config_bar,
    )

  def test_raises_without_resolvable_name(self):
    with self.assertRaisesRegex(
        ValueError, "'config_bar': Could not resolve reference"
    ):
      utils.resolve_function_reference(
          function_name='config_bar',
          mode=_IRRELEVANT_MODE,
          module=None,
          allow_imports=True,
          failure_msg_prefix='',
      )

  def test_raises_with_imports_disabled(self):
    with self.assertRaisesRegex(ValueError, 'available names: '):
      utils.resolve_function_reference(
          function_name=(
              'fiddle._src.absl_flags.submodule_for_flags_test.config_bar'
          ),
          mode=_IRRELEVANT_MODE,
          module=utils,
          allow_imports=False,
          failure_msg_prefix='',
      )


class WithOverridesTest(absltest.TestCase):

  def test_with_overrides(self):
    @dataclasses.dataclass
    class Foo:
      x: Any
      y: Any

    subconfig = config.Config(Foo, x=1, y=2)
    cfg = config.Config(
        Foo,
        x=subconfig,
        y=subconfig,
    )
    updated_cfg = utils.with_overrides(cfg, {'x.y': 3})
    # This should affect both places, since we are mutating a single shared
    # object:
    self.assertEqual(updated_cfg.x.y, 3)
    self.assertEqual(updated_cfg.y.y, 3)
    # The original config should be unchanged however:
    self.assertEqual(cfg.x.y, 2)
    self.assertEqual(cfg.y.y, 2)


class InitConfigFromExpressionTest(absltest.TestCase):

  def test_from_function_name(self):
    cfg = utils.init_config_from_expression(
        'base_experiment',
        module=sys.modules[__name__],
        allow_imports=False,
    )

    self.assertEqual(cfg, base_experiment())

  def test_from_function_call(self):
    cfg = utils.init_config_from_expression(
        'another_base_config()',
        module=sys.modules[__name__],
        allow_imports=False,
    )

    self.assertEqual(cfg, another_base_config())

  def test_from_fully_qualified_name(self):
    cfg = utils.init_config_from_expression(
        'fiddle._src.absl_flags.utils_test.base_experiment',
        module=None,
        allow_imports=True,
    )

    self.assertEqual(cfg, base_experiment())


if __name__ == '__main__':
  absltest.main()
