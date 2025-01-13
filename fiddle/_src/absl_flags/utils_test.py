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

from absl.testing import absltest
from fiddle._src.absl_flags import submodule_for_flags_test
from fiddle._src.absl_flags import utils

_IRRELEVANT_MODE = utils.ImportDottedNameDebugContext.BASE_CONFIG


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


if __name__ == '__main__':
  absltest.main()
