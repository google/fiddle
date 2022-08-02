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

"""Tests for adding_tags."""

import dataclasses

from absl.testing import absltest
import fiddle as fdl
from fiddle.experimental import adding_tags
from fiddle.experimental import auto_config


class Tag1(fdl.Tag):
  """One sample tag."""


class Tag2(fdl.Tag):
  """A second sample tag."""


@dataclasses.dataclass
class SampleClass:
  arg1: int
  arg2: str


class AddingTagsTest(absltest.TestCase):

  def test_adding_tags(self):

    @auto_config.auto_config
    def sample_fn(x):
      x_tagged = adding_tags.adding_tags(tags=Tag1, value=x)
      arg2 = adding_tags.adding_tags(tags=(Tag1, Tag2), value='42')
      return SampleClass(arg1=x_tagged, arg2=arg2)

    expected_object = SampleClass(5, '42')
    self.assertEqual(expected_object, sample_fn(5))

    expected_config = fdl.Config(SampleClass, arg1=5, arg2='42')
    fdl.set_tags(expected_config, 'arg1', {Tag1})
    fdl.set_tags(expected_config, 'arg2', {Tag1, Tag2})

    self.assertEqual(expected_config, sample_fn.as_buildable(5))


if __name__ == '__main__':
  absltest.main()
