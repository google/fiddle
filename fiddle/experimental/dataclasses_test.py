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

"""Tests for dataclasses."""

import dataclasses
import types

from absl.testing import absltest
import fiddle as fdl
from fiddle.experimental import dataclasses as fdl_dc


class SampleTag(fdl.Tag):
  """A tag used in testing."""


class AdditionalTag(fdl.Tag):
  """A second tag to test multiple tags."""


@dataclasses.dataclass
class ATaggedType:
  untagged: str
  tagged: str = fdl_dc.field(tags=SampleTag, default='tagged')
  double_tagged: str = fdl_dc.field(
      tags=(AdditionalTag, SampleTag), default_factory=lambda: 'other_field')


class DataclassesTest(absltest.TestCase):

  def test_dataclass_tagging(self):
    config = fdl.Config(ATaggedType)

    self.assertEqual({SampleTag}, fdl.get_tags(config, 'tagged'))
    self.assertEqual({SampleTag, AdditionalTag},
                     fdl.get_tags(config, 'double_tagged'))

    fdl.set_tagged(config, tag=AdditionalTag, value='set_correctly')

    self.assertEqual(config.double_tagged, 'set_correctly')

  def test_metadata_passthrough(self):
    other_metadata = types.MappingProxyType({'something': 4})
    constructed_field = fdl_dc.field(metadata=other_metadata)

    self.assertIn('something', constructed_field.metadata)
    self.assertEqual(4, constructed_field.metadata['something'])


if __name__ == '__main__':
  absltest.main()
