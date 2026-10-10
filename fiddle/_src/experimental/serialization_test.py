# coding=utf-8
# Copyright 2026 The Fiddle-Config Authors.
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

"""Tests for JSON serialization."""

from absl.testing import absltest
from absl.testing import parameterized
import fiddle as fdl
from fiddle._src.experimental import serialization


class SerializationTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('empty', b''),
      ('ascii', b'hello'),
      ('all_byte_values', bytes(range(256))),
      ('short_unicode_escape', br'\u00ff'),
      ('long_unicode_escape', br'\U000000ff'),
      ('incomplete_escape', br'\u123'),
      ('invalid_escape', br'\uZZZZ'),
      ('out_of_range_escape', br'\U00110000'),
  )
  def test_bytes_round_trip(self, value):
    config = fdl.Config(dict, payload=value)
    restored = serialization.load_json(serialization.dump_json(config))
    self.assertEqual(fdl.build(restored), {'payload': value})

  def test_load_bytes_from_legacy_unicode_escape_encoding(self):
    serialized = r'''{
      "root": {
        "type": {"type": "pyref", "module": "builtins", "name": "bytes"},
        "items": [["IdentityElement()", {"type": "leaf", "value": "\u1234"}]],
        "metadata": null
      },
      "objects": {}, "refcounts": {}, "version": "0.0.1"
    }'''
    self.assertEqual(serialization.load_json(serialized), br'\u1234')


if __name__ == '__main__':
  absltest.main()
