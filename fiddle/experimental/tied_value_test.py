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

"""Tests for tied_value."""

from absl.testing import absltest
import fiddle as fdl
from fiddle.experimental import tied_value
from fiddle.testing.example import fake_encoder_decoder


class DType(fdl.Tag):
  """Sample dtype tag."""


def tied_config():
  config = fake_encoder_decoder.fixture.as_buildable().encoder
  config.attention.dtype = config.mlp.dtype = tied_value.new(value='float32')
  return config


def tied_tagged_config():
  """Tests the interaction between TaggedValueCls and TiedValue."""
  config = fake_encoder_decoder.fixture.as_buildable().encoder
  config.attention.dtype = config.mlp.dtype = tied_value.new(
      DType.new('float32'))
  return config


def tagged_tied_config():
  """Tests the interaction between TaggedValueCls and TiedValue."""
  config = fake_encoder_decoder.fixture.as_buildable().encoder
  config.attention.dtype = config.mlp.dtype = DType.new(
      tied_value.new('float32'))
  return config


class TiedValueTest(absltest.TestCase):

  def test_new_doesnt_rewrap(self):
    tied = tied_value.new(value='float32')
    self.assertIsInstance(tied, tied_value.TiedValue)
    self.assertEqual(tied.value, 'float32')

    wrapped = tied_value.new(tied)
    self.assertIs(wrapped, tied)

  def test_basic_sharing(self):
    config = tied_config()
    encoder = fdl.build(config)
    self.assertEqual(encoder.attention.dtype, 'float32')
    self.assertEqual(encoder.mlp.dtype, 'float32')

    config.attention.dtype = 'bfloat16'
    encoder = fdl.build(config)
    self.assertEqual(encoder.attention.dtype, 'bfloat16')
    self.assertEqual(encoder.mlp.dtype, 'bfloat16')

  def test_basic_sharing_2(self):
    # Sets the shared field through the other path.
    config = tied_config()
    config.mlp.dtype = 'float64'
    encoder = fdl.build(config)
    self.assertEqual(encoder.attention.dtype, 'float64')
    self.assertEqual(encoder.mlp.dtype, 'float64')

  def test_tagged_and_tied_iteraction(self):
    """Tests interaction between values that are tagged and tied.

    Note: This only applies to TaggedValueCls, which is a bit legacy. Usually
    tags attached to fields should be used.
    """
    for config in [tied_tagged_config(), tagged_tied_config()]:
      config.attention.dtype.value = 'bfloat16'
      encoder = fdl.build(config)
      self.assertEqual(encoder.attention.dtype, 'bfloat16')
      self.assertEqual(encoder.mlp.dtype, 'bfloat16')

  def test_untie_and_set_tied(self):
    config = tied_config()
    tied_value.untie(config.attention, 'dtype')
    config.attention.dtype = 'bfloat16'
    encoder = fdl.build(config)
    self.assertEqual(encoder.attention.dtype, 'bfloat16')
    self.assertEqual(encoder.mlp.dtype, 'float32')

  def test_untie_and_set_untied(self):
    config = tied_config()
    tied_value.untie(config.attention, 'dtype')
    config.mlp.dtype = 'bfloat16'
    encoder = fdl.build(config)
    self.assertEqual(encoder.attention.dtype, 'float32')
    self.assertEqual(encoder.mlp.dtype, 'bfloat16')

  def test_more_than_two_tied(self):
    config = tied_config()
    config.embedders['tokens'].dtype = tied_value.get_tied(config.mlp, 'dtype')
    config.attention.dtype = 'complex64'
    encoder = fdl.build(config)
    self.assertEqual(encoder.attention.dtype, 'complex64')
    self.assertEqual(encoder.mlp.dtype, 'complex64')
    self.assertEqual(encoder.embedders['tokens'].dtype, 'complex64')

  def test_ties_object(self):
    config = tied_config()
    object_value = object()
    config.attention.dtype = object_value
    encoder = fdl.build(config)
    self.assertIs(encoder.attention.dtype, object_value)
    self.assertIs(encoder.mlp.dtype, object_value)

  def test_works_in_lists(self):
    """TiedValue works in lists, but you need to use `.value` here."""
    config = [tied_value.new(1)] * 2 + [tied_value.new(2)]
    config[0].value = 3
    self.assertEqual(fdl.build(config), [3, 3, 2])


if __name__ == '__main__':
  absltest.main()
