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

"""Tests for placeholders."""

import copy

from absl.testing import absltest
import fiddle as fdl
from fiddle import placeholders


def return_kwargs(**kwargs):
  return kwargs


fine_key = placeholders.PlaceholderKey("fine_key")
test_key = placeholders.PlaceholderKey("my_test_key")


class PlaceholdersTest(absltest.TestCase):

  def test_placeholder_fn_exception(self):
    with self.assertRaisesRegex(
        placeholders.PlaceholderNotFilledError,
        "Expected.*placeholders.*replaced.*one "
        "with name 'my_test_key' was not set"):
      placeholders.placeholder_fn(key=test_key)

  def test_placeholder_fn_none_value(self):
    # Test that when the value is None, that's OK.
    self.assertIsNone(placeholders.placeholder_fn(key=test_key, value=None))

  def test_placeholder_default_none(self):
    cfg = placeholders.Placeholder(key=test_key, default=None)
    self.assertIsNone(fdl.build(cfg))

  def test_one_placeholder_unset_in_config(self):
    cfg = fdl.Config(
        return_kwargs,
        foo=placeholders.Placeholder(key=fine_key, default=None),
        bar=placeholders.Placeholder(key=test_key))
    with self.assertRaisesRegex(
        placeholders.PlaceholderNotFilledError,
        "Expected.*placeholders.*replaced.*one "
        "with name 'my_test_key' was not set"):
      fdl.build(cfg)

  def test_set_placeholder(self):
    cfg = fdl.Config(
        return_kwargs,
        foo=placeholders.Placeholder(key=fine_key, default=None),
        bar=placeholders.Placeholder(key=test_key))
    placeholders.set_placeholder(cfg, key=test_key, value=1)
    self.assertDictEqual(fdl.build(cfg), {"foo": None, "bar": 1})

  def test_set_two_placeholders(self):
    cfg = fdl.Config(
        return_kwargs,
        foo=placeholders.Placeholder(key=fine_key, default=None),
        bar=placeholders.Placeholder(key=test_key))
    placeholders.set_placeholder(cfg, key=test_key, value=1)
    placeholders.set_placeholder(cfg, key=fine_key, value=2)
    self.assertDictEqual(fdl.build(cfg), {"foo": 2, "bar": 1})

  def test_double_set_placeholder(self):
    cfg = fdl.Config(
        return_kwargs,
        foo=placeholders.Placeholder(key=fine_key, default=None),
        bar=placeholders.Placeholder(key=test_key))
    placeholders.set_placeholder(cfg, key=test_key, value=1)
    placeholders.set_placeholder(cfg, key=test_key, value=2)
    self.assertDictEqual(fdl.build(cfg), {"foo": None, "bar": 2})

  def test_set_only_placeholders_in_subtree(self):
    activation_dtype = placeholders.PlaceholderKey("activation_dtype")
    cfg = fdl.Config(
        return_kwargs,
        output_logits=fdl.Config(
            return_kwargs,
            dtype=placeholders.Placeholder(activation_dtype, "float32")),
        encoder=fdl.Config(
            return_kwargs,
            dtype=placeholders.Placeholder(activation_dtype, "float32")),
    )

    # At first, the defaults are used.
    self.assertDictEqual(
        fdl.build(cfg), {
            "encoder": {
                "dtype": "float32"
            },
            "output_logits": {
                "dtype": "float32"
            },
        })

    # Setting the dtype for the entire tree overrides both.
    placeholders.set_placeholder(cfg, activation_dtype, "bfloat16")
    self.assertDictEqual(
        fdl.build(cfg), {
            "encoder": {
                "dtype": "bfloat16"
            },
            "output_logits": {
                "dtype": "bfloat16"
            },
        })

    # Setting the dtype for the output logits just overrides that one.
    placeholders.set_placeholder(cfg.output_logits, activation_dtype, "float64")
    self.assertDictEqual(
        fdl.build(cfg), {
            "encoder": {
                "dtype": "bfloat16"
            },
            "output_logits": {
                "dtype": "float64"
            },
        })

  def test_deepcopy_and_object_id(self):
    """Tests that deepcopy() is fine, and key IDs are used, not names."""
    fine_key1 = placeholders.PlaceholderKey("fine_key")
    fine_key2 = placeholders.PlaceholderKey("fine_key")

    def foo(a, b, c, d):
      return a, b, c, d

    cfg = fdl.Config(foo)
    cfg.a = placeholders.Placeholder(fine_key1)
    cfg.b = placeholders.Placeholder(fine_key2)
    cfg.c = placeholders.Placeholder(fine_key1)
    cfg.d = 4
    copied = copy.deepcopy(cfg)
    copied.d = 40

    placeholders.set_placeholder(cfg, fine_key1, 1)
    placeholders.set_placeholder(cfg, fine_key2, 2)

    self.assertEqual(fdl.build(cfg), (1, 2, 1, 4))

    with self.assertRaises(placeholders.PlaceholderNotFilledError):
      fdl.build(copied)

    placeholders.set_placeholder(copied, fine_key1, 10)
    placeholders.set_placeholder(copied, fine_key2, 20)
    self.assertEqual(fdl.build(copied), (10, 20, 10, 40))

  def test_deepcopy_linked_value_object(self):

    def foo(a, b):
      return a, b

    shared_value = fdl.Config(foo, 1, 2)
    cfg = fdl.Config(
        foo,
        a=fdl.Placeholder(fine_key, shared_value),
        b=fdl.Placeholder(fine_key, shared_value),
    )
    copied = copy.deepcopy(cfg)
    self.assertIsNot(copied.a, cfg.a)
    self.assertIsNot(copied.a.value, cfg.a.value)
    self.assertIs(copied.a.key, cfg.a.key)
    self.assertIs(copied.b.key, cfg.b.key)
    self.assertIs(cfg.a.value, cfg.b.value)
    self.assertIs(copied.a.value, copied.b.value)

  def test_shallow_copy_placeholders(self):
    cfg = placeholders.Placeholder(fine_key)
    copied = copy.copy(cfg)

    self.assertIs(copied.key, fine_key)
    self.assertIs(copied.value, placeholders.NO_VALUE)

    with self.assertRaises(placeholders.PlaceholderNotFilledError):
      fdl.build(copied)
    placeholders.set_placeholder(cfg, fine_key, 1)
    self.assertEqual(fdl.build(cfg), 1)

    with self.assertRaises(placeholders.PlaceholderNotFilledError):
      fdl.build(copied)
    placeholders.set_placeholder(copied, fine_key, 2)
    self.assertEqual(fdl.build(cfg), 1)
    self.assertEqual(fdl.build(copied), 2)


if __name__ == "__main__":
  absltest.main()
