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

"""Tests for selectors."""

import copy
import dataclasses
import typing
from typing import Any, List

from absl.testing import absltest
import fiddle as fdl
from fiddle.experimental import auto_config
from fiddle.experimental import selectors
from fiddle.testing.example import person_friend_toy


def fake_init_fn(rng, shape, dtype):
  del rng, shape, dtype  # unused
  return 1234


@dataclasses.dataclass
class Attention:
  dtype: Any
  kernel_init: Any
  bias_init: Any


@dataclasses.dataclass
class CrossAttention(Attention):
  """Example subclass."""


@dataclasses.dataclass
class Mlp:
  dtype: Any
  use_bias: bool


# In real models there are multiple layers, but pretend there is just one for
# this test.
@dataclasses.dataclass
class FakeEncoder:
  attention: Attention
  mlp: Mlp


@dataclasses.dataclass
class FakeDecoder:
  self_attention: Attention
  encoder_decoder_attention: Attention
  mlp: Mlp


@dataclasses.dataclass
class FakeEncoderDecoder:
  encoder: FakeEncoder
  decoder: FakeDecoder


class AnyInitializerTag(fdl.Tag):
  """Base tag class."""


class KernelInitializerTag(AnyInitializerTag):
  """Sub-tag class for the kernel initializer."""


class BiasInitializerTag(AnyInitializerTag):
  """Sub-tag class for the bias initializer."""


def fcn(name, args):
  return {name: args}


def fcn2(name, args):
  return {name: args}


@auto_config.auto_config
def fixture():
  shared = fcn2("shared", ())
  a = fcn("a", shared)
  b = fcn("b", shared)
  c = fcn2("c", ())
  return fcn("root", [a, b, c])


def encoder_decoder_config() -> fdl.Config[FakeEncoderDecoder]:
  # This config node would usually not be shared, but is here so that we can
  # test how seen nodes are only visited once for efficiency.
  bias_init = fdl.Partial(fake_init_fn, None, None, None)
  encoder_cfg = fdl.Config(
      FakeEncoder,
      attention=fdl.Config(Attention, "float32", "kernel1", bias_init),
      mlp=fdl.Config(Mlp, "float32", False),
  )
  decoder_cfg = fdl.Config(
      FakeDecoder,
      self_attention=fdl.Config(Attention, "float32", "kernel1", bias_init),
      encoder_decoder_attention=fdl.Config(CrossAttention, "float32", "kernel2",
                                           "bias2"),
      mlp=fdl.Config(Mlp, "float32", False),
  )
  return fdl.Config(FakeEncoderDecoder, encoder_cfg, decoder_cfg)


# The next few classes are similar to the example from the colabs, but without
# Flax.


@dataclasses.dataclass
class AddRange:
  start: int
  stop: int
  dtype: Any

  def __call__(self, x):
    return [a + b for a, b in zip(x, range(self.start, self.stop))]


@dataclasses.dataclass
class AddConstant:
  value: Any
  dtype: Any

  def __call__(self, x):
    return x + self.value


@dataclasses.dataclass
class Sequential:
  submodules: List[Any]

  def __call__(self, x):
    for module in self.submodules:
      x = module(x)
    return x


class ActivationDType(fdl.Tag):
  """The requested data-type for module outputs."""


def colab_example_base_config() -> fdl.Config[Sequential]:
  add_range = fdl.Config(AddRange, 0, 4, ActivationDType.new(default="float32"))
  add_const = fdl.Config(AddConstant, 1, ActivationDType.new(default="float32"))
  return fdl.Config(Sequential, submodules=[add_range, add_const])


class SelectionTest(absltest.TestCase):

  def test_unsupported_select_arg_combinations(self):
    cfg = encoder_decoder_config()
    with self.assertRaises(NotImplementedError):
      selectors.select(cfg, fn_or_cls=Attention, tag=ActivationDType)
    with self.assertRaises(NotImplementedError):
      selectors.select(cfg, tag=ActivationDType, match_subclasses=False)


class NodeSelectionTest(absltest.TestCase):

  def test_matches_everything(self):
    cfg = encoder_decoder_config()
    sel = typing.cast(selectors.NodeSelection, selectors.select(cfg))
    self.assertTrue(sel._matches(cfg.encoder))
    self.assertTrue(sel._matches(cfg.encoder.attention))
    self.assertTrue(sel._matches(cfg.encoder.mlp))
    self.assertTrue(sel._matches(cfg.decoder))
    self.assertTrue(sel._matches(cfg.decoder.self_attention))
    self.assertTrue(sel._matches(cfg.decoder.encoder_decoder_attention))
    self.assertTrue(sel._matches(cfg.decoder.mlp))

  def test_matches_based_on_type(self):
    cfg = encoder_decoder_config()
    sel = typing.cast(selectors.NodeSelection, selectors.select(cfg, Attention))
    self.assertFalse(sel._matches(cfg.encoder))
    self.assertTrue(sel._matches(cfg.encoder.attention))
    self.assertFalse(sel._matches(cfg.encoder.mlp))

    # Matches a subclass.
    self.assertTrue(sel._matches(cfg.decoder.encoder_decoder_attention))

  def test_matches_function_call(self):
    cfg = encoder_decoder_config()
    for cfg_node in selectors.select(cfg, fake_init_fn):
      self.assertIs(cfg_node.__fn_or_cls__, fake_init_fn)

  def test___iter__(self):
    cfg = encoder_decoder_config()

    # There are 3 attention instances, 1 for the encoder and 2 for the decoder.
    self.assertLen(list(selectors.select(cfg, Attention)), 3)

    # There are 2 attention instances that are exactly the Attention class.
    self.assertLen(
        list(selectors.select(cfg, Attention, match_subclasses=False)), 2)

    # There is only one cross-attention instance.
    self.assertLen(list(selectors.select(cfg, CrossAttention)), 1)

    # The shared kernel init node is only visited once.
    self.assertLen(list(selectors.select(cfg, fake_init_fn)), 1)

  def test_replace_fails_if_replacing_whole_tree(self):
    config = fixture.as_buildable()
    with self.assertRaisesRegex(ValueError, "replace.*not supported.*root"):
      selectors.select(config, fcn).replace("replaced")

  def test_replace_shallow_copy(self):
    config = fixture.as_buildable()
    root_name_history = copy.deepcopy(config.__argument_history__["name"])
    selectors.select(config, fcn2).replace(["replaced"], deepcopy=False)
    built = fdl.build(config)
    self.assertEqual(
        built,
        {"root": [
            {
                "a": ["replaced"]
            },
            {
                "b": ["replaced"]
            },
            ["replaced"],
        ]})
    self.assertIs(built["root"][0]["a"], built["root"][1]["b"])
    self.assertIs(built["root"][0]["a"], built["root"][2])

    # Check that history for root buildable is maintained.
    self.assertEqual(root_name_history, config.__argument_history__["name"])

    # Note: currently the replaced argument history is not updated, because this
    # is not yet supported by the traversal API. In the future, we should modify
    # config.args[0].__argument_history__['args'].

  def test_replace_deep_copy(self):
    config = fixture.as_buildable()
    replacement = ["replaced"]
    selectors.select(config, fcn2).replace(replacement)
    built = fdl.build(config)
    self.assertEqual(
        built,
        {"root": [
            {
                "a": ["replaced"]
            },
            {
                "b": ["replaced"]
            },
            ["replaced"],
        ]})

    # Fields that were aliased in the old configuration will remain aliased.
    self.assertIs(built["root"][0]["a"], built["root"][1]["b"])

    # Fields that were not aliased will get deepcopied values.
    self.assertIsNot(built["root"][0]["a"], built["root"][2])

    self.assertIsNot(built["root"][0]["a"], replacement)
    self.assertIsNot(built["root"][2], replacement)

  def test_replace_sharing(self):
    shared_f = fdl.Config(fcn, "shared", [1, fdl.Config(fcn2, "child", [1, 2])])
    cfg = fdl.Config(fcn, "root", [shared_f, shared_f])
    selectors.select(cfg, fcn2).replace(None, deepcopy=True)
    self.assertIs(cfg.args[0], cfg.args[1])

  def test_replace_doesnt_unlink(self):
    config = person_friend_toy.build_example.as_buildable()
    self.assertIs(config.child.friend, config.friend.friend)
    selectors.select(config.friend, person_friend_toy.Toy).replace(None)
    self.assertIs(config.child.friend, config.friend.friend)

  def test_set(self):
    cfg = encoder_decoder_config()
    selectors.select(cfg, Attention).set(
        dtype="override_dtype", kernel_init="override_init")
    self.assertEqual(cfg.encoder.attention.dtype, "override_dtype")
    self.assertEqual(cfg.decoder.self_attention.dtype, "override_dtype")
    self.assertEqual(cfg.encoder.attention.kernel_init, "override_init")
    self.assertEqual(cfg.decoder.self_attention.kernel_init, "override_init")

  def test_debug_get(self):
    cfg = encoder_decoder_config()
    attention_kernels = list(
        selectors.select(cfg, Attention).get("kernel_init"))
    self.assertCountEqual(["kernel1", "kernel1", "kernel2"], attention_kernels)


class TagSelectionTest(absltest.TestCase):

  def test_iter_values(self):
    config = encoder_decoder_config()
    fdl.add_tag(config.encoder.attention, "kernel_init", KernelInitializerTag)
    config.encoder.attention.kernel_init = "test_init_fn"
    self.assertEqual(
        list(selectors.select(config, tag=KernelInitializerTag)),
        ["test_init_fn"])
    self.assertEqual(
        list(selectors.select(config, tag=AnyInitializerTag)), ["test_init_fn"])

  def test_select_tag_subclasses(self):
    config = fdl.Config(Attention, dtype="float32")
    fdl.add_tag(config, "kernel_init", KernelInitializerTag)
    fdl.add_tag(config, "bias_init", BiasInitializerTag)

    selectors.select(config, tag=KernelInitializerTag).replace(value=4)
    selectors.select(config, tag=BiasInitializerTag).replace(value=1)
    self.assertEqual(fdl.build(config), Attention("float32", 4, 1))
    self.assertEqual(
        list(selectors.select(config, tag=KernelInitializerTag)), [4])
    self.assertEqual(
        list(selectors.select(config, tag=BiasInitializerTag)), [1])

    selectors.select(config, tag=AnyInitializerTag).replace(value=2)
    self.assertEqual(fdl.build(config), Attention("float32", 2, 2))

  def test_colab_example(self):

    class FakeInt32:
      pass

    cfg = colab_example_base_config()
    selectors.select(cfg, tag=ActivationDType).replace(value=FakeInt32())

  def test_replace_doesnt_infinitely_recurse(self):
    """Tests that replace() works when the tag is in the replacement."""
    config = fdl.Config(Attention, dtype="float32")
    fdl.add_tag(config, "kernel_init", KernelInitializerTag)
    config.bias_init = 2

    selectors.select(
        config, tag=KernelInitializerTag).replace(
            value=KernelInitializerTag.new("actual_value"))
    self.assertEqual(fdl.build(config), Attention("float32", "actual_value", 2))


if __name__ == "__main__":
  absltest.main()
