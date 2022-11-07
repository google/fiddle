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

"""Tests for configurator."""

from typing import Union

from absl.testing import absltest
import fiddle as fdl
from fiddle.experimental import configurator
from fiddle.testing import test_util
from fiddle.testing.example import fake_encoder_decoder
from pytype.tests import test_base as pytype_test_base

C = configurator.configurator


def encoder_decoder_fixture():
  dtype = "float32"
  kernel_init = "uniform()"
  bias_init = "zeros()"
  shared_token_embedder = C(fake_encoder_decoder.TokenEmbedder)(dtype)
  return C(fake_encoder_decoder.FakeEncoderDecoder)(
      encoder=C(fake_encoder_decoder.FakeEncoder)(
          embedders={
              "tokens": shared_token_embedder,
              "position": None
          },
          attention=C(fake_encoder_decoder.Attention)(dtype, kernel_init,
                                                      bias_init),
          mlp=C(fake_encoder_decoder.Mlp)(dtype, False,
                                          ["embed", "num_heads", "head_dim"]),
      ),
      decoder=C(fake_encoder_decoder.FakeDecoder)(
          embedders={
              "tokens": shared_token_embedder
          },
          self_attention=C(fake_encoder_decoder.Attention)(dtype, kernel_init,
                                                           bias_init),
          encoder_decoder_attention=C(fake_encoder_decoder.CrossAttention)(
              dtype, kernel_init, bias_init),
          mlp=C(fake_encoder_decoder.Mlp)(dtype, False,
                                          ["num_heads", "head_dim", "embed"]),
      ))


class ConfiguratorTest(test_util.TestCase):

  def test_basic_usage(self):
    config = encoder_decoder_fixture()
    self.assertDagEqual(config, fake_encoder_decoder.fixture.as_buildable())

  def test_as_actual_type(self):
    # This test mostly exists to ensure that there are no pytype errors.

    def fixture2() -> fdl.Config[fake_encoder_decoder.FakeEncoderDecoder]:
      return C.cast_to_config(encoder_decoder_fixture())

    self.assertIsNotNone(fixture2())

  def test_as_permissive_type(self):
    # This test mostly exists to ensure that there are no pytype errors.

    def fixture3(
    ) -> Union[fake_encoder_decoder.FakeEncoderDecoder,
               fdl.Config[fake_encoder_decoder.FakeEncoderDecoder]]:
      return C.cast_to_permissive_type(encoder_decoder_fixture())

    def foo(encoder_decoder: fake_encoder_decoder.FakeEncoderDecoder):
      return encoder_decoder

    self.assertIsNotNone(fixture3())
    self.assertIsNotNone(foo(fixture3()))


class _PytypeTest(pytype_test_base.BaseTest):

  def test_type_error(self):
    # It's a bit difficult to make imports work with PyType during unit testing,
    # so here we just test a skeleton implementation of the code, to make sure
    # arguments are type-checked.

    self.CheckWithErrors("""
    from typing import cast, TypeVar
    T = TypeVar("T")

    class _Configurator:
      def __call__(self, fn_or_cls: T) -> T:
        def inner(*args, **kwargs):
          return (fn_or_cls, args, kwargs)
        return cast(T, inner)

    C = _Configurator()

    def foo(x):
      return x

    C(foo)(x="foo")
    C(foo)(y="bar")  # wrong-keyword-args
    """)


if __name__ == "__main__":
  absltest.main()
