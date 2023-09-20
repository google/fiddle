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
from typing import Any, List, Optional, Union
import unittest

from absl.testing import absltest
from fiddle._src import building
from fiddle._src import config
from fiddle._src.testing.example import fake_encoder_decoder
from fiddle._src.validation import check_types


class AbstractDataset:

  def it(self):
    yield 0


class MyDataset(AbstractDataset):

  def it(self):
    yield 1


class BadDataset:

  def func(self):
    yield 2


class Experiment:

  def __init__(self, dataset: AbstractDataset):
    self.dataset = dataset

  def run(self):
    data = next(self.dataset.it())
    print(data)


@dataclasses.dataclass
class BadTokenEmbedder:
  not_a_dtype: Any


@dataclasses.dataclass
class UnionAndOptionalHolder:
  union_field: Union[str, fake_encoder_decoder.TokenEmbedder]
  optional_field: Optional[str]


@dataclasses.dataclass
class StackedEncoder:
  encoders: List[fake_encoder_decoder.FakeEncoder]


class CheckTypesTest(absltest.TestCase, unittest.TestCase):

  def test_type_validation(self):
    cfg = config.Config(Experiment, dataset=BadDataset())

    # Without any type validation check, building a cfg object with an arg value
    # whose type is different from the annotated type succeeds:
    building.build(cfg)

    # However, with type validation, this should fail:
    with self.assertRaisesRegex(
        TypeError,
        ".*For attribute.*dataset.*.*BadDataset.*is not of"
        " annotated/declared type.*AbstractDataset.*",
    ):
      check_types.check_types(cfg)

    # And passing in the right type (or derived type) should succeed
    cfg = config.Config(Experiment, dataset=MyDataset())
    check_types.check_types(cfg)

  def test_type_validation_fails_for_auto_config_decorators(self):
    cfg = fake_encoder_decoder.fixture.as_buildable()
    check_types.check_types(cfg)

  def test_type_validation_leaves_not_checked(self):
    # Declare a config, which has a non-`Buildable` leaf node `embedders`, which
    # is a container that has elements in it of the wrong type:
    cfg = config.Config(
        fake_encoder_decoder.FakeEncoderDecoder,
        encoder=config.Config(
            fake_encoder_decoder.FakeEncoder,
            embedders={
                "tokens": BadTokenEmbedder("NotADType"),  # Incorrect type
                "position": None,
            },
            attention=fake_encoder_decoder.Attention(
                "float32", "uniform()", "zeros()"
            ),
            mlp=fake_encoder_decoder.Mlp(
                "float32", False, ["embed", "num_heads", "head_dim"]
            ),
        ),
        decoder=config.Config(
            fake_encoder_decoder.FakeDecoder,
            embedders={},
            self_attention=fake_encoder_decoder.Attention(
                "float32", "uniform()", "zeros()"
            ),
            encoder_decoder_attention=fake_encoder_decoder.CrossAttention(
                "float32", "uniform()", "zeros()"
            ),
            mlp=fake_encoder_decoder.Mlp(
                "float32", False, ["embed", "num_heads", "head_dim"]
            ),
        ),
    )
    # check_types() doesn't check types of non-`Buildable` leaf nodes, so a call
    # to it should succeed in this case:
    check_types.check_types(cfg)

  def test_type_validation_invalid_non_leaf_node(self):
    # Declare a config that has an incorrect arg type for a non-leaf `Buildable`
    # node's argument values:
    cfg = config.Config(
        fake_encoder_decoder.FakeEncoderDecoder,
        encoder=config.Config(
            fake_encoder_decoder.FakeEncoder,
            embedders={
                "tokens": fake_encoder_decoder.TokenEmbedder("float32"),
                "position": None,
            },
            attention=fake_encoder_decoder.Attention(
                "float32", "uniform()", "zeros()"
            ),
            mlp=fake_encoder_decoder.Mlp(
                "float32", False, ["embed", "num_heads", "head_dim"]
            ),
        ),
        decoder=config.Config(
            fake_encoder_decoder.FakeDecoder,
            embedders={},
            self_attention=MyDataset(),  # Incorrect type passed
            encoder_decoder_attention=fake_encoder_decoder.CrossAttention(
                "float32", "uniform()", "zeros()"
            ),
            mlp=BadDataset(),  # Incorrect type passed
        ),
    )
    # Type validation should fail and detect it correctly:
    with self.assertRaisesRegex(
        TypeError,
        ".*[\n].*For attribute.*self_attention.*MyDataset.*is not of"
        " annotated/declared type.*Attention.*[\n].*For"
        " attribute.*mlp.*BadDataset.*is not of annotated/declared type.*Mlp.*",
    ):
      check_types.check_types(cfg)

  @unittest.skipIf(
      sys.version_info < (3, 10),
      "Union and optional checks are supported using isinstance() for python"
      " 3.10 and above",
  )
  def test_union_and_optional_types(self):
    cfg = config.Config(
        UnionAndOptionalHolder,
        union_field=fake_encoder_decoder.TokenEmbedder("float32"),
        optional_field="float32",
    )
    check_types.check_types(cfg)

    cfg = config.Config(
        UnionAndOptionalHolder,
        union_field="float32",
        optional_field="float32",
    )
    check_types.check_types(cfg)

    cfg = config.Config(
        UnionAndOptionalHolder,
        union_field=fake_encoder_decoder.TokenEmbedder("float32"),
        optional_field=None,
    )
    check_types.check_types(cfg)

    cfg = config.Config(
        UnionAndOptionalHolder,
        union_field=BadTokenEmbedder("float32"),
        optional_field="float32",
    )
    with self.assertRaisesRegex(
        TypeError,
        ".*For attribute .*union_field provided type:"
        " .*BadTokenEmbedder.* is not of annotated/declared"
        " type: typing.Union.*str,"
        " fiddle._src.testing.example.fake_encoder_decoder.TokenEmbedder.*",
    ):
      check_types.check_types(cfg)

  def test_subscripted_generic_types_succeeds(self):
    cfg = config.Config(
        StackedEncoder,
        encoders=[
            fake_encoder_decoder.FakeEncoder(
                embedders={},
                attention=fake_encoder_decoder.Attention(
                    "float32", "uniform()", "zeros()"
                ),
                mlp=fake_encoder_decoder.Mlp(
                    "float32", False, ["embed", "num_heads", "head_dim"]
                ),
            )
        ],
    )
    check_types.check_types(cfg)

  def test_subscripted_generic_types_invalid_leaf_nodes_not_type_checked(self):
    # Passing in incorrect types as leaf nodes inside standard containers won't
    # be type validated, and will succeed
    cfg = config.Config(
        StackedEncoder,
        encoders=[
            BadDataset(),  # incorrect type passed
            MyDataset(),  # incorrect type passed
        ],
    )
    check_types.check_types(cfg)

  def test_subscripted_generic_types_invalid_non_leaf_nodes_type_checked(self):
    # However, passing a dict for `encoders` should trigger an error, since the
    # type of `encoders` is a list:
    cfg = config.Config(
        StackedEncoder,
        encoders={
            "encoder1": fake_encoder_decoder.FakeEncoder(
                embedders={},
                attention=fake_encoder_decoder.Attention(
                    "float32", "uniform()", "zeros()"
                ),
                mlp=fake_encoder_decoder.Mlp(
                    "float32", False, ["embed", "num_heads", "head_dim"]
                ),
            )
        },
    )
    with self.assertRaisesRegex(
        TypeError,
        ".*For attribute .*encoders provided type:"
        " .*dict.* is not of annotated/declared"
        " type: typing.List.*FakeEncoder.*",
    ):
      check_types.check_types(cfg)


if __name__ == "__main__":
  unittest.main()
