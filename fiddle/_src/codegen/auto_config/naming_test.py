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

"""Tests for naming."""

import dataclasses
from typing import Dict

from absl.testing import absltest
import fiddle as fdl
from fiddle import daglish
from fiddle._src.codegen import namespace as namespace_lib
from fiddle._src.codegen.auto_config import naming
from fiddle._src.testing.example import fake_encoder_decoder


@dataclasses.dataclass(frozen=True)
class Qux:
  pass


@dataclasses.dataclass(frozen=True)
class Foo:
  a: int = 0
  bar: Dict[str, Qux] = dataclasses.field(default_factory=dict)


def new_path_first_namer():
  return naming.PathFirstNamer(namespace_lib.Namespace())


def new_type_first_namer():
  return naming.TypeFirstNamer(namespace_lib.Namespace())


class NamingTest(absltest.TestCase):

  def test_suffix_is_short_for_large_configs(self):
    path = (daglish.Attr("encoder"), daglish.Attr("encoder_layer"),
            daglish.Attr("attention"), daglish.Attr("qkv_projection"))
    self.assertEqual(naming.suffix_first_path(path), "qkv_projection")

  def test_suffix_first_path_indices(self):
    path = (daglish.Attr("foo"), daglish.Index(0), daglish.Index(1))
    self.assertEqual(naming.suffix_first_path(path), "foo_0_1")

  def test_suffix_first_path_keys(self):
    # NOTE: It would be reasonable to change this to be "foo_0_bar", if that
    # longer name is more relevant/helpful.
    path = (daglish.Attr("foo"), daglish.Index(0), daglish.Key("bar"))
    self.assertEqual(naming.suffix_first_path(path), "foo_0_bar")


class NamerTest(absltest.TestCase):

  def test_name_from_candidates(self):
    namer = new_path_first_namer()
    self.assertEqual(namer.name_from_candidates(["foo"]), "foo")
    self.assertEqual(namer.name_from_candidates(["foo"]), "foo_2")
    self.assertEqual(namer.name_from_candidates(["foo"]), "foo_3")


class PathFirstNamerTest(absltest.TestCase):

  def test_name_for_doesnt_add_root_unnecessarily(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    namer = new_path_first_namer()
    self.assertEqual(namer.name_for(config, []), "fake_encoder_decoder")
    # The logic that adds "root" is kind of internal, but if it had been added,
    # then that name would have been preferred to the "_2" suffixed version.
    # See below for a case that has multiple candidate names and falls back on
    # them before adding suffixes.
    self.assertEqual(namer.name_for(config, []), "fake_encoder_decoder_2")

  def test_name_for_falls_back_to_another_path(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    namer = new_path_first_namer()
    paths = [
        (daglish.Attr("foo"), daglish.Attr("Bar")),
        (daglish.Attr("baz"), daglish.Attr("qux")),
    ]
    self.assertEqual(namer.name_for(config, paths), "bar")
    self.assertEqual(namer.name_for(config, paths), "qux")

  def test_name_for_falls_back_to_root(self):
    config = [{"hi": 3}]
    namer = new_path_first_namer()
    self.assertEqual(namer.name_for(config, []), "root")
    self.assertEqual(namer.name_for(config, []), "root_2")

  def test_name_for_encoder(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    namer = new_path_first_namer()
    name = namer.name_for(config.encoder.embedders["tokens"],
                          [(daglish.Attr("encoder"), daglish.Attr("embedders"),
                            daglish.Key("tokens"))])
    self.assertEqual(name, "embedders_tokens")

  def test_can_get_names_for_every_element(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    namer = new_path_first_namer()
    name_set = set()
    num_values = 0
    for value, path in daglish.iterate(config):
      name_set.add(namer.name_for(value, [path]))
      num_values += 1
    self.assertLen(name_set, num_values)

  def test_names_as_documented_example_1(self):
    config = fdl.Config(Foo, bar={"qux": fdl.Config(Qux)})  # pytype: disable=wrong-arg-types  # use-fiddle-overlay
    namer = new_path_first_namer()
    names = {
        daglish.path_str(path): namer.name_for(value, [path])
        for value, path in daglish.iterate(config)
    }
    self.assertDictEqual(names, {
        "": "foo",
        ".bar": "bar",
        ".bar['qux']": "bar_qux"
    })

  def test_names_as_documented_example_2(self):
    config = [{1: "hi"}]
    namer = new_path_first_namer()
    self.assertEqual(
        namer.name_for(
            config[0][1],
            [(
                daglish.Index(0),
                daglish.Key(1),
            )],
        ),
        "unnamed_var",
    )
    self.assertEqual(
        namer.name_for(config[0], [(daglish.Index(0),)]), "unnamed_var_2"
    )
    self.assertEqual(namer.name_for(config, [()]), "root")


class TypeFirstNamerTest(absltest.TestCase):

  def test_name_for_doesnt_add_root_unnecessarily(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    namer = new_type_first_namer()
    self.assertEqual(namer.name_for(config, []), "fake_encoder_decoder")
    # The logic that adds "root" is kind of internal, but if it had been added,
    # then that name would have been preferred to the "_2" suffixed version.
    # See below for a case that has multiple candidate names and falls back on
    # them before adding suffixes.
    self.assertEqual(namer.name_for(config, []), "fake_encoder_decoder_2")

  def test_name_for_falls_back_to_another_path(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    namer = new_type_first_namer()
    paths = [
        (daglish.Attr("foo"), daglish.Attr("Bar")),
        (daglish.Attr("baz"), daglish.Attr("qux")),
    ]
    # N.B. CHANGE FROM PATH FIRST: This will try the `fake_encoder_decoder` name
    # first.
    self.assertEqual(namer.name_for(config, paths), "fake_encoder_decoder")
    self.assertEqual(namer.name_for(config, paths), "bar")
    self.assertEqual(namer.name_for(config, paths), "qux")

    # When we run out of candidates, the first/preferred candidate gets suffixes
    # appended.
    self.assertEqual(namer.name_for(config, paths), "fake_encoder_decoder_2")

  def test_name_for_falls_back_to_root(self):
    config = [{"hi": 3}]
    namer = new_type_first_namer()
    self.assertEqual(namer.name_for(config, []), "root")
    self.assertEqual(namer.name_for(config, []), "root_2")

  def test_name_for_encoder(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    namer = new_type_first_namer()
    name = namer.name_for(
        config.encoder.embedders["tokens"],
        [(
            daglish.Attr("encoder"),
            daglish.Attr("embedders"),
            daglish.Key("tokens"),
        )],
    )
    # N.B. Change from path-first namer.
    self.assertEqual(name, "token_embedder")

  def test_can_get_names_for_every_element(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    namer = new_type_first_namer()
    name_set = set()
    num_values = 0
    for value, path in daglish.iterate(config):
      name_set.add(namer.name_for(value, [path]))
      num_values += 1
    self.assertLen(name_set, num_values)

  def test_names_as_documented_example_1(self):
    config = fdl.Config(Foo, bar={"qux": fdl.Config(Qux)})  # pytype: disable=wrong-arg-types  # use-fiddle-overlay
    namer = new_type_first_namer()
    names = {
        daglish.path_str(path): namer.name_for(value, [path])
        for value, path in daglish.iterate(config)
    }
    # N.B. Change from path-first namer: .bar['qux'] is 'qux' not 'bar_qux'.
    self.assertDictEqual(
        names, {"": "foo", ".bar": "bar", ".bar['qux']": "qux"}
    )

  def test_names_as_documented_example_2(self):
    config = [{1: "hi"}]
    namer = new_type_first_namer()
    with self.assertRaises(ValueError):
      namer.name_for(
          config[0][1],
          [(
              daglish.Index(0),
              daglish.Key(1),
          )],
      )
    with self.assertRaises(ValueError):
      namer.name_for(config[0], [(daglish.Index(0),)])
    self.assertEqual(namer.name_for(config, [()]), "root")


if __name__ == "__main__":
  absltest.main()
