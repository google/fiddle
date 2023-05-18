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

"""Tests for grep."""

from absl.testing import absltest
import fiddle as fdl
from fiddle._src.testing.example import fake_encoder_decoder
from fiddle.debug import grep


class GrepTest(absltest.TestCase):

  def test_grep_matches_paths_or_values(self):
    config = [1, 2, 3, 4, 5]
    results = []
    grep.grep(config=config, pattern="2", output_fn=results.append)
    self.assertEqual(results, ["[1]: 2", "[2]: 3"])

  def test_grep_matches_path_and_value(self):
    """Just like grep'ing over text, it is OK to match path and value."""
    config = ["hi", "bye"]
    results = []
    grep.grep(config=config, pattern=r"1.*'bye'", output_fn=results.append)
    self.assertEqual(results, ["[1]: 'bye'"])

  def test_config_matches_fn(self):
    """Tests the output for Config nodes.

    By default, Config objects' str representation will print out all of their
    sub-values. However, this is probably way too much console spam and not
    what the user intends, when they want to match on a sub-value.

    On the other hand, it is important to match on names.
    """

    def foo(x):
      return x

    config = fdl.Config(foo, x=3)
    # Only the leaf should be shown; stuff like argument_history should not be
    # matched.
    results = []
    grep.grep(config=config, pattern="3", output_fn=results.append)
    self.assertEqual(results, [".x: 3"])

    # The function name should match. Note: Maybe we should represent the
    # root config object somehow?
    results = []
    grep.grep(config=config, pattern="foo", output_fn=results.append)
    self.assertEqual(results, [": <Config(foo)>"])

  def test_shared_matchable_by_both_paths(self):
    config = fake_encoder_decoder.fixture.as_buildable()

    results = []
    grep.grep(config=config, pattern="TokenEmb", output_fn=results.append)
    self.assertEqual(
        results,
        [
            ".encoder.embedders['tokens']: <Config(TokenEmbedder)>",
            ".decoder.embedders['tokens']: <Config(TokenEmbedder)>",
        ],
    )

    results = []
    grep.grep(config=config, pattern="enc.*emb.*tok", output_fn=results.append)
    self.assertEqual(
        results,
        [
            ".encoder.embedders['tokens'].dtype: 'float32'",
            ".encoder.embedders['tokens']: <Config(TokenEmbedder)>",
        ],
    )

    results = []
    grep.grep(config=config, pattern="dec.*emb.*tok", output_fn=results.append)
    self.assertEqual(
        results,
        [
            ".decoder.embedders['tokens'].dtype: 'float32'",
            ".decoder.embedders['tokens']: <Config(TokenEmbedder)>",
        ],
    )

  def test_case_insensitive(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    results = []
    grep.grep(config=config, pattern="tokenemb", output_fn=results.append)
    self.assertEqual(results, [])

    results = []
    grep.grep(
        config=config,
        pattern="tokenemb",
        case_sensitive=False,
        output_fn=results.append,
    )
    self.assertEqual(
        results,
        [
            ".encoder.embedders['tokens']: <Config(TokenEmbedder)>",
            ".decoder.embedders['tokens']: <Config(TokenEmbedder)>",
        ],
    )

  def test_excluding_path(self):
    config = fake_encoder_decoder.fixture.as_buildable()

    results = []
    grep.grep(
        config=config,
        pattern="TokenEmb",
        exclude="enc.*emb.*tok.*:",
        output_fn=results.append,
    )
    self.assertEqual(
        results,
        [".decoder.embedders['tokens']: <Config(TokenEmbedder)>"],
    )

  def test_excluding_case_insensitive(self):
    config = fake_encoder_decoder.fixture.as_buildable()

    results = []
    grep.grep(
        config=config,
        pattern=r"encoder.*\(",
        exclude="tokenemb",
        case_sensitive=False,
        output_fn=results.append,
    )
    self.assertEqual(
        results,
        [
            ".encoder.attention.kernel_init: 'uniform()'",
            ".encoder.attention.bias_init: 'zeros()'",
            ".encoder.attention: <Config(Attention)>",
            ".encoder.mlp: <Config(Mlp)>",
            ".encoder: <Config(FakeEncoder)>",
            ".decoder.encoder_decoder_attention.kernel_init: 'uniform()'",
            ".decoder.encoder_decoder_attention.bias_init: 'zeros()'",
            ".decoder.encoder_decoder_attention: <Config(CrossAttention)>",
        ],
    )


if __name__ == "__main__":
  absltest.main()
