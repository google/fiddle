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

"""Tests for no_custom_objects."""

import dataclasses
import typing

from absl.testing import absltest
import fiddle as fdl
from fiddle import daglish
from fiddle._src.testing.example import fake_encoder_decoder
from fiddle._src.validation import no_custom_objects


def foo(x):
  return x


class MyNamedtuple(typing.NamedTuple):
  a: int
  b: str


@dataclasses.dataclass(frozen=True)
class MyDataclass:
  a: int
  b: str


class NoCustomObjectsTest(absltest.TestCase):

  def test_empty_history(self):
    self.assertEqual(
        no_custom_objects._concise_history(entries=[]), "(no history)"
    )

  def test_concise_history(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    config.encoder.attention.dtype = "float64"
    with self.subTest("overridden"):
      history = no_custom_objects._concise_history(
          entries=config.encoder.attention.__argument_history__["dtype"]
      )
      self.assertRegex(history, r"Set in .+:\d+:test_concise_history")
    with self.subTest("not_overridden"):
      history = no_custom_objects._concise_history(
          entries=config.encoder.__argument_history__["attention"]
      )
      self.assertRegex(history, r"Set in .+fake_encoder.+:\d+:fixture")
    with self.subTest("deleted"):
      del config.encoder.attention.dtype
      history = no_custom_objects._concise_history(
          entries=config.encoder.attention.__argument_history__["dtype"]
      )
      self.assertRegex(history, r"Deleted in .+:\d+:test_concise_history")

  def test_get_history_from_state(self):
    config = fdl.Config(foo, {"a": {"b": 1}})
    traversal = daglish.MemoizedTraversal(NotImplemented, config)  # pytype: disable=wrong-arg-types
    state = traversal.initial_state()
    state = daglish.State(
        state.traversal,
        (*state.current_path, daglish.Attr("x")),
        config.x,
        state,
    )
    state = daglish.State(
        state.traversal,
        (*state.current_path, daglish.Key("a")),
        config.x["a"],
        state,
    )
    history1 = no_custom_objects._get_history_from_state(state=state)
    state = daglish.State(
        state.traversal,
        (*state.current_path, daglish.Key("b")),
        config.x["a"]["b"],
        state,
    )
    history2 = no_custom_objects._get_history_from_state(state=state)
    self.assertNotEmpty(history2)
    self.assertIs(
        history1,
        history2,
        msg=(
            "Since dictionaries do not have state, _get_history_from_state"
            " should traverse to an ancestor to find the history."
        ),
    )

  def test_get_config_errors_empty(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    self.assertEmpty(no_custom_objects.get_config_errors(config=config))

  def test_get_config_errors_namedtuple(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    config.encoder.attention = MyNamedtuple(1, "a")
    errors = no_custom_objects.get_config_errors(config=config)
    self.assertLen(errors, 1)
    self.assertRegex(
        errors[0],
        r"Found.*namedtuple.*at \.encoder\.attention.*Set"
        r" in.*:\d+:test_get_config_errors_namedtuple",
    )

  def test_get_config_errors_dataclass(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    config.encoder.attention = MyDataclass(1, "a")
    errors = no_custom_objects.get_config_errors(config=config)
    self.assertLen(errors, 1)
    self.assertRegex(
        errors[0],
        r"Found.*dataclass.*at \.encoder\.attention.*Set"
        r" in.*:\d+:test_get_config_errors_dataclass",
    )

  def test_get_config_errors_not_empty(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    config.encoder.attention = object()
    errors = no_custom_objects.get_config_errors(config=config)
    self.assertLen(errors, 1)
    self.assertRegex(
        errors[0],
        r"Found.*object.*at \.encoder\.attention.*Set"
        r" in.*:\d+:test_get_config_errors_not_empty",
    )

  def test_check_no_custom_objects_okay(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    no_custom_objects.check_no_custom_objects(config)

  def test_check_no_custom_objects_error(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    config.encoder.attention = object()
    config.decoder.self_attention = object()
    with self.assertRaisesRegex(
        ValueError,
        r"Custom objects were found.*Custom objects:\n  Found.*object.*at"
        r" \.encoder\.attention, Set in"
        r" .*:\d+:test_check_no_custom_objects_error\n  Found.*object.*at"
        r" \.decoder\.self_attention, Set in"
        r" .*:\d+:test_check_no_custom_objects_error",
    ):
      no_custom_objects.check_no_custom_objects(config=config)

  def test_no_history_custom_objects_error(self):
    config = {
        "encoder_attention": object(),
        "decoder_self_attention": object(),
    }
    with self.assertRaisesRegex(
        ValueError,
        r"Custom objects were found.*Custom objects:"
        r"\n  Found.*object.*at \['encoder_attention'\],.*no history.*"
        r"\n  Found.*object.*at \['decoder_self_attention'\],.*no history.*",
    ):
      no_custom_objects.check_no_custom_objects(config=config)


if __name__ == "__main__":
  absltest.main()
