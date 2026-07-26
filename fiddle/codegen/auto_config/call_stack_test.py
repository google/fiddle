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

"""Tests for call_stack."""

import itertools

from absl.testing import absltest
import fiddle as fdl
from fiddle import daglish
from fiddle.codegen.auto_config import call_stack
from fiddle.codegen.auto_config import ir_dag
from fiddle.testing.example import fake_encoder_decoder


class CallStackTest(absltest.TestCase):

  def test_most_specific_common_call(self):
    fn = ir_dag.FixtureFunction("foo", return_type=fdl.Config)
    fn2 = ir_dag.FixtureFunction("bar", return_type=str)
    call1 = ir_dag.Call(fn)

    # bar() is called twice in foo(). These should be distinct.
    call2 = ir_dag.Call(fn2, parent=call1)
    call3 = ir_dag.Call(fn2, parent=call1)

    self.assertIs(
        call_stack.most_specific_common_call([call1, call2, call3]), call1)

    for x in [call1, call2, call3]:
      self.assertIs(call_stack.most_specific_common_call([x, x]), x)

    # call2 and call3 should be object-distinct, resulting in call1 being the
    # most specific shared call.
    self.assertIs(call_stack.most_specific_common_call([call2, call3]), call1)


class SpecificValueFixturesTest(absltest.TestCase):

  def test_encoder_decoder_fixtures(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    fixture_assignment = call_stack.SpecificValueFixtures({
        id(config.encoder): "encoder_fixture",
        id(config.decoder): "decoder_fixture",
    })
    ir = ir_dag.generate_ir_dag(config)
    fixture_assignment.assign(ir)

    call_names = {}  # Call --> str name mapping.

    def name_call(call: ir_dag.Call) -> str:
      if call in call_names:
        return call_names[call]
      for idx in itertools.count(1):
        name = f"{call.fn.name}_{idx}"
        if name not in call_names.values():
          call_names[call] = name
          return name
      assert False  # unreachable

    path_to_calls = {
        daglish.path_str(path): name_call(value.call)
        for value, path in daglish.iterate(ir, memoized=False)
    }
    self.assertEqual(path_to_calls[""], "config_fixture_1")
    self.assertEqual(path_to_calls[".encoder"], "encoder_fixture_1")
    self.assertEqual(path_to_calls[".decoder"], "decoder_fixture_1")

    # This shared node must appear in the parent.
    self.assertEqual(path_to_calls[".encoder.embedders['tokens']"],
                     "config_fixture_1")


if __name__ == "__main__":
  absltest.main()
