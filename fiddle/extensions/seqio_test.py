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

"""Tests for fiddle.extensions.seqio."""

from typing import List

from absl.testing import absltest
import fiddle as fdl
from fiddle.codegen import codegen
import fiddle.extensions.seqio
import seqio


def tokens(code: str) -> List[str]:
  return code.strip().split()


class SeqioTest(absltest.TestCase):

  def test_codegen(self):
    fiddle.extensions.seqio.enable()
    cfg = fdl.Config(seqio.Evaluator)
    code = "\n".join(codegen.codegen_dot_syntax(cfg).lines())
    expected = """
import fiddle as fdl
import seqio


def build_config():
  root = fdl.Config(seqio.Evaluator)

  return root
    """
    self.assertEqual(tokens(code), tokens(expected))


if __name__ == "__main__":
  absltest.main()
