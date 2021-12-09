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

"""Tests for daglish."""

from typing import List

from absl.testing import absltest
import fiddle as fdl
from fiddle.experimental import daglish


class DaglishTest(absltest.TestCase):

  def test_path_fragment(self):
    cfg = fdl.Config(lambda x: x)
    path: List[daglish.PathElement] = [
        daglish.ListOrTupleItem([4], 1),
        daglish.DictItem({}, "a"),
        daglish.AttributeItem(object(), "foo"),
        daglish.DictItem({}, 2),
        daglish.AttributeItem(cfg, "bar"),
    ]
    path_str = "".join(x.code for x in path)
    self.assertEqual(path_str, "[1]['a'].foo[2].bar")


if __name__ == "__main__":
  absltest.main()
