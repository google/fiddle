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

"""Tests for serialization."""

import pickle

from absl.testing import absltest
import fiddle as fdl
from fiddle.experimental import serialization


class Unserializable:

  def __getstate__(self):
    raise NotImplementedError()


def identity_fn(arg1):
  return arg1


class SerializationTest(absltest.TestCase):

  def test_pickling_non_serializable_history_deepcopy_default(self):
    cfg = fdl.Config(identity_fn, arg1=Unserializable())
    cfg.arg1 = 4
    serialization.clear_argument_history(cfg)
    with self.assertRaises(NotImplementedError):
      pickle.dumps(cfg)

  def test_pickling_non_serializable_history(self):
    cfg = fdl.Config(identity_fn, arg1=Unserializable())
    cfg.arg1 = 4
    cfg = serialization.clear_argument_history(cfg)
    pickle.dumps(cfg)

  def test_pickling_non_serializable_history_mutation(self):
    cfg = fdl.Config(identity_fn, arg1=Unserializable())
    cfg.arg1 = 4
    serialization.clear_argument_history(cfg, deepcopy=False)
    pickle.dumps(cfg)


if __name__ == "__main__":
  absltest.main()
