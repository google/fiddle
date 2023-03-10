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

"""Tests Fiddle integration with Flax.

There are currently no Flax extensions, but some iteractions that should be
integration tested.
"""

from absl.testing import absltest
import fiddle as fdl
from fiddle._src.experimental import auto_config
from flax import linen as nn


class FlaxTest(absltest.TestCase):

  def test_auto_config_classmethod(self):

    class MyClass(nn.Module):
      x: int
      y: str
      z: float = 2.0

      @auto_config.auto_config
      @classmethod
      def simple(cls):
        """Test simple docstring."""
        return cls(x=1, y="1", z=1.0)

    config = MyClass.simple.as_buildable()
    self.assertIsInstance(config, fdl.Config)


if __name__ == "__main__":
  absltest.main()
