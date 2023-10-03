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

"""Sample binary that uses our flags reincarnation."""

import sys

from absl import app
import fiddle as fdl
from fiddle import absl_flags as fdl_flags
from fiddle import daglish
from fiddle._src.testing.example import fake_encoder_decoder


_SAMPLE_FLAG = fdl_flags.DEFINE_fiddle_config(
    "sample_config",
    help_string="Sample binary config",
    default_module=sys.modules[__name__],
)


def base_experiment() -> fdl.Config:
  return fake_encoder_decoder.fixture.as_buildable()


def set_dtypes(config, dtype: str):
  def traverse(value, state):
    if state.current_path and state.current_path[-1] == daglish.Attr("dtype"):
      return dtype
    return state.map_children(value)

  return daglish.MemoizedTraversal.run(traverse, config)




def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  print("Config after applying all the flags: ", _SAMPLE_FLAG.value)


if __name__ == "__main__":
  app.run(main)
