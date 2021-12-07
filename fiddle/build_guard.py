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

"""Internal utility to ensure there are no nested `fdl.build` calls."""

import contextlib
import threading


class _BuildGuardState(threading.local):

  def __init__(self):
    super().__init__()
    self.in_build = False


state = _BuildGuardState()


@contextlib.contextmanager
def in_build():
  """A context manager to ensure fdl.build is not called recursively."""
  if state.in_build:
    raise ValueError(
        'It is forbidden to call `fdl.build` inside another `fdl.build` call.')
  state.in_build = True
  try:
    yield
  finally:
    state.in_build = False
