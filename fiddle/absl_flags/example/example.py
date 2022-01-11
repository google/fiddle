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

r"""An example demonstrating Fiddle and absl_flags.

Run this example with the following command:

```sh
python3 -m fiddle.absl_flags.example.example \
  --fdl_config=simple \
  --fiddler=swap_weight_and_bias \
  --fdl.model.b=0.73
  --fdl.data.filename='"other.txt"'  # Alt syntax: --fdl.data.filename=\"b.txt\"
```
"""

from typing import Sequence

from absl import app
import fiddle as fdl
from fiddle import absl_flags
from fiddle import printing
from fiddle.absl_flags.example import configs


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise ValueError(f'Unexpected CLI arguments: {argv!r}')
  cfg = absl_flags.create_buildable_from_flags(configs)
  print(printing.as_str_flattened(cfg))
  runner = fdl.build(cfg)
  runner.run()


if __name__ == '__main__':
  app.run(main, flags_parser=absl_flags.flags_parser)
