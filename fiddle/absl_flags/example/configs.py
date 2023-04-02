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

"""A collection of Fiddle-based configurations and fiddlers.

The Fiddle recommended pattern is to define one or more "base" configurations
(e.g. `base` in this example). As new experiments are performed, they simply
mutate the base configuration or another experiment. (Fiddle's history tracking
lets you easily determine which experiment set what value when there's a large
stack of incremental experiments.)

> Note: This pattern often works well for iterative product launches and/or
> product versions as well.

If there are multiple "base" configurations, it's often convenient to define
them in separate Python files (modules) for organization purposes.
"""

import fiddle as fdl
from fiddle.absl_flags.example import business_logic
from fiddle.absl_flags.example import tags
from fiddle.experimental import auto_config


# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield


@auto_config.auto_config
def base() -> business_logic.BulkInferenceRunner:
  """A simple base configuration.

  Base configuration functions must be 0-arity, or have defaults for every
  parameter, and must return a `fdl.Config` (or some other `fdl.Buildable`
  subclass).

  Note: these base configurations are often written using `@auto_config` to
  make them easier to read, but they certainly do not have to.

  Best practices include checking in a link to the results of your experiment.
  Your future self (and collaborators) will thank you.

  https://link.to.tensorboard.or.similar/experiment.metadata
  """
  model = business_logic.MyLinearModel(
      w=-3.8, b=0.5, activation_dtype=tags.ActivationDtype.new('float32')
  )
  loader = business_logic.MyDataLoader(filename='data.txt')
  return business_logic.BulkInferenceRunner(model, loader)


experiment_1 = base.as_buildable


def experiment_2() -> fdl.Config[business_logic.BulkInferenceRunner]:
  """Hypothesis: increasing the w parameter to improve accuracy.

  https://link.to.tensorboard.or.similar/experiment.metadata
  """
  config = experiment_1()
  config.model.w = 27
  return config


def experiment_3() -> fdl.Config[business_logic.BulkInferenceRunner]:
  """Hypothesis: increasing the b parameter to improve precision.

  https://link.to.tensorboard.or.similar/experiment.metadata
  """
  config = experiment_2()
  config.model.b = 42
  return config


def experiment_4() -> fdl.Config[business_logic.BulkInferenceRunner]:
  """Hypothesis: load from better dataset improves f1 score.

  Note: restarting from baseline.

  https://link.to.tensorboard.or.similar/experiment.metadata
  """
  config = experiment_1()  # experiment 2 & 3 were terrible ideas.
  config.loader = 'abc'
  return config


def swap_weight_and_bias(cfg: fdl.Config[business_logic.BulkInferenceRunner]):
  """A sample fiddler."""
  tmp = cfg.model.w
  cfg.model.w = cfg.model.b
  cfg.model.b = tmp
