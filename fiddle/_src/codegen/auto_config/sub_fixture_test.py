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

"""Tests for transform_sub_fixtures pass."""

import dataclasses
import functools
from typing import Any
from absl.testing import absltest
import fiddle as fdl
from fiddle._src.codegen.auto_config import init_task
from fiddle._src.codegen.auto_config import ir_to_cst
from fiddle._src.codegen.auto_config import make_symbolic_references
from fiddle._src.codegen.auto_config import sub_fixture
from fiddle._src.codegen.auto_config import test_fixtures
from fiddle._src.testing.example import demo_configs
from fiddle._src.testing.example import fake_encoder_decoder


@dataclasses.dataclass(frozen=True)
class SampleClass:
  x: Any
  y: Any
  z: Any = None


class FindLeastCommonAncestorTest(absltest.TestCase):

  def test_shared_node_pattern(self):
    """The tested config is as follows.

                             ┌────────┐
                   ┌─────────► config ◄────────────┐
                   │         └────────┘            │
                   │                               │
                   │                               │
               ┌───┴───┐                       ┌───┴───┐
        ┌──────►   a   ◄──────┐                │   b   │
        │      └───▲───┘      │                └───▲───┘
        │          │          │                    │
        │          │          │                    │
        │          │          │                    │
    ┌───┴───┐  ┌───┴───┐  ┌───┴───┐                │
    │   f1  │  │   f2  │  │   f3  │                │
    └───▲───┘  └───▲───┘  └───▲───┘                │
        │          │          │                    │
        │          │          │       ┌────────┐   │
        └──────────┴──────────┴───────┤ shared ├───┘
                                      └────────┘
    """
    shared = fdl.Config(SampleClass, 'shared-x', 'shared-y')
    f1 = fdl.Config(SampleClass, 'f1-x', shared)
    f2 = fdl.Config(SampleClass, 'f2-x', shared)
    f3 = fdl.Config(SampleClass, 'f3-x', shared)
    a = fdl.Config(SampleClass, f1, f2, f3)
    b = fdl.Config(SampleClass, 'b-x', shared)
    config = fdl.Config(SampleClass, a, b)

    node_to_parents_by_id = sub_fixture._get_node_to_parents_mapping(config)

    with self.subTest(name='node_to_parents_mapping'):
      self.assertEmpty(node_to_parents_by_id[id(config)])
      self.assertSetEqual(node_to_parents_by_id[id(a)], {id(config)})
      self.assertSetEqual(node_to_parents_by_id[id(b)], {id(config)})
      self.assertSetEqual(node_to_parents_by_id[id(f1)], {id(a)})
      self.assertSetEqual(node_to_parents_by_id[id(f2)], {id(a)})
      self.assertSetEqual(
          node_to_parents_by_id[id(f3)], node_to_parents_by_id[id(f2)]
      )
      self.assertSetEqual(
          node_to_parents_by_id[id(shared)], {id(f1), id(f2), id(f3), id(b)}
      )

    with self.subTest(name='is_super_ancestor'):
      is_super_ancestor = functools.partial(
          sub_fixture._is_super_ancestor,
          node_to_parents_by_id=node_to_parents_by_id,
      )
      self.assertTrue(is_super_ancestor(id(config), {id(config)}))
      self.assertTrue(is_super_ancestor(id(config), {id(a), id(config)}))
      self.assertTrue(is_super_ancestor(id(config), {id(a), id(b)}))
      self.assertTrue(is_super_ancestor(id(config), {id(shared)}))
      self.assertTrue(
          is_super_ancestor(id(config), {id(f1), id(f2), id(f3), id(shared)})
      )
      self.assertFalse(is_super_ancestor(id(a), {id(f1), id(shared)}))
      self.assertTrue(is_super_ancestor(id(a), {id(f1), id(f2), id(f3)}))
      self.assertFalse(
          is_super_ancestor(id(a), {id(f1), id(f2), id(f3), id(shared)})
      )

    # LCA: least common ancestor
    with self.subTest(checking='find_LCA', name='shared_node'):
      ancestor_id = sub_fixture._find_least_common_ancestor(
          {id(shared), id(f2), id(f3)}, node_to_parents_by_id
      )
      self.assertEqual(ancestor_id, id(config))

    with self.subTest(
        checking='find_LCA', name='shared_node_not_root_as_ancestor'
    ):
      config.y.y = 'b-y'
      node_to_parents_by_id = sub_fixture._get_node_to_parents_mapping(config)
      ancestor_id = sub_fixture._find_least_common_ancestor(
          {id(shared), id(f2), id(f3)}, node_to_parents_by_id
      )
      self.assertEqual(ancestor_id, id(a))

    with self.subTest(checking='find_LCA', name='node_with_parent'):
      ancestor_id = sub_fixture._find_least_common_ancestor(
          {id(f1), id(a)}, node_to_parents_by_id
      )
      self.assertEqual(ancestor_id, id(a))

    with self.subTest(checking='find_LCA', name='three_nodes_at_same_lv'):
      ancestor_id = sub_fixture._find_least_common_ancestor(
          {id(f1), id(f2), id(f3)}, node_to_parents_by_id
      )
      self.assertEqual(ancestor_id, id(a))

    with self.subTest(checking='find_LCA', name='four_nodes_at_different_lv'):
      ancestor_id = sub_fixture._find_least_common_ancestor(
          {id(f1), id(f2), id(f3), id(a)}, node_to_parents_by_id
      )
      self.assertEqual(ancestor_id, id(a))


class SubFixtureTest(absltest.TestCase):

  def test_find_shared_nodes(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    fixtures = {'fake_encoder': config.encoder, 'fake_decoder': config.decoder}
    node_to_paraents = sub_fixture._get_node_to_parents_mapping(config)
    shared_nodes, shared_node_paths = sub_fixture._find_shared_nodes(
        config, fixtures, node_to_paraents
    )

    with self.subTest('shared_node_lib'):
      shared_node = config.encoder.embedders['tokens']
      self.assertIn(id(shared_node), shared_nodes)
      self.assertLen(shared_nodes, 1)

    with self.subTest('shared_node_path'):
      for node in shared_nodes:
        self.assertIn(node, shared_node_paths)
        path = shared_node_paths[node]
        self.assertLen(path, 2)

  def test_find_nested_shared_nodes(self):
    config = demo_configs.nested_node_sharing_config.as_buildable()
    fixtures = {
        'fx_a': config.x,
        'fx_f1': config.x.x,
        'fx_f2': config.x.y,
        'fx_f3': config.x.z,
    }
    node_to_parents = sub_fixture._get_node_to_parents_mapping(config)
    shared_nodes, shared_node_paths = sub_fixture._find_shared_nodes(
        config, fixtures, node_to_parents
    )
    with self.subTest('shared_node_lib'):
      shared_node = config.x.x.y
      self.assertIn(id(shared_node), shared_nodes)
      self.assertLen(shared_nodes, 1)

    with self.subTest('shared_node_path'):
      for node in shared_nodes:
        self.assertIn(node, shared_node_paths)
        path = shared_node_paths[node]
        self.assertLen(path, 1)

    with self.subTest('shared_node_definition'):
      task = init_task.init_task(config)
      sub_fixture.transform_sub_fixtures(task, fixtures)
      self.assertEmpty(task.top_level_call.fn.variables)
      for call_instance in task.top_level_call.children:
        if call_instance.fn.name.value == 'fx_a':
          self.assertLen(call_instance.fn.variables, 1)
        else:
          self.assertEmpty(call_instance.fn.variables)

      make_symbolic_references.import_symbols(task)
      make_symbolic_references.replace_callables_and_configs_with_symbols(task)
      code = ir_to_cst.code_for_task(task).code
      # The shared node should be defined in `fx_a` instead of the top fixture.
      expected = """
      from fiddle._src.testing.example import demo_configs
      from fiddle.experimental import auto_config


      @auto_config.auto_config
      def config_fixture():
          return demo_configs.Simple(x=fx_a(), y=demo_configs.Simple(x='b-x', y='b-y'))


      @auto_config.auto_config
      def fx_f1(simple):
          return demo_configs.Simple(x='f1-x', y=simple)


      @auto_config.auto_config
      def fx_f2(simple):
          return demo_configs.Simple(x='f2-x', y=simple)


      @auto_config.auto_config
      def fx_f3(simple):
          return demo_configs.Simple(x='f3-x', y=simple)


      @auto_config.auto_config
      def fx_a():
          simple = demo_configs.Simple(x='shared-x', y='shared-y')
          return demo_configs.Simple(x=fx_f1(simple=simple), y=fx_f2(simple=simple), z=fx_f3(simple=simple))
      """
      self.assertEqual(code.split(), expected.split(), msg=code)

  def test_linear_nested_shared_node(self):
    # When a node is shared by multiple sub-fixtures, it's not always a shared
    # node. For example, in a config: A <- B <- C <- D and A, B, C are
    # sub-fixtures, D should not be identified as a shared node.
    config = demo_configs.linear_nested_config.as_buildable()
    fixtures = {
        'A': config.x,
        'B': config.x.x,
        'C': config.x.x.x,
    }
    node_to_parents = sub_fixture._get_node_to_parents_mapping(config)
    shared_nodes, shared_node_paths = sub_fixture._find_shared_nodes(
        config, fixtures, node_to_parents
    )
    self.assertEmpty(shared_nodes)
    self.assertEmpty(shared_node_paths)

  def test_fake_encoder_decoder(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    config.encoder.embedders = None
    config.decoder.embedders = None

    task = init_task.init_task(config)
    sub_fixture.transform_sub_fixtures(
        task, {'fake_encoder': config.encoder, 'fake_decoder': config.decoder}
    )
    self.assertLen(task.top_level_call.children, 2)

  def test_nested_sub_fixture(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    task = init_task.init_task(config)
    sub_fixture.transform_sub_fixtures(
        task,
        {'fake_encoder': config.encoder, 'attention': config.encoder.attention},
    )
    self.assertLen(task.top_level_call.children, 2)

  def test_conflicting_name_raises_errors(self):
    config = fake_encoder_decoder.fixture.as_buildable()
    task = init_task.init_task(config)
    with self.assertRaisesRegex(
        ValueError, 'already exists in the top level fixture'
    ):
      sub_fixture.transform_sub_fixtures(
          task, {'config_fixture': config.encoder}
      )

  # TODO(b/284359119): Support fdl.ArgFactory as sub-fixture.
  def test_arg_factory_raises_errors(self):
    config = fdl.Partial(
        test_fixtures.EncoderLayer,
        attention=fdl.ArgFactory(
            test_fixtures.Attention,
            kernel_init=fdl.ArgFactory(
                test_fixtures.initializer, name='const', dtype='float32'
            ),
        ),
    )
    task = init_task.init_task(config)
    with self.assertRaisesRegex(ValueError, 'fdl.ArgFactory is not supported'):
      sub_fixture.transform_sub_fixtures(task, {'attention': config.attention})


if __name__ == '__main__':
  absltest.main()
