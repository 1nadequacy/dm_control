# Copyright 2019 The dm_control Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Ant Domain."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dm_control import mujoco
from dm_control.mujoco.wrapper import mjbindings
from dm_control.rl import control
from dm_control.suite import base
from dm_control.suite import common
from dm_control.utils import containers
from dm_control.utils import rewards
from dm_control.utils import xml_tools

from lxml import etree
import numpy as np
from scipy import ndimage
from enum import Enum

enums = mjbindings.enums
mjlib = mjbindings.mjlib

_DEFAULT_TIME_LIMIT = 100
SUITE = containers.TaggedTasks()


def generate_valid_pos(lower_bound=1., upper_boud=14.):
    position = np.random.uniform(lower_bound, upper_boud, size=(2, ))
    sign = np.random.randint(0, 2, size=(2, )) * 2 - 1
    return position * sign


def make_model(num_walls=0, random_goal=False):
    xml_string = common.read_model('ant.xml')
    parser = etree.XMLParser(remove_blank_text=True)
    mjcf = etree.XML(xml_string, parser)

    if random_goal:
        target_site = xml_tools.find_element(mjcf, 'site', 'target')
        x, y = generate_valid_pos()
        target_site.attrib['pos'] = '{} {} .05'.format(x, y)

    return etree.tostring(mjcf, pretty_print=True)


def _create_reach(sparse=False,
                  with_goal=False,
                  time_limit=_DEFAULT_TIME_LIMIT,
                  random=None,
                  environment_kwargs=None):
    xml_string = make_model(num_walls=0, random_goal=with_goal)
    physics = Physics.from_xml_string(xml_string, common.ASSETS)
    environment_kwargs = environment_kwargs or {}
    n_sub_steps = environment_kwargs.pop('n_sub_steps', 5)
    task = Ant(sparse=sparse, with_goal=with_goal, random=random)

    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        n_sub_steps=n_sub_steps,
        **environment_kwargs)


@SUITE.add()
def reach(time_limit=_DEFAULT_TIME_LIMIT, random=None,
          environment_kwargs=None):
    return _create_reach(False, False, time_limit, random, environment_kwargs)


@SUITE.add()
def reach_random(time_limit=_DEFAULT_TIME_LIMIT,
                 random=None,
                 environment_kwargs=None):
    return _create_reach(False, True, time_limit, random, environment_kwargs)


@SUITE.add()
def reach_sparse(time_limit=_DEFAULT_TIME_LIMIT,
                 random=None,
                 environment_kwargs=None):
    return _create_reach(True, False, time_limit, random, environment_kwargs)


@SUITE.add()
def reach_random_sparse(time_limit=_DEFAULT_TIME_LIMIT,
                        random=None,
                        environment_kwargs=None):
    return _create_reach(True, True, time_limit, random, environment_kwargs)


class Physics(mujoco.Physics):
    def torso_upright(self):
        return np.asarray(self.named.data.xmat['torso', 'zz'])

    def self_to_target(self):
        return self.named.data.site_xpos['target'] - self.named.data.xpos[
            'torso']

    def target_position(self):
        return self.named.data.site_xpos['target']

    def self_to_target_distance(self):
        return np.linalg.norm(self.self_to_target()[:2])


class Ant(base.Task):
    def __init__(self, sparse=False, with_goal=False, random=None):
        super(Ant, self).__init__(random=random)
        self.sparse = sparse
        self.with_goal = with_goal

    def initialize_episode(self, physics):
        x_pos, y_pos = generate_valid_pos()
        z_pos = 0
        num_contacts = 1
        while num_contacts > 0:
            try:
                with physics.reset_context():
                    physics.named.data.qpos['root'][:3] = x_pos, y_pos, z_pos
            except control.PhysicsError:
                # We may encounter a PhysicsError here due to filling the contact
                # buffer, in which case we simply increment the height and continue.
                pass
            num_contacts = physics.data.ncon
            z_pos += 0.01
        super(Ant, self).initialize_episode(physics)

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['position'] = physics.position()
        obs['velocity'] = physics.velocity()
        if self.with_goal:
            obs['target'] = physics.target_position()
        return obs

    def get_reward(self, physics):
        floor_radius = physics.named.model.geom_size['floor', 0] * np.sqrt(
            2) * 2
        target_radius = physics.named.model.site_size['target', 0]
        torso_radius = physics.named.model.geom_size['torso_geom', 0]
        margin = (
            target_radius + torso_radius) if self.sparse else floor_radius

        reach_reward = rewards.tolerance(
            physics.self_to_target_distance(),
            bounds=(0, target_radius),
            sigmoid='linear',
            margin=margin,
            value_at_margin=0)

        return reach_reward
