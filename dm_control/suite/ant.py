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

enums = mjbindings.enums
mjlib = mjbindings.mjlib

_DEFAULT_TIME_LIMIT = 100
SUITE = containers.TaggedTasks()


def make_model(num_walls=0):
    xml_string = common.read_model('ant.xml')
    parser = etree.XMLParser(remove_blank_text=True)
    mjcf = etree.XML(xml_string, parser)

    return etree.tostring(mjcf, pretty_print=True)


@SUITE.add()
def reach(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    xml_string = make_model(num_walls=0)
    physics = Physics.from_xml_string(xml_string, common.ASSETS)
    task = Ant(random=random)
    environment_kwargs = environment_kwargs or {}
    return control.Environment(
        physics,
        task,
        time_limit=time_limit,
        n_sub_steps=5,
        **environment_kwargs)


class Physics(mujoco.Physics):
    def torso_upright(self):
        return np.asarray(self.named.data.xmat['torso', 'zz'])
        
    def self_to_target(self):
        return self.named.data.site_xpos['target'] - self.named.data.xpos['torso']
        
    def self_to_target_distance(self):
        return np.linalg.norm(self.self_to_target()[:2])
    
    
class Ant(base.Task):
    def __init__(self, random=None):
        super(Ant, self).__init__(random=random)
        
    def initialize_episode(self, physics):
        spawn_radius = 0.9 * physics.named.model.geom_size['floor', 0]
        start_choices = [(-12, -12), (-12, 12), (12, -12)]
        idx = np.random.randint(len(start_choices))
        x_pos, y_pos = start_choices[idx]
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
        return obs

    def get_reward(self, physics):
        floor_max_distance = physics.named.model.geom_size['floor', 0] * np.sqrt(2) * 2
        target_radius = physics.named.model.site_size['target', 0]
        reach_reward = rewards.tolerance(
            physics.self_to_target_distance(),
            bounds=(0, target_radius),
            sigmoid='linear',
            margin=floor_max_distance, value_at_margin=0)
        
        return reach_reward