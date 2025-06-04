from os import path
from typing import Dict, Union, List

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from src.utils.geometry import quat2rot
from scipy.spatial.transform import Rotation as R

DEFAULT_CAMERA_CONFIG = {
    "distance": 20,
}


class AntCustomEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        robot_path: str,
        frame_skip: int = 5,
        default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1,
        ctrl_cost_weight: float = 0.5,
        cfrc_cost_weight: float = 5e-4,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = False,
        pert_force=None,
        **kwargs,
    ):
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            robot_path,
        )

        utils.EzPickle.__init__(
            self,
            xml_file_path,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            cfrc_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            pert_force,
            **kwargs,
        )
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._cfrc_cost_weight = cfrc_cost_weight
        self._reset_noise_scale = reset_noise_scale
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

        MujocoEnv.__init__(
            self,
            xml_file_path,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            width=832,
            height=496,
            camera_name="track",
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += (
            self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation
        )

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }
        self.body_ids = [1, 18]  # Fixed body IDs
        self.force = None
        self.previous_state = None
        self.stuck = 0
        if pert_force is not None:
            self.force = pert_force

    def step(self, action):
        forward_rewards = 0.0
        xy_positions_before = []
        xy_positions_after = []

        for body_id in self.body_ids:
            pos_before = self.data.xpos[body_id][:2].copy()
            xy_positions_before.append(pos_before)
            # print("pos before : ",xy_positions_before )

        if self.force is not None:
            self.apply_force()

        self.do_simulation(action, self.frame_skip)

        for body_id in self.body_ids:
            pos_after = self.data.xpos[body_id][:2].copy()
            xy_positions_after.append(pos_after)
            # print("pos after : ",xy_positions_after)

        velocities = []
        for before, after in zip(xy_positions_before, xy_positions_after):
            vel = (after - before) / self.dt
            velocities.append(vel)
            forward_rewards += vel[0]  # x-velocity
        forward_reward = np.linalg.norm(np.array(xy_positions_after)-np.array(xy_positions_before))

        x_velocity = forward_rewards/2.0
        forward_reward = forward_rewards * self._forward_reward_weight

        distance = np.linalg.norm(xy_positions_after[0] - xy_positions_after[1])
        separation_penalty = (distance)/len(self.body_ids)

        healthy_reward = 1.0
        ctrl_cost = np.linalg.norm(action) ** 2 * self._ctrl_cost_weight
        cfrc_cost = np.linalg.norm(self.data.cfrc_ext[1:]) ** 2 * self._cfrc_cost_weight

        # print("\n ctrl_cost : ", ctrl_cost) #value around 6 or 7
        # print("\n cfrc_cost : ", cfrc_cost) #value close to 0, sometimes goes up to 4

        reward = 2.0*forward_reward - separation_penalty
        forward_reward = (forward_rewards / len(self.body_ids)) * 1000.0
        # print("forward speed : ", forward_reward)
        # print("separation penalty : ", separation_penalty)

        #print forward_reward and other costs to see the order of magnitude to adapt the coeff
        # reward = healthy_reward + forward_reward - ctrl_cost - cfrc_cost - separation_penalty
        
        observation = self._get_obs()

        info = {
            "reward_forward": forward_reward,
            "vel_x": x_velocity,
            "healthy_reward": healthy_reward,
            "ctrl_cost": ctrl_cost,
            "cfrc_cost": cfrc_cost,
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
            "separation" : separation_penalty,
            "inter_distance":distance,
        }

        terminated = False
        qacc = self.data.qacc

        # print("qpos shape : ", self.data.qpos.shape)
        # print(dir(self.data))

        # if (forward_reward < 200) : #try 400 ?
        #     terminated = True
        
        # for body_id in self.body_ids:
        #     quat = self.data.xquat[body_id]  # [w, x, y, z]
        #     rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  #convert to [x, y, z, w]
        #     euler = rot.as_euler('xyz', degrees=True)
        #     roll, pitch, _ = euler

        #     if abs(roll) > 70 or abs(pitch) > 70:
        #         # print(f"Terminating due to rotation: body {body_id}, roll={roll:.2f}, pitch={pitch:.2f}")
        #         terminated = True

        #     info[f"body_{body_id}_roll"] = roll
        #     info[f"body_{body_id}_pitch"] = pitch

        if np.any(np.isnan(qacc)) or np.any(np.isinf(qacc)) or np.any(np.abs(qacc) > 1e6):
            terminated = True

        ant1_height = self.data.xpos[self.body_ids[0]][2]
        ant2_height = self.data.xpos[self.body_ids[1]][2]

        if ((ant1_height < 0.2) or (ant1_height > 1.0) or (ant2_height < 0.2) or (ant2_height > 1.0)):
            terminated = True
            #look at rotation of the core, look if it's standing still

        if np.isinf(observation).any():
            terminated = True

        self.previous_state = observation

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate((position, velocity))

    # def _get_obs(self):
    #     ant1_pos = self.data.qpos[:15]  # First ant's 15-DOF position
    #     ant2_pos = self.data.qpos[15:]  # Second ant's position
    #     ant1_vel = self.data.qvel[:14]  # First ant's velocity
    #     ant2_vel = self.data.qvel[14:]  # Second ant's velocity
    #     return np.concatenate([ant1_pos, ant2_pos, ant1_vel, ant2_vel])

    def apply_force(self):
        for body_id in self.body_ids:
            force = self.force
            pert = self.np_random.uniform(low=-0.1, high=0.1, size=3)
            self.data.xfrc_applied[body_id] = force

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale
            * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }
