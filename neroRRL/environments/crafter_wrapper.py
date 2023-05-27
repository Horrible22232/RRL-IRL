import crafter
import numpy as np

from pathlib import Path
from random import randint

from neroRRL.environments.env import Env
from neroRRL.expert.modules import dreamerv3
from neroRRL.expert.modules.dreamerv3 import embodied
from neroRRL.expert.modules.dreamerv3.embodied.envs import from_gym

class CrafterWrapper(Env):
    """
    This class wraps the crafter environment.
        https://github.com/danijar/crafter
    """
    def __init__(self, expert_params = None, reset_params = None, realtime_mode = False, record_trajectory = False, expert = None) -> None:
        """Instantiates the memory-gym environment.
        
        Arguments:
            env_name {string} -- Name of the memory-gym environment
            reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
            realtime_mode {bool} -- Whether to render the environment in realtime. (default: {False})
            record_trajectory {bool} -- Whether to record the trajectory of an entire episode. This can be used for video recording. (default: {False})
        """
        if reset_params is None:
            self._default_reset_params = {"start-seed": 0, "num-seeds": 100}
        else:
            self._default_reset_params = reset_params
            
        # Set the expert
        self._expert = expert

        # render_mode = None if not realtime_mode else "debug_rgb_array"
        
        # self._env = gym.make(env_name, disable_env_checker = True, render_mode = render_mode)
        
        # Use the dreamerv3 config to make sure that the environment is compatible with the dreamerv3 agent
        config_path = './model/expert/crafter/config.yaml' if expert_params is None else expert_params["config_path"]
        config_path = Path(config_path)
        config = embodied.Config.load(config_path)
        
        # Create the environment like in the dreamerv3 code
        self._env = crafter.Env() 
        self._env = from_gym.FromGym(self._env)
        self._env = dreamerv3.wrap_env(self._env, config)

        self._realtime_mode = realtime_mode
        self._record = record_trajectory

        self._visual_observation_space = self._env.observation_space
        self._vector_observation_space = None
        
    @property
    def _has_expert(self):
        """Returns whether the environment has an expert."""
        return self._expert is not None
    
    @property
    def _expert_policy(self):
        """Returns the expert policy."""
        return self._expert
        
    @property
    def unwrapped(self):
        """Return this environment in its vanilla (i.e. unwrapped) state."""
        return self

    @property
    def visual_observation_space(self):
        """Returns the shape of the visual component of the observation space as a tuple."""
        return self._visual_observation_space

    @property
    def vector_observation_space(self):
        """Returns the shape of the vector component of the observation space as a tuple."""
        return self._vector_observation_space

    @property
    def action_space(self):
        """Returns the shape of the action space of the agent."""
        return self._env.action_space

    @property
    def max_episode_steps(self):
        """Returns the maximum number of steps that an episode can last."""
        return self._env.max_episode_steps

    @property
    def seed(self):
        """Returns the seed of the current episode."""
        return self._seed

    @property
    def action_names(self):
        """Returns a list of action names. It has to be noted that only the names of action branches are provided and not the actions themselves!"""
        return [["no-op", "rotate left", "rotate right", "move forward"]]

    @property
    def get_episode_trajectory(self):
        """Returns the trajectory of an entire episode as dictionary (vis_obs, vec_obs, rewards, actions)."""
        self._trajectory["action_names"] = self.action_names
        return self._trajectory if self._trajectory else None

    def reset(self, reset_params = None):
        """Resets the environment.
        
        Keyword Arguments:
            reset_params {dict} -- Provides parameters, like a seed, to configure the environment. (default: {None})
        
        Returns:
            {numpy.ndarray} -- Visual observation
            {numpy.ndarray} -- Vector observation
        """
        # Process reset parameters
        if reset_params is None:
            reset_params = self._default_reset_params
        else:
            reset_params = reset_params

        # Sample seed
        self._seed = randint(reset_params["start-seed"], reset_params["start-seed"] + reset_params["num-seeds"] - 1)

        # Remove reset params that are not processed directly by the environment
        options = reset_params.copy()
        options.pop("start-seed", None)
        options.pop("num-seeds", None)
        options.pop("seed", None)

        # Reset the environment to retrieve the initial observation
        act = {'action': self._env.act_space['action'].sample(), 'reset': np.array(True)}
        env_data = self._env.step(act)
        
        vis_obs = env_data['image'] / 255.0
        
        # Track rewards of an entire episode
        self._rewards = []
        
        if self._has_expert:
            self._forward_expert(obs)

        if self._realtime_mode:
            self._env.render()

        # Prepare trajectory recording
        self._trajectory = {
            "vis_obs": [self._env.render()], "vec_obs": [None],
            "rewards": [0.0], "actions": []
        } if self._record else None

        return vis_obs, None

    def step(self, action):
        """Runs one timestep of the environment's dynamics.
        
        Arguments:
            action {int} -- The to be executed action
        
        Returns:
            {numpy.ndarray} -- Visual observation
            {numpy.ndarray} -- Vector observation
            {float} -- (Total) Scalar reward signaled by the environment
            {bool} -- Whether the episode of the environment terminated
            {dict} -- Further episode information (e.g. cumulated reward) retrieved from the environment once an episode completed
        """
            
        act = {'action': action, 'reset': np.array(False)}
        env_data = self._env.step(act)
        
        vis_obs, reward, done, info = env_data['image'] / 255.0, env_data['reward'], env_data['is_last'], {}
        
        # Track rewards of an entire episode
        self._rewards.append(reward)

        if self._realtime_mode or self._record:
            img = env_data['image']
            
        # Wrap up episode information once completed (i.e. done)
        if done or truncation:
            info = {"reward": sum(self._rewards),
                    "length": len(self._rewards)}
        else:
            info = None

        # Record trajectory data
        if self._record:
            self._trajectory["vis_obs"].append(env_data['image'])
            self._trajectory["vec_obs"].append(None)
            self._trajectory["rewards"].append(reward)
            self._trajectory["actions"].append([action])

        return vis_obs, None, reward, done, info

    def close(self):
        """Shuts down the environment."""
        self._env.close()