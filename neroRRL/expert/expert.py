from neroRRL.expert.modules.dreamerv3.embodied.envs import from_gym
from neroRRL.expert.modules.dreamerv3.embodied.core import Checkpoint
from neroRRL.expert.modules import dreamerv3
from neroRRL.expert.modules.dreamerv3 import embodied
import numpy as np
import crafter
from pathlib import Path


base_path = "./model/expert/crafter/"
config_path = Path(base_path + 'config.yaml')
model_path = Path(base_path + 'checkpoint.ckpt')



config = embodied.Config.load(config_path)
config = config.update({"jax.platform" : "cpu"})

env = crafter.Env() 
env = from_gym.FromGym(env)
env = dreamerv3.wrap_env(env, config)

step = embodied.Counter()
agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)

checkpoint = Checkpoint()
checkpoint.agent = agent