from neroRRL.expert.modules.dreamerv3.embodied.envs import from_gym
from neroRRL.expert.modules.dreamerv3.embodied.core import Checkpoint
from neroRRL.expert.modules import dreamerv3
from neroRRL.expert.modules.dreamerv3 import embodied
import numpy as np
import crafter
from pathlib import Path
import torch
import torch.nn.functional as F
import tensorflow as tf
from torch.distributions import OneHotCategorical


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

checkpoint.load(model_path, keys=['agent'])

state = None
act = {'action': env.act_space['action'].sample(), 'reset': np.array(True)}
done, rewards, iter = False, [], 0
while not done:
    obs = env.step(act)
    obs = {k: v[None] if isinstance(v, (list, dict)) else np.array([v]) for k, v in obs.items()}
    act, state, task_outs = agent.policy(obs, state, mode='eval')
    break
    act = {'action': act['action'][0], 'reset': obs['is_last'][0]}
    # Log result
    # clear_output()
    # time.sleep(0.5)
    rewards.append(obs["reward"][0])
    done = obs["is_terminal"]
    print("\riter:", iter, "reward:", np.sum(rewards), "done:", done[0], end='', flush=True)
    iter += + 1

# Get the logits from the task outputs
action_logits = task_outs['action'].logits.tolist()

# Move the distribution to CPU
logits = torch.tensor(action_logits) 

print(logits)

# Create an equivalent PyTorch Categorical distribution
distribution = OneHotCategorical(logits=logits)
# Sample from the distribution
sample = distribution.sample()

print("sample:", sample)
print("log_prob:", distribution.log_prob(sample))
print("prob:", torch.exp(distribution.log_prob(sample)))