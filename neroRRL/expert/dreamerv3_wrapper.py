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

checkpoint.load(model_path, keys=['agent'])

state = None
act = {'action': env.act_space['action'].sample(), 'reset': np.array(True)}
done, rewards, iter = False, [], 0
while not done:
    obs = env.step(act)
    obs = {k: v[None] if isinstance(v, (list, dict)) else np.array([v]) for k, v in obs.items()}
    act, state = agent.policy(obs, state, mode='eval')
    act = {'action': act['action'][0], 'reset': obs['is_last'][0]}
    # Log result
    # clear_output()
    # time.sleep(0.5)
    rewards.append(obs["reward"][0])
    done = obs["is_terminal"]
    print("\riter:", iter, "reward:", np.sum(rewards), "done:", done[0], end='', flush=True)
    iter += + 1