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

class DreamerV3Wrapper:
    """Converts the DreamerV3 model to a PyTorch model."""
    
    def __init__(self, config, model_path, observation_space, action_space, device):    
        """
        Loads the DreamerV3 model and config from the given paths.
        
        Arguments:
            config_path {str} -- _description_
            model_path {str} _description_
        """
        # Convert to Path objects
        model_path = Path(model_path)
        
        # Load the config and model
        if device == "cpu":
            config = config.update({"jax.platform" : "cpu"})
        
        step = embodied.Counter()
        agent = dreamerv3.Agent(observation_space, action_space, step, config)
        
        checkpoint = Checkpoint()
        checkpoint.agent = agent
        
        checkpoint.load(model_path, keys=['agent'])
        
        self.agent = agent
    
    
    def forward(self, obs, state):
        """ The forward pass of the model.

        Arguments:
            obs {dict} -- The observation
            state (_type_): _description_

        Returns:
            {OneHotCategorical} -- The action distribution
            {dict} -- The state
        """
        action, state, task_outs = self.agent.policy(obs, state, mode='eval')
        
        # Get the logits from the task outputs
        action_logits = task_outs['action'].logits.tolist()
        # Move the distribution to CPU
        logits = torch.tensor(action_logits)
        # Create an equivalent PyTorch Categorical distribution
        policy = OneHotCategorical(logits=logits)
        return policy, state, action
        
    def __call__(self, obs, state):
        """Calls the forward pass of the model."""
        return self.forward(obs, state)


base_path = "./model/expert/crafter/"
config_path = Path(base_path + 'config.yaml')
model_path = Path(base_path + 'checkpoint.ckpt')


config = embodied.Config.load(config_path)
config = config.update({"jax.platform" : "cpu"})

env = crafter.Env() 
env = from_gym.FromGym(env)
env = dreamerv3.wrap_env(env, config)

agent = DreamerV3Wrapper(config, model_path, env.obs_space, env.act_space, "cpu")

state = None
act = {'action': env.act_space['action'].sample(), 'reset': np.array(True)}
done, rewards, iter = False, [], 0
while not done:
    obs = env.step(act)
    obs = {k: v[None] if isinstance(v, (list, dict)) else np.array([v]) for k, v in obs.items()}
    policy, state, act = agent(obs, state)
    
    act = {'action': act["action"][0], 'reset': obs['is_last'][0]}
    # Log result
    # clear_output()
    # time.sleep(0.5)
    rewards.append(obs["reward"][0])
    done = obs["is_terminal"]
    print("\riter:", iter, "reward:", np.sum(rewards), "done:", done[0], end='', flush=True)
    iter += + 1

# Get the logits from the task outputs
# action_logits = task_outs['action'].logits.tolist()

# # Move the distribution to CPU
# logits = torch.tensor(action_logits) 

# print(logits)

# # Create an equivalent PyTorch Categorical distribution
# distribution = OneHotCategorical(logits=logits)
# # Sample from the distribution
# sample = distribution.sample()

# print("sample:", sample)
# print("log_prob:", distribution.log_prob(sample))
# print("prob:", torch.exp(distribution.log_prob(sample)))