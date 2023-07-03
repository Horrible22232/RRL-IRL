from neroRRL.expert.modules.dreamerv3.embodied.envs import from_gym
from neroRRL.expert.modules.dreamerv3.embodied.core import Checkpoint
from neroRRL.expert.modules import dreamerv3
from neroRRL.expert.modules.dreamerv3 import embodied
import numpy as np
import crafter
from pathlib import Path
import torch
import jax
import torch.nn.functional as F
import tensorflow as tf
from torch.distributions import OneHotCategorical

class DreamerV3Wrapper:
    """Converts the DreamerV3 model to a PyTorch model."""
    
    def __init__(self, config_path, model_path, observation_space, action_space, device):    
        """Loads the DreamerV3 model and config from the given paths.
        
        Arguments:
            config_path {str} -- The config path to the model
            model_path {str} -- The path to the model
            observation_space {box} -- The observation space
            action_space {tuple} -- The action space
            device {str} -- The device to run the model on
        """
        # Convert to Path objects
        config_path = Path(config_path)
        model_path = Path(model_path)
        # Load the config
        config = embodied.Config.load(config_path)
        # Set the device
        if device.type == "cpu":
            config = config.update({"jax.platform" : "cpu"})
        elif device.type == "cuda":
            config = config.update({"jax.platform" : "gpu"})
        # Create the agent
        step = embodied.Counter()
        agent = dreamerv3.Agent(observation_space, action_space, step, config)
        # Create the checkpoint to load the model
        checkpoint = Checkpoint()
        checkpoint.agent = agent
        checkpoint.load(model_path, keys=['agent'])
        # Set the final loaded agent
        self.agent = agent
    
    
    def forward(self, obs, state):
        """The forward pass of the dreamerv3 model.

        Arguments:
            obs {dict} -- The observation
            state {tuple} -- The state

        Returns:
            {OneHotCategorical} -- The action distribution
            {dict} -- The state
        """
        # Get the task outputs from the agent
        _, state, task_outs = self.agent.policy(obs, state, mode='eval')
        # Get the logits from the task outputs and move them to the cpu
        action_logits = jax.device_get(task_outs['action'].logits)
        action_logits = action_logits.tolist()
        # Convert the logits to a tensor
        logits = torch.tensor(action_logits)
        # Create an equivalent PyTorch Categorical distribution
        policy = OneHotCategorical(logits=logits)
        # Return the policy and state
        return policy, state
        
    def __call__(self, obs, state):
        """Calls the forward pass of the model."""
        forward_pass = self.forward(obs, state)
        # assert False
        return forward_pass


def test():
    base_path = "./model/expert/crafter/"
    config_path = Path(base_path + 'config.yaml')
    model_path = Path(base_path + 'checkpoint.ckpt')

    config = embodied.Config.load(config_path)

    env = crafter.Env() 
    env = from_gym.FromGym(env)
    env = dreamerv3.wrap_env(env, config)

    agent = DreamerV3Wrapper(config_path, model_path, env.obs_space, env.act_space, torch.device("cpu"))

    state = None
    act = {'action': env.act_space['action'].sample(), 'reset': np.array(True)}
    done, rewards, iter = False, [], 0
    while not done:
        obs = env.step(act)
        print("image.shape", obs['image'].shape)
        
        print(obs.keys())
        obs = {k: [v, v] if isinstance(v, (list, dict)) else np.array([v, v]) for k, v in obs.items()}
        policy, state = agent(obs, state)
        
        act = {'action': policy.sample().cpu().numpy()[0], 'reset': obs['is_last'][0]}
        
        # Log result
        # clear_output()
        # time.sleep(0.5)
        rewards.append(obs["reward"][0])
        done = obs["is_terminal"]
        print("\riter:", iter, "reward:", np.sum(rewards), "done:", done[0], end='', flush=True)
        iter += + 1
        print(type(state))
        print(state)
        print(policy.sample().cpu().numpy())
        print(state.shape)
        assert False
    # Get the logits from the task outputs
    # action_logits = task_outs['action'].logits.tolist()

    # Move the distribution to CPU
    #logits = torch.tensor(action_logits) 

    #print(logits)

    # Create an equivalent PyTorch Categorical distribution
    #distribution = OneHotCategorical(logits=logits)
    # Sample from the distribution
    sample = policy.sample()

    print("sample:", sample)
    print("log_prob:", policy.log_prob(sample))
    print("prob:", torch.exp(policy.log_prob(sample)))
    
# test()