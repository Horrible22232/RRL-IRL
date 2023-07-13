import torch
from neroRRL.environments.expert_rewards.expert_rewards import *

def test_jsd_reward():
    """Test the jsd_reward function
    """
    # Create some dummy data
    actions = torch.randn(10, 1)  # Add an extra dimension
    policy_probs = torch.softmax(torch.randn(10, 1), dim=0)  # Add an extra dimension
    
    # Case 1: Similar policies
    # Create slightly perturbed expert_policy_probs
    noise = torch.randn(10, 1) * 0.01  # Small random noise, add an extra dimension
    expert_policy_probs_similar = torch.softmax(policy_probs + noise, dim=0)  # Adding noise to policy_probs
    
    # Create dummy policies
    policy_similar = [type('', (), {'probs': policy_probs})()]
    expert_policy_similar = type('', (), {'probs': expert_policy_probs_similar})

    # Call the function with the dummy data
    reward_similar = jsd_reward(policy_similar, expert_policy_similar, actions)
    
    # Print the output
    print(f"Reward with similar policies: {reward_similar}")
    
    # Case 2: Different policies
    expert_policy_probs_different = torch.softmax(torch.randn(10, 1), dim=0)  # expert_policy is different from policy, add an extra dimension
    
    # Create dummy policies
    policy_different = [type('', (), {'probs': policy_probs})()]
    expert_policy_different = type('', (), {'probs': expert_policy_probs_different})

    # Call the function with the dummy data
    reward_different = jsd_reward(policy_different, expert_policy_different, actions)
    
    # Print the output
    print(f"Reward with different policies: {reward_different}")