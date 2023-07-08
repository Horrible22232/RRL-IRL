import torch

def jsd_reward(policy, expert_policy, actions):
    """_summary_

    Args:
        policy (_type_): _description_
        expert_policy (_type_): _description_
        actions (_type_): _description_

    Returns:
        _type_: _description_
    """
    policy = policy[0]
    agent_probs = policy.probs
    expert_probs = expert_policy.probs

    # Calculate the Jensen-Shannon Divergence
    m = 0.5 * (agent_probs + expert_probs)
    jsd = 0.5 * torch.sum(agent_probs * torch.log(agent_probs / m), dim=1) + 0.5 * torch.sum(expert_probs * torch.log(expert_probs / m), dim=1)

    # Ensure jsd is positive and above a certain minimum to avoid numerical instability
    jsd = torch.clamp(jsd, min=1e-10)

    # Calculate the similarity score
    similarity_score = 1 - jsd

    # Invert the similarity score so that higher values mean greater similarity
    expert_reward = similarity_score.cpu().numpy()

    return expert_reward