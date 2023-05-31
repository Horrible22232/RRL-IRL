from neroRRL.environments.wrappers.expert_rewards import PyTorchEnv

class KLInverseReward(PyTorchEnv):
    """Gives a reward based on the inverse of the KL divergence between the expert and agent policy.
    """
    def __init__(self, env):
        super().__init__(env)
        
    def _generate_expert_reward(self, policy_state):
        """Generates the expert reward based on the policy state of the expert. 

        Arguments:
            policy_state {dict} -- The policy state of the expert.
        
        Returns:
            float -- The expert reward.
        """
        expert_policy_state = self._expert_policy_state
        
        return 0