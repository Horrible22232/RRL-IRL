from neroRRL.expert.modules.dreamerv3.embodied.envs import from_gym
from neroRRL.expert.modules.dreamerv3.embodied.core import Checkpoint
from neroRRL.expert.modules import dreamerv3
import numpy as np
from pathlib import Path


base_path = "./model/expert/crafter/"
config_path = Path(base_path + 'config.yaml')
model_path = Path(base_path + 'checkpoint.ckpt')

checkpoint = Checkpoint()
checkpoint.agent = agent