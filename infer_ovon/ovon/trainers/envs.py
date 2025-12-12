# from typing import Optional

# import gym
# import habitat
# import numpy as np
# from habitat import Dataset
# from habitat.core.environments import RLTaskEnv
# from habitat.gym.gym_wrapper import HabGymWrapper


# class CustomRLTaskEnv(RLTaskEnv):
#     def after_update(self):
#         self._env.episode_iterator.after_update()


# @habitat.registry.register_env(name="CustomGymHabitatEnv")
# class CustomGymHabitatEnv(gym.Wrapper):
#     """
#     A registered environment that wraps a RLTaskEnv with the HabGymWrapper
#     to use the default gym API.
#     """

#     def __init__(
#         self, config: "DictConfig", dataset: Optional[Dataset] = None
#     ):
#         base_env = CustomRLTaskEnv(config=config, dataset=dataset)
#         env = HabGymWrapper(env=base_env)
#         super().__init__(env)
