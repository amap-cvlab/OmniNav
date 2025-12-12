from ovon import config
from ovon.dataset import ovon_dataset
from ovon.measurements import collision_penalty, nav, sum_reward
from ovon.models import (
    clip_policy,
    objaverse_clip_policy,
    ovrl_policy,
    transformer_policy,
)
from ovon.obs_transformers import (
    image_goal_encoder,
    relabel_imagegoal,
    relabel_teacher_actions,
    resize,
)
from ovon.task import rewards, sensors, simulator
from ovon.trainers import dagger_trainer, ppo_trainer_no_2d, ver_trainer
from ovon.utils import visualize_trajectories

try:
    import frontier_exploration
except ModuleNotFoundError as e:
    # If the error was due to the frontier_exploration package not being installed, then
    # pass, but warn. Do not pass if it was due to another package being missing.
    if e.name != "frontier_exploration":
        raise e
    else:
        print(
            "Warning: frontier_exploration package not installed. Things may not work. "
            "To install:\n"
            "git clone git@github.com:naokiyokoyama/frontier_exploration.git &&\n"
            "cd frontier_exploration && pip install -e ."
        )
