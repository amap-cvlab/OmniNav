import habitat_sim
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
from omegaconf import DictConfig


@registry.register_simulator(name="OVONSim-v0")
class OVONSim(HabitatSim):
    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.navmesh_settings = self.load_navmesh_settings()
        self.recompute_navmesh(
            self.pathfinder,
            self.navmesh_settings,
            include_static_objects=False,
        )
        self.curr_scene_goals = {}

    def load_navmesh_settings(self):
        agent_cfg = self.habitat_config.agents.main_agent
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_height = agent_cfg.height
        navmesh_settings.agent_radius = agent_cfg.radius
        navmesh_settings.agent_max_climb = (
            self.habitat_config.navmesh_settings.agent_max_climb
        )
        navmesh_settings.cell_height = self.habitat_config.navmesh_settings.cell_height
        return navmesh_settings

    def reconfigure(
        self,
        habitat_config: DictConfig,
        should_close_on_new_scene: bool = True,
    ):
        is_same_scene = habitat_config.scene == self._current_scene
        super().reconfigure(habitat_config, should_close_on_new_scene)
        if not is_same_scene:
            self.recompute_navmesh(
                self.pathfinder,
                self.navmesh_settings,
                include_static_objects=False,
            )
            self.curr_scene_goals = {}
