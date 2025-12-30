
import habitat_sim
from habitat_sim import Simulator as Sim



def make_simple_cfg(settings):
    # simulator backend
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = settings["scene"]
    #sim_cfg.navmesh_settings = habitat_sim.nav.NavMeshSettings()
    #sim_cfg.navmesh_settings.agent_height = settings.get("agent_height", 1.5)
    #sim_cfg.navmesh_settings.agent_radius = settings.get("agent_radius", 0.17)

    # agent
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.action_space['turn_left'] = habitat_sim.agent.ActionSpec("turn_left", habitat_sim.agent.ActuationSpec(amount=30.0))
    agent_cfg.action_space['turn_right'] = habitat_sim.agent.ActionSpec("turn_right", habitat_sim.agent.ActuationSpec(amount=30.0))
    agent_cfg.action_space['move_forward'] = habitat_sim.agent.ActionSpec("move_forward", habitat_sim.agent.ActuationSpec(amount=0.25))
    agent_cfg.height = 1.41
    agent_cfg.radius = 0.17

    # In the 1st example, we attach only one sensor,
    # a RGB visual sensor, to the agent
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    rgb_sensor_spec.hfov = settings["hfov"]

    # Depth sensor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.hfov = settings["hfov"]

    agent_cfg.sensor_specifications = [rgb_sensor_spec, depth_sensor_spec]
    
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def get_simulator(scene_path, start_position, start_rotation):
    # initialize environment
    sim_settings = {
        "scene": scene_path,  # Scene path
        "default_agent": 0,  # Index of the default agent
        "sensor_height": 1.31,  # Height of sensors in meters, relative to the agent
        "width": 360,  # Spatial resolution of the observations
        "height": 640, 
        'hfov': 42
    }
    cfg = make_simple_cfg(sim_settings)
    sim = Sim(cfg)
    sim.recompute_navmesh(sim.pathfinder, habitat_sim.NavMeshSettings())
    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    agent_state.position = start_position
    agent_state.rotation = start_rotation
    agent.set_state(agent_state)
    return sim, agent