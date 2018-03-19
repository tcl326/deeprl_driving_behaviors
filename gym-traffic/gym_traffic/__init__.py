from gym.envs.registration import register

register(
    id='Traffic-Simple-gui-v0',
    entry_point='gym_traffic.envs:TrafficEnvSimple',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "gui"},
    nondeterministic=True
)

register(
    id='Traffic-Simple-cli-v0',
    entry_point='gym_traffic.envs:TrafficEnvSimple',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "cli"},
    nondeterministic=True
)

register(
    id='Traffic-Cross2-gui-v0',
    entry_point='gym_traffic.envs:TrafficEnvCross2',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "gui"},
    nondeterministic=True
)

register(
    id='Traffic-Cross2-cli-v0',
    entry_point='gym_traffic.envs:TrafficEnvCross2',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "cli"},
    nondeterministic=True
)

register(
    id='Traffic-Multi-cli-v0',
    entry_point='gym_traffic.envs:TrafficEnvMultiAgentsSimple',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "cli"},
    nondeterministic=True
)

register(
    id='Traffic-Multi-gui-v0',
    entry_point='gym_traffic.envs:TrafficEnvMultiAgentsSimple',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1000},
    kwargs={"mode": "gui"},
    nondeterministic=True
)
