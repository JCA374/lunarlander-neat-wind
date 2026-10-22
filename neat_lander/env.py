"""Flying one LunarLander episode with a NEAT network, in wind."""

import math

import gymnasium as gym
import neat
import numpy as np

ENV_ID = 'LunarLander-v2'
MAX_STEPS = 1000
# Wind: a steady-ish sideways force plus a turning torque, both slowly varying gusts. The lander
# does not observe the wind; it has to notice the drift in its own velocity and correct for it.
WIND_POWER = 15.0
TURBULENCE_POWER = 1.5
# Landing between the flags (x = -0.2 .. 0.2 in observation units) earns up to CENTRE_BONUS extra
# fitness, the full amount exactly in the middle, falling to 0 at the flags.
CENTRE_BONUS = 50.0
PAD_HALF_WIDTH = 0.2
CENTRED = 0.1                   # |x| below this counts as a centred landing
# Render geometry of the 600x400 rgb_array frame (30 px per metre), used to draw flight paths
VIEW_W, VIEW_H, SCALE = 600, 400, 30.0


def make_env(render_mode=None, wind=True):
    return gym.make(ENV_ID, render_mode=render_mode, enable_wind=wind, wind_power=WIND_POWER,
                    turbulence_power=TURBULENCE_POWER)


def seed_wind(env, seed):
    """Gymnasium draws the gust phase from the global numpy RNG once per env, not from the reset
    seed. Tie it to the seed, so one seed means the same wind for every genome, every time."""
    rng = np.random.default_rng([int(seed), 7])
    env.unwrapped.wind_idx = int(rng.integers(-9999, 9999))
    env.unwrapped.torque_idx = int(rng.integers(-9999, 9999))


def steady_push(seed):
    """The constant sideways force of a curriculum flight: 30-100% of WIND_POWER, either direction."""
    rng = np.random.default_rng([int(seed), 11])
    return float(rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 1.0) * WIND_POWER)


def wind_force(env):
    """The sideways wind force the next env.step will apply (the same formula as gymnasium)."""
    u = env.unwrapped
    if not u.enable_wind or u.legs[0].ground_contact or u.legs[1].ground_contact:
        return 0.0
    k = u.wind_idx
    return math.tanh(math.sin(0.02 * k) + math.sin(math.pi * 0.01 * k)) * u.wind_power


def centre_bonus(landed, x):
    """Extra fitness for landing near the middle between the flags."""
    return CENTRE_BONUS * max(0.0, 1.0 - abs(x) / PAD_HALF_WIDTH) if landed else 0.0


def make_net(genome, config):
    """Feed-forward unless the config allows loops (feed_forward = False): then a recurrent network,
    where every connection reads the value its source node had one step earlier, so the network can
    carry information from step to step (memory)."""
    if config.genome_config.feed_forward:
        return neat.nn.FeedForwardNetwork.create(genome, config)
    return neat.nn.RecurrentNetwork.create(genome, config)


def node_values(net):
    """Every node's current value (the recurrent network keeps two buffers and flips between them)."""
    if isinstance(net.values, list):
        return dict(net.values[1 - net.active])
    return dict(net.values)


def choose_action(net, obs):
    """The network outputs one score per action; the highest score fires."""
    return int(np.argmax(net.activate(obs)))


def run_episode(net, seed, env=None, frames=False, path=False, trace=False, steady=0.0):
    """Fly one episode. Returns a dict with reward, steps, outcome and optional recordings:

    frames: rgb arrays, path: lander positions in frame pixels,
    trace: per-step observation, action and every node's value (for drawing a live network).
    'reward' is the plain env score; 'fitness' adds the centre-landing bonus.
    With frames, path or trace, 'wind' holds the sideways wind force at each step.
    steady (0..1, curriculum training only): blend Gymnasium's gusts (weight 1 - steady) with one constant
    push for the whole flight (weight steady, see steady_push). 0 is the real task.
    """
    own_env = env is None
    if own_env:
        env = make_env('rgb_array' if frames else None)
    seed_wind(env, seed)        # before reset: reset already takes one step in the wind
    u = env.unwrapped
    u.wind_power, u.turbulence_power = WIND_POWER * (1 - steady), TURBULENCE_POWER * (1 - steady)
    push = steady * steady_push(seed) if steady else 0.0
    if hasattr(net, 'reset'):
        net.reset()             # a recurrent network starts every flight with a blank memory
    obs, _ = env.reset(seed=int(seed))
    out = {'seed': int(seed), 'reward': 0.0, 'frames': [], 'path': [], 'states': [],
           'actions': [], 'values': [], 'wind': []}
    record_wind = frames or path or trace
    terminated = truncated = False
    reward = 0.0
    for _ in range(MAX_STEPS):
        if frames:
            out['frames'].append(env.render())
        if path:
            out['path'].append(_lander_px(env))
        if record_wind:
            out['wind'].append(wind_force(env) + (push if push and not _touching(env) else 0.0))
        action = choose_action(net, obs)
        if trace:
            out['states'].append(np.array(obs))
            out['actions'].append(action)
            out['values'].append(node_values(net))
        if push and not _touching(env):     # like Gymnasium's wind: only while both legs are off the ground
            u.lander.ApplyForceToCenter((push, 0.0), True)
        obs, reward, terminated, truncated, _ = env.step(action)
        out['reward'] += float(reward)
        out['steps'] = out.get('steps', 0) + 1
        if terminated or truncated:
            break
    if frames:
        out['frames'].append(env.render())
    if path:
        out['path'].append(_lander_px(env))
    if own_env:
        env.close()
    # The env ends with +100 when the lander comes to rest and -100 on a crash or leaving the screen
    if terminated and reward > 0:
        out['outcome'] = 'landed'
    elif terminated:
        out['outcome'] = 'crashed'
    else:
        out['outcome'] = 'timeout'
    out['landed'] = out['outcome'] == 'landed'
    out['x_final'] = float(obs[0])
    out['fitness'] = out['reward'] + centre_bonus(out['landed'], out['x_final'])
    return out


def _lander_px(env):
    p = env.unwrapped.lander.position
    return (p.x * SCALE, VIEW_H - p.y * SCALE)


def _touching(env):
    u = env.unwrapped
    return u.legs[0].ground_contact or u.legs[1].ground_contact
