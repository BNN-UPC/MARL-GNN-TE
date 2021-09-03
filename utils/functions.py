import gin.tf
from itertools import tee
import numpy as np


def load_gin_configs(gin_files, gin_bindings):
    """Loads gin configuration files.
    Args:
      gin_files: list, of paths to the gin configuration files for this
        experiment.
      gin_bindings: list, of gin parameter bindings to override the values in
        the config files.
    """
    gin.parse_config_files_and_bindings(gin_files,
                                        bindings=gin_bindings,
                                        skip_unknown=False)


def linearly_decaying_epsilon(decay_period, step, warmup_steps, epsilon):
    """Returns the current epsilon for the agent's epsilon-greedy policy.
    This follows the Nature DQN schedule of a linearly decaying epsilon (Mnih et
    al., 2015). The schedule is as follows:
      Begin at 1. until warmup_steps steps have been taken; then
      Linearly decay epsilon from 1. to epsilon in decay_period steps; and then
      Use epsilon from there on.
    Args:
      decay_period: float, the period over which epsilon is decayed.
      step: int, the number of training steps completed so far.
      warmup_steps: int, the number of steps taken before epsilon is decayed.
      epsilon: float, the final value to which to decay the epsilon parameter.
    Returns:
      A float, the current epsilon value computed according to the schedule.
    """
    steps_left = decay_period + warmup_steps - step
    bonus = (1.0 - epsilon) * steps_left / decay_period
    bonus = np.clip(bonus, 0., 1. - epsilon)
    return epsilon + bonus


def pairwise_iteration(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def find_min_max_path_length(path_lengths):
    max_length = 0.0
    min_length = 100.0
    for source in path_lengths.keys():
      for dest in path_lengths[source].keys():
        length = path_lengths[source][dest]
        if length > max_length:
          max_length = length
        elif length < min_length:
          min_length = length
    return min_length, max_length


def get_traffic_matrix(tm_file, nodes):
    tm = np.zeros((nodes,nodes))
    with open(tm_file) as fd:
        fd.readline()
        fd.readline()
        for line in fd:
            camps = line.split(" ")
            tm[int(camps[1]),int(camps[2])] = float(camps[3])
    return (tm)
