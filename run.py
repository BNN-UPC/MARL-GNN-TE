import os
import sys

import numpy as np
import tensorflow as tf
from agents.ppo_agent import PPOAgent
from lib.run_experiment import Runner
from utils.functions import load_gin_configs

from absl import app
from absl import flags


flags.DEFINE_string('base_dir', 'logs',
                    'Base directory to host all required sub-directories.')
flags.DEFINE_string('network', 'logs',
                    'Base directory to host all required sub-directories.')
flags.DEFINE_multi_string(
    'gin_files', ["configs/general.gin", "configs/ppo.gin"], 'List of paths to gin configuration files')
flags.DEFINE_multi_string(
    'gin_bindings', [],
    'Gin bindings to override the values set in the config files ')

FLAGS = flags.FLAGS


def main(unused_argv):
    """Main method.
    Args:
      unused_argv: Arguments (unused).
    """
    load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr

    runner = Runner()

    f = open(os.path.join(runner.agent.writer_dir, 'out.txt'), 'w+')
    sys.stdout = f
    sys.stderr = f

    runner.run_experiment()

    sys.stdout = orig_stdout
    sys.stderr = orig_stderr
    f.close()


if __name__ == '__main__':
    app.run(main)
