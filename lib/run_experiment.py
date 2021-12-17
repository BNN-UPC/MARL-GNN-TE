
from environment.environment import Environment
from agents.ppo_agent import PPOAgent

import os
import logging
import tensorflow as tf

import gin.tf

@gin.configurable
class Runner(object):
    
    def __init__(self, 
                 algorithm='PPO',
                 reload_model=False,
                 model_dir=None,
                 only_eval=False,
                 base_dir='logs',
                 checkpoint_base_dir='checkpoints',
                 save_checkpoints=True):
        
        env = Environment()
        self.save_checkpoints = save_checkpoints
        if algorithm == 'PPO':
            if reload_model:
                old_actor = False #if '_agg' in model_dir else True
                self.agent = PPOAgent(env, old_actor=old_actor, save_checkpoints=save_checkpoints)
            else:
                self.agent = PPOAgent(env, save_checkpoints=save_checkpoints)
        else:
            #Insert  here your customized RL algorithm 
            assert (False), 'RL Algorithm %s is not implemented' %algorithm
        
        self.base_dir= base_dir
        self.checkpoint_base_dir = checkpoint_base_dir
        
        self.only_eval = only_eval
        if reload_model or self.only_eval:
            self.agent.load_saved_model(model_dir, only_eval)
        self.set_logs_and_checkpoints()
    
    def run_experiment(self):
        if self.only_eval:
            self.agent.only_evaluate()
        else:
            self.agent.train_and_evaluate()

    def set_logs_and_checkpoints(self):
        experiment_identifier = self.agent.set_experiment_identifier(self.only_eval)

        writer_dir = os.path.join(self.base_dir, experiment_identifier)
        if not os.path.exists(writer_dir):
            os.makedirs(writer_dir)
        
        checkpoint_dir = os.path.join(self.checkpoint_base_dir, experiment_identifier)
        if self.save_checkpoints and (not os.path.exists(checkpoint_dir)):
            os.makedirs(checkpoint_dir)

        self.agent.set_writer_and_checkpoint_dir(writer_dir, checkpoint_dir)
        
        f = open(os.path.join(writer_dir, 'out.log'), 'w+')
        f.close()
        fh = logging.FileHandler(os.path.join(writer_dir, 'out.log'))
        fh.setLevel(logging.DEBUG) # or any level you want
        tf.get_logger().addHandler(fh)

    