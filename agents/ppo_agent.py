from tensorflow import keras
import tensorflow as tf
import tensorflow_probability as tfp
from environment.environment import Environment
from lib.actor import  Actor
from lib.critic import Critic
from utils.functions import linearly_decaying_epsilon
from utils.defo_process_results import get_traffic_matrix
import utils.tf_logs as tf_logs
import copy
import numpy as np
import random
import os
import logging
import time
import csv

import gin.tf


@gin.configurable
class PPOAgent(object):
    '''An implementation of a GNN-based PPO Agent'''

    def __init__(self, 
                 env,
                 eval_env_type=['GBN','NSFNet','GEANT2'],
                 num_eval_samples=10,
                 clip_param=0.2,
                 critic_loss_factor=0.5,
                 entropy_loss_factor=0.001,
                 normalize_advantages=True,
                 max_grad_norm=1.0,
                 gamma=0.99,
                 gae_lambda=0.95,
                 horizon=None,
                 batch_size=25,
                 epochs=3,
                 last_training_sample=99,
                 eval_period=50,
                 max_evals=100,
                 select_max_action=False,
                 optimizer=tf.keras.optimizers.Adam(
                   learning_rate=0.0003,
                   beta_1=0.9,
                   epsilon=0.00001),
                 change_traffic = False,
                 change_traffic_period = 1,
                 base_dir='logs',
                 checkpoint_base_dir='checkpoints', 
                 save_checkpoints=True):

        self.env = env
        self.eval_env_type = eval_env_type
        self.num_eval_samples = num_eval_samples
        self.clip_param = clip_param

        self._get_actor_critic_functions()
        self.num_actions = self.env.n_links

        self.optimizer = optimizer
        self.critic_loss_factor = critic_loss_factor
        self.entropy_loss_factor = entropy_loss_factor
        self.normalize_advantages = normalize_advantages
        self.max_grad_norm = max_grad_norm

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.given_horizon = horizon
        self.define_horizon()
        self.epochs = epochs
        self.batch_size = batch_size
        self.last_training_sample = last_training_sample
        self.eval_period = eval_period
        self.max_evals = max_evals
        self.select_max_action = select_max_action
        self.change_traffic = change_traffic
        self.change_traffic_period = change_traffic_period
        self.eval_step = 0
        self.eval_episode = 0 
        self.base_dir= base_dir
        self.checkpoint_base_dir = checkpoint_base_dir
        self.save_checkpoints = save_checkpoints
        self.reload_model = False
        self.change_sample = False

    def _get_actor_critic_functions(self):
        self.actor = Actor(self.env.G, num_features=self.env.num_features)
        self.actor.build()
        self.critic = Critic(self.env.G, num_features=self.env.num_features)
        self.critic.build()

    def define_horizon(self):
        if self.given_horizon is not None:
            self.horizon = self.given_horizon
        elif self.env.network == 'NSFNet':
            self.horizon = 100
        elif self.env.network == 'GBN':
            self.horizon = 150
        elif self.env.network == 'GEANT2':
            self.horizon = 200
        else:
            self.horizon = 200
    
    def reset_env(self):
        self.env.reset(change_sample=self.change_sample)
        if self.change_sample and len(self.env.env_type) > 1:
            actor_model = copy.deepcopy(self.actor.trainable_variables)
            critic_model = copy.deepcopy(self.critic.trainable_variables)
            self._get_actor_critic_functions()
            self.load_model(actor_model, critic_model)
            self.define_horizon()
        self.change_sample = False
    

    def gae_estimation(self, rewards, values, last_value):
        last_gae_lambda = 0
        advantages = np.zeros_like(values, dtype=np.float32)
        for i in reversed(range(self.horizon)):
            if i == self.horizon - 1:
                next_value = last_value
            else:
                next_value = values[i+1]
            delta = rewards[i] + self.gamma * next_value  - values[i]
            advantages[i] = last_gae_lambda = delta + self.gamma * self.gae_lambda * last_gae_lambda
        returns = values + advantages
        if self.normalize_advantages:
            advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return returns, advantages 


    def run_episode(self):
        # reset state at the beginning of each iteration
        self.reset_env()
        state = self.env.get_state()
        states = np.zeros((self.horizon, self.env.n_links*self.actor.num_features), dtype=np.float32)
        actions = np.zeros(self.horizon, dtype=np.float32)
        rewards = np.zeros(self.horizon, dtype=np.float32)
        log_probs = np.zeros(self.horizon, dtype=np.float32) 
        values = np.zeros(self.horizon, dtype=np.float32)

        for t in range(self.horizon):
            action, log_prob = self.act(state)
            value = self.run_critic(state)
            next_state, reward = self.env.step(action.numpy())
            states[t] = state
            actions[t] = action
            rewards[t] = reward
            log_probs[t] = log_prob
            values[t] = value.numpy()[0]
            state = next_state
        value = self.run_critic(state)
        last_value = value.numpy()[0]

        return states, actions, rewards, log_probs, values, last_value

    def run_update(self, training_episode, states, actions, returns, advantages, log_probs):
        actor_losses, critic_losses, losses = [], [], []
        inds = np.arange(self.horizon)
        for _ in range(self.epochs):
            np.random.shuffle(inds) 
            for start in range(0, self.horizon, self.batch_size):
                end = start + self.batch_size
                minibatch_ind = inds[start:end]
                actor_loss, critic_loss, loss, grads = self.compute_losses_and_grads(states[minibatch_ind], 
                                                                actions[minibatch_ind], returns[minibatch_ind], 
                                                                advantages[minibatch_ind], log_probs[minibatch_ind])
                self.apply_grads(grads)
                actor_losses.append(actor_loss.numpy())
                critic_losses.append(critic_loss.numpy())
                losses.append(loss.numpy())

        return actor_losses, critic_losses, losses


    def train_and_evaluate(self):
        training_episode = -1
        while not (self.env.num_sample == self.last_training_sample and self.change_sample == True):
            training_episode += 1
            print('Episode ', training_episode, '...')
            states, actions, rewards, log_probs, values, last_value = self.run_episode()
            returns, advantages = self.gae_estimation(rewards, values, last_value)

            actor_losses, critic_losses, losses = self.run_update(training_episode, states, actions, returns, advantages, log_probs)
            tf_logs.training_episode_logs(self.writer, self.env, training_episode, states, rewards, losses, actor_losses, critic_losses)
            
            if (training_episode+1) % self.eval_period == 0:
                self.training_eval()
                if self.save_checkpoints:
                    self.actor._set_inputs(states[0])
                    self.critic._set_inputs(states[0])
                    self.save_model(os.path.join(self.checkpoint_dir, 'episode'+str(self.eval_episode)))
                if self.change_traffic and self.eval_episode % self.change_traffic_period == 0:
                    self.change_sample = True
            

    def only_evaluate(self):
        self.env.initialize_environment(num_sample=self.last_training_sample+1)
        for _ in range(self.max_evals):
            self.evaluation()
            self.change_sample = True
    
    def generate_eval_env(self):
        self.eval_envs = {}
        for eval_env_type in self.eval_env_type: 
            self.eval_envs[eval_env_type] = Environment(env_type=eval_env_type,
                                                        traffic_profile=self.env.traffic_profile,
                                                        routing=self.env.routing)

    def generate_eval_actor_critic_functions(self):
        self.eval_actor = {}
        self.eval_critic = {}
        for eval_env_type in self.eval_env_type:
            self.eval_actor[eval_env_type] = Actor(self.eval_envs[eval_env_type].G, num_features=self.env.num_features)
            self.eval_actor[eval_env_type].build()
            self.eval_critic[eval_env_type] = Critic(self.eval_envs[eval_env_type].G, num_features=self.env.num_features)
            self.eval_critic[eval_env_type].build()

    def update_eval_actor_critic_functions(self):
        for eval_env_type in self.eval_env_type:
            for w_model, w_eval_actor in zip(self.actor.trainable_variables, 
                                        self.eval_actor[eval_env_type].trainable_variables):
                w_eval_actor.assign(w_model)
            for w_model, w_eval_critic in zip(self.critic.trainable_variables, 
                                        self.eval_critic[eval_env_type].trainable_variables):
                w_eval_critic.assign(w_model)

    def training_eval(self):
        # Evaluation phase
        print('\n\tEvaluation ' + str(self.eval_episode) + '...\n')

        if self.eval_episode == 0:
            self.generate_eval_env()
            self.generate_eval_actor_critic_functions()
        
        self.update_eval_actor_critic_functions()

        for eval_env_type in self.eval_env_type:
            self.eval_envs[eval_env_type].define_num_sample(100)
        
            total_min_max = []
            mini_eval_episode = self.eval_episode * self.num_eval_samples
            for _ in range(self.num_eval_samples):
                self.eval_envs[eval_env_type].reset(change_sample=True)
                state = self.eval_envs[eval_env_type].get_state()
                tf_logs.eval_step_logs(self.writer, self.eval_envs[eval_env_type], self.eval_step, state)
                if self.eval_envs[eval_env_type].link_traffic_to_states:
                    max_link_utilization = [np.max(state[:self.eval_envs[eval_env_type].n_links])]
                    mean_link_utilization = [np.mean(state[:self.eval_envs[eval_env_type].n_links])]
                probs, values = [], []

                for i in range(self.horizon):
                    self.eval_step += 1
                    action, log_prob = self.eval_act(self.eval_actor[eval_env_type], state, select_max=self.select_max_action)
                    value = self.eval_critic[eval_env_type](state)
                    next_state, reward = self.eval_envs[eval_env_type].step(action.numpy())
                    probs.append(np.exp(log_prob))
                    values.append(value.numpy()[0])
                    state = next_state
                    if self.eval_envs[eval_env_type].link_traffic_to_states:
                        max_link_utilization.append(np.max(state[:self.eval_envs[eval_env_type].n_links]))
                        #mean_link_utilization.append(np.mean(state[:self.eval_envs[eval_env_type].n_links]))

                    tf_logs.eval_step_logs(self.writer, self.eval_envs[eval_env_type], self.eval_step, state)
                
                if self.env.link_traffic_to_states:
                    total_min_max.append(np.min(max_link_utilization))
                    #tf_logs.eval_final_log(self.writer, mini_eval_episode, max_link_utilization, eval_env_type)
                mini_eval_episode += 1

            tf_logs.eval_top_log(self.writer, self.eval_episode, total_min_max, eval_env_type)
        self.eval_episode += 1
        

    def evaluation(self):
        # Evaluation phase
        print('\n\tEvaluation ' + str(self.eval_episode) + '...\n')
        self.reset_env()
        state = self.env.get_state()
        tf_logs.eval_step_logs(self.writer, self.env, self.eval_step, state)
        if self.env.link_traffic_to_states:
            max_link_utilization = [np.max(state[:self.env.n_links])]
            mean_link_utilization = [np.mean(state[:self.env.n_links])]
        probs, values = [], []

        for i in range(self.horizon):
            self.eval_step += 1
            action, log_prob = self.act(state, select_max=self.select_max_action)
            value = self.run_critic(state)
            next_state, reward = self.env.step(action.numpy())
            probs.append(np.exp(log_prob))
            values.append(value.numpy()[0])
            state = next_state
            if self.env.link_traffic_to_states:
                max_link_utilization.append(np.max(state[:self.env.n_links]))
                mean_link_utilization.append(np.mean(state[:self.env.n_links]))

            tf_logs.eval_step_logs(self.writer, self.env, self.eval_step, state, reward, probs[i], values[i])
        
        if self.env.link_traffic_to_states:
            tf_logs.eval_final_log(self.writer, self.eval_episode, max_link_utilization, ('+').join(self.env.env_type))
            if self.only_eval: self.write_eval_results(self.eval_episode, np.min(max_link_utilization))
        
        #self.eval_step += 10
        self.eval_episode += 1

    @tf.function
    def compute_actor_loss(self, new_log_probs, old_log_probs, advantages):  
        ratio = tf.exp(new_log_probs - old_log_probs)
        pg_loss_1 = - advantages * ratio
        pg_loss_2 = - advantages * tf.clip_by_value(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
        actor_loss = tf.reduce_mean(tf.maximum(pg_loss_1, pg_loss_2))
        return actor_loss

    @tf.function
    def get_new_log_prob_and_entropy(self, state, action):
        logits = self.actor(state, training=True)
        probs = tfp.distributions.Categorical(logits=logits)
        return (probs.log_prob(action), probs.entropy())

    @tf.function
    def compute_losses_and_grads(self, states, actions, returns, advantages, old_log_probs):
        with tf.GradientTape(persistent=True) as tape:
            new_log_probs, entropy = tf.map_fn(lambda x: self.get_new_log_prob_and_entropy(x[0], x[1]), (states, actions), fn_output_signature=(tf.float32, tf.float32))

            values = tf.map_fn(lambda x: self.critic(x, training=True), states, fn_output_signature=tf.float32)
            values = tf.reshape(values, [-1])

            critic_loss = tf.reduce_mean(tf.square(returns - values))
            entropy_loss = tf.reduce_mean(entropy)
            actor_loss = self.compute_actor_loss(new_log_probs, old_log_probs, advantages)
            loss = actor_loss - self.entropy_loss_factor*entropy_loss + self.critic_loss_factor*critic_loss
            
        grads = tape.gradient(loss, self.actor.trainable_variables+self.critic.trainable_variables)
        if self.max_grad_norm is not None:
            # Clip the gradients (normalize)
            grads, _grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)
        
        return actor_loss, critic_loss, loss, grads

    def apply_grads(self, grads):
        self.optimizer.apply_gradients(zip(grads, self.actor.trainable_variables+self.critic.trainable_variables))

    @tf.function
    def act(self, state, select_max=False):
        logits = self.actor(state)
        probs = tfp.distributions.Categorical(logits=logits)
        if select_max:
            action = tf.argmax(logits)
        else:
            action = probs.sample()

        return action, probs.log_prob(action)

    @tf.function
    def eval_act(self, actor, state, select_max=False):
        logits = actor(state)
        probs = tfp.distributions.Categorical(logits=logits)
        if select_max:
            action = tf.argmax(logits)
        else:
            action = probs.sample()

        return action, probs.log_prob(action)

    @tf.function
    def run_critic(self, state):
        return self.critic(state)

    
    def save_model(self, checkpoint_dir):
        self.actor.save(checkpoint_dir+'/actor')
        self.critic.save(checkpoint_dir+'/critic')


    def load_model(self, actor_model, critic_model):
        for w_model, w_actor in zip(actor_model, 
                                     self.actor.trainable_variables):
            w_actor.assign(w_model)
        for w_model, w_critic in zip(critic_model, 
                                     self.critic.trainable_variables):
            w_critic.assign(w_model)


    def load_saved_model(self, model_dir, only_eval):
        model = keras.models.load_model(model_dir+'/actor')
        for w_model, w_actor in zip(model.trainable_variables, 
                                     self.actor.trainable_variables):
            w_actor.assign(w_model)
        if not only_eval:
            model = keras.models.load_model(model_dir+'/critic')
            for w_model, w_critic in zip(model.trainable_variables, 
                                        self.critic.trainable_variables):
                w_critic.assign(w_model)
        self.model_dir = model_dir
        self.reload_model = True

    def write_eval_results(self, step, value):
        csv_dir = os.path.join('./notebooks/logs', self.experiment_identifier)
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir)
        with open(csv_dir+'/results.csv', "a") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerow([step,value])

    def set_experiment_identifier(self, only_eval):
        self.only_eval = only_eval
        mode = 'eval' if only_eval else 'training'

        if mode == 'training': 
            #ENVIRONMENT
            network = '+'.join([str(elem) for elem in self.env.env_type])
            traffic_profile = self.env.traffic_profile
            routing = self.env.routing
            env_folder = ('-').join([network,traffic_profile,routing])

            #PPOAGENT
            batch = 'batch'+str(self.batch_size)
            gae_lambda = 'gae'+str(self.gae_lambda)
            lr = 'lr'+str(self.optimizer.get_config()['learning_rate'])
            epsilon = 'epsilon'+str(self.optimizer.epsilon)
            clip = 'clip'+str(self.clip_param)
            gamma = 'gamma'+str(self.gamma)
            period = 'period'+str(self.eval_period)
            epoch = 'epoch'+str(self.epochs)
            agent_folder = ('-').join([batch,lr,epsilon,gae_lambda,clip,gamma,period,epoch])

            #ACTOR-CRITIC
            state_size = 'size'+str(self.actor.link_state_size)
            iters = 'iters'+str(self.actor.message_iterations)
            aggregation = self.actor.aggregation
            nn_size = 'nnsize'+str(self.actor.final_hidden_layer_size)
            dropout = 'drop'+str(self.actor.dropout_rate)
            activation = self.actor.activation_fn
            function_folder = ('-').join([state_size,iters,aggregation,nn_size,dropout,activation])

            self.experiment_identifier = os.path.join(mode, env_folder, agent_folder, function_folder)
        
        else:
            model_dir = self.model_dir

            network = '+'.join([str(elem) for elem in self.env.env_type])
            traffic_profile = self.env.traffic_profile
            routing = self.env.routing
            eval_env_folder = ('-').join([network,traffic_profile,routing])

            #RELOADED MODEL
            env_folder = os.path.join(model_dir.split('/')[3])
            agent_folder = os.path.join(model_dir.split('/')[4])
            function_folder = os.path.join(model_dir.split('/')[5])
            episode = os.path.join(model_dir.split('/')[6])

            self.experiment_identifier = os.path.join(mode, eval_env_folder, env_folder, agent_folder, function_folder, episode)

        return self.experiment_identifier


    def set_writer_and_checkpoint_dir(self, writer_dir, checkpoint_dir):
        self.writer_dir = writer_dir
        self.checkpoint_dir = checkpoint_dir
        self.writer = tf.summary.create_file_writer(self.writer_dir)