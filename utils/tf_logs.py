import gin.tf
import numpy as np
import tensorflow as tf
  

def training_step_logs(writer, env, training_step, loss, action, state, weights):
    with writer.as_default():
        with tf.name_scope('Training'):
            tf.summary.scalar("Loss", loss, step=training_step)
            tf.summary.scalar("Selected Link", action, step=training_step)
            if env.link_traffic_to_states:
                link_utilization = np.mean(state[:env.n_links])
                tf.summary.scalar("Mean Link Utilization", link_utilization, step=training_step)
                tf.summary.scalar("Max Link Utilization", np.max(state[:env.n_links]), step=training_step)
                tf.summary.scalar("Min Link Utilization", np.min(state[:env.n_links]), step=training_step)
            if env.weigths_to_states:
                tf.summary.scalar("Weight mean", np.mean(weights), step=training_step)
                tf.summary.scalar("Weight max", np.max(weights), step=training_step)
                tf.summary.scalar("Weight min", np.min(weights), step=training_step)
                tf.summary.scalar("Weight std", np.std(weights), step=training_step)
        writer.flush()


def training_episode_logs(writer, env, episode, states, assigned_rewards, losses=None, actor_losses=None, critic_losses=None):
    with writer.as_default():
        with tf.name_scope('Training'):
            tf.summary.scalar("Reward mean", np.mean(assigned_rewards), step=episode)
            #tf.summary.scalar("Reward max", np.max(assigned_rewards), step=episode)
            #tf.summary.scalar("Reward min", np.min(assigned_rewards), step=episode)
            #if env.link_traffic_to_states:
                #mean_link_utilization = [np.mean(elem[:env.n_links]) for elem in states]
                #tf.summary.scalar("Mean Link Utilization mean", np.mean(mean_link_utilization), step=episode)
                #tf.summary.scalar("Mean Link Utilization max", np.max(mean_link_utilization), step=episode)
                #tf.summary.scalar("Mean Link Utilization min", np.min(mean_link_utilization), step=episode) 
            if losses is not None:
                tf.summary.scalar("Loss mean", np.mean(losses), step=episode)
            if actor_losses is not None:
                tf.summary.scalar("Actor Loss mean", np.mean(actor_losses), step=episode)
            if critic_losses is not None:
                tf.summary.scalar("Critic Loss mean", np.mean(critic_losses), step=episode)
        writer.flush()


def eval_step_logs(writer, env, eval_step, state, reward=None, prob=None, value=None):
    network = ('+').join(env.env_type)
    with writer.as_default():
        with tf.name_scope('Eval'):
            if reward is not None: 
                tf.summary.scalar("reward", reward, step=eval_step)
            if prob is not None: 
                tf.summary.scalar("Prob", prob, step=eval_step)
            if value is not None: 
                tf.summary.scalar("Value", value, step=eval_step)
                
            traffic = state[:env.n_links]
            #tf.summary.scalar("traffic mean", np.mean(traffic), step=eval_step)
            #tf.summary.scalar("traffic std", np.std(traffic), step=eval_step)
            #tf.summary.scalar(network + " - Max Traffic", np.max(traffic), step=eval_step)
            #tf.summary.scalar("traffic min", np.min(traffic), step=eval_step)
            
            weights = env.raw_weights
            #tf.summary.scalar("weights mean", np.mean(weights), step=eval_step)
            #tf.summary.scalar("weights std", np.std(weights), step=eval_step)
            #tf.summary.scalar("weights max", np.max(weights), step=eval_step)
            #tf.summary.scalar("weights min", np.min(weights), step=eval_step)
            tf.summary.scalar(network + " - Weights Diff Min Max", np.max(weights) - np.min(weights), step=eval_step)
        
        writer.flush()

def eval_final_log(writer, eval_episode, max_link_utilization, network):
    with writer.as_default():
        with tf.name_scope('Eval'):
            #tf.summary.scalar("Number of Nodes", num_nodes, step=eval_episode)
            #tf.summary.scalar("Number of Sample", num_sample, step=eval_episode)
            tf.summary.scalar(network + " - Starting Max LU", max_link_utilization[0], step=eval_episode)
            #idx_min_max = np.argmin(max_link_utilization)
            #tf.summary.scalar("Min Max LU", max_link_utilization[idx_min_max], step=eval_episode)
            #tf.summary.scalar("Min Max LU - Mean LU", mean_link_utilization[idx_min_max], step=eval_episode)
            #episode_length = len(max_link_utilization)
            idx_min_max = np.argmin(max_link_utilization)
            tf.summary.scalar(network + " - Min Max LU", max_link_utilization[idx_min_max], step=eval_episode)
            #idx_min_mean = np.argmin(mean_link_utilization)
            #tf.summary.scalar("Min Mean LU", mean_link_utilization[idx_min_mean], step=eval_episode)
            #tf.summary.scalar("Min Mean LU - Max LU", max_link_utilization[idx_min_mean], step=eval_episode)
        writer.flush()


def eval_top_log(writer, eval_episode, min_max, network):
    with writer.as_default():
        with tf.name_scope('Eval'):
            mean_min_max = np.mean(min_max)
            tf.summary.scalar(network + " - MEAN Min Max LU", mean_min_max, step=eval_episode)

        writer.flush()
