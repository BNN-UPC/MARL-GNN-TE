import tensorflow as tf
from tensorflow import keras
import numpy as np
import networkx

import gin.tf


@gin.configurable
class Critic(keras.Model):
    def __init__(self, 
                 graph, 
                 num_features = 2,
                 link_state_size=16,
                 #message_hidden_layer_size=64,
                 aggregation='min_max',
                 first_hidden_layer_size=128,
                 dropout_rate=0.5,
                 final_hidden_layer_size=64,
                 message_iterations = 8,
                 activation_fn='tanh',
                 final_activation_fn='linear'):
        super(Critic, self).__init__()
        # HYPERPARAMETERS
        self.num_features = num_features
        self.n_links = graph.number_of_edges()
        self.link_state_size = link_state_size 
        self.message_hidden_layer_size = final_hidden_layer_size
        self.aggregation = aggregation
        self.message_iterations = message_iterations
        
        self.num_readout_input_aggregations = 4

        # FIXED INPUTS
        # for a link-link connection "i", self.incoming_links[i] is the incoming link
        # and self.outcoming_links[i] is the outcoming_link
        # see environment class function "_generate_link_indices_and_adjacencies()" for details 
        self.incoming_links = graph.nodes()['graph_data']['incoming_links']
        self.outcoming_links = graph.nodes()['graph_data']['outcoming_links']

        # NEURAL NETWORKS
        self.hidden_layer_initializer = tf.keras.initializers.Orthogonal(gain=np.sqrt(2))
        self.final_layer_initializer = tf.keras.initializers.Orthogonal(gain=1)
        self.kernel_regularizer = None #keras.regularizers.l2(0.01)
        #keras.initializers.VarianceScaling(scale=1.0 / np.sqrt(3.0),mode='fan_in', distribution='uniform')
        self.activation_fn = activation_fn 
        self.final_hidden_layer_size = final_hidden_layer_size
        self.first_hidden_layer_size = first_hidden_layer_size
        self.dropout_rate = dropout_rate
        self.final_activation_fn = final_activation_fn
        self.define_network()


    def define_network(self):
        # message
        self.create_message = keras.models.Sequential(name='create_message')
        self.create_message.add(keras.layers.Dense(self.message_hidden_layer_size,
                kernel_initializer=self.hidden_layer_initializer, activation=self.activation_fn))
        self.create_message.add(keras.layers.Dense(self.link_state_size,
                kernel_initializer=self.hidden_layer_initializer, activation=self.activation_fn))

        # link update
        self.link_update = keras.models.Sequential(name='link_update')
        self.link_update.add(keras.layers.Dense(self.first_hidden_layer_size,
                kernel_initializer=self.hidden_layer_initializer, activation=self.activation_fn))
        self.link_update.add(keras.layers.Dense(self.final_hidden_layer_size,
                kernel_initializer=self.hidden_layer_initializer, activation=self.activation_fn))
        self.link_update.add(keras.layers.Dense(self.link_state_size,
                kernel_initializer=self.hidden_layer_initializer, activation=self.activation_fn))

        # readout
        self.readout = keras.models.Sequential(name='readout')
        self.readout.add(keras.layers.Dense(self.first_hidden_layer_size, kernel_initializer=self.hidden_layer_initializer, 
                                kernel_regularizer=self.kernel_regularizer, activation=self.activation_fn))
        self.readout.add(keras.layers.Dropout(self.dropout_rate))
        self.readout.add(keras.layers.Dense(self.final_hidden_layer_size, kernel_initializer=self.hidden_layer_initializer, 
                                kernel_regularizer=self.kernel_regularizer, activation=self.activation_fn))
        self.readout.add(keras.layers.Dropout(self.dropout_rate))
        self.readout.add(keras.layers.Dense(1, kernel_initializer=self.final_layer_initializer, 
                                kernel_regularizer=self.kernel_regularizer, activation=self.final_activation_fn))
       

    def build(self, input_shape=None):
        del input_shape
        self.create_message.build(input_shape = [None, 2 * self.link_state_size])
        if self.aggregation == 'sum':
            self.link_update.build(input_shape = [None, 2 * self.link_state_size])
        elif self.aggregation == 'min_max':
            self.link_update.build(input_shape = [None, 3 * self.link_state_size])
        self.readout.build(input_shape = [None, self.link_state_size*self.num_readout_input_aggregations])
        self.built = True

    @tf.function
    def message_passing(self, input):
        input_tensor = tf.convert_to_tensor(input)
        link_states = tf.reshape(input_tensor, [self.num_features,self.n_links])
        link_states = tf.transpose(link_states)
        padding = [[0,0],[0,self.link_state_size-self.num_features]]
        link_states = tf.pad(link_states, padding)

        # message passing part
        # links exchange information with their neighbors to update their states
        for _ in range(self.message_iterations):
            incoming_link_states = tf.gather(link_states, self.incoming_links)
            outcoming_link_states = tf.gather(link_states, self.outcoming_links)
            message_inputs = tf.cast(tf.concat([incoming_link_states, outcoming_link_states], axis=1), tf.float32)
            messages = self.create_message(message_inputs)

            aggregated_messages = self.message_aggregation(messages)
            link_update_input = tf.cast(tf.concat([link_states, aggregated_messages], axis=1), tf.float32)
            link_states = self.link_update(link_update_input)

        return link_states

    @tf.function
    def message_aggregation(self, messages):
        if self.aggregation == 'sum':
            aggregated_messages = tf.math.unsorted_segment_sum(messages, self.outcoming_links, num_segments=self.n_links)
        elif self.aggregation == 'min_max':
            agg_max = tf.math.unsorted_segment_max(messages, self.outcoming_links, num_segments=self.n_links)
            agg_min = tf.math.unsorted_segment_min(messages, self.outcoming_links, num_segments=self.n_links)
            aggregated_messages = tf.concat([agg_max, agg_min], axis=1)
        return aggregated_messages
    
    @tf.function
    def generate_readout_input(self, link_states):
        ls_mean = tf.reduce_mean(link_states, axis=0)
        ls_max = tf.reduce_max(link_states, axis=0)
        ls_min = tf.reduce_min(link_states, axis=0)
        ls_std = tf.math.reduce_std(link_states, axis=0)

        readout_input = tf.concat([ls_mean,ls_max,ls_min,ls_std], axis=0)
        readout_input = tf.expand_dims(readout_input, axis=0)
        return readout_input

    @tf.function
    def call(self, input):
        
        link_states = self.message_passing(input)
        #link_states = tf.expand_dims(link_states, axis=0)

        readout_input = self.generate_readout_input(link_states)

        V = self.readout(readout_input)
        V = tf.reshape(V, [-1])

        return V

