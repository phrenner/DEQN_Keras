import tensorflow as tf
import importlib
import keras
import os
from Model import Base_Model_Class

class InputGather(keras.layers.Layer):
    def call(self, x, inds):
        return tf.gather(x, inds, axis=-1)
    
class ConcatLayer(keras.layers.Layer):
    def call(self, lst):
        return tf.concat(lst, axis=-1)
    
class Model(Base_Model_Class):
    def __init__(self, MODEL_NAME,config,states,policy_states,policy_dict,current_episode,global_default_dtype):
        super().__init__(MODEL_NAME,config,states,policy_states,policy_dict,current_episode,global_default_dtype)
        self.config = config
        self.MODEL_NAME = MODEL_NAME
        self.states = states
        self.policy_states = policy_states
        self.policy_dict = policy_dict
        self.current_episode = current_episode
        if self.config["ENABLE_TB"]:
            self.writer = tf.summary.create_file_writer(os.getcwd())

        self.pred_epoch_loss_tracker = keras.metrics.MeanAbsoluteError(name="pred_loss")
        self.epoch_loss_tracker = keras.metrics.Mean(name="epoch_loss")
        self.net_epoch_loss_tracker = keras.metrics.Mean(name="net_epoch_loss")

        variables = importlib.import_module(MODEL_NAME + ".Variables")
        self.states_main = variables.states_main
        state_space_dim_ = len(self.states_main)

        tf.keras.backend.set_floatx(config.get('keras_precision','float32'))

        ### MOD FOR CUSTOM NET ###
        # get lists from Variables.py in the model's directory
        # list of states; those lists can overlap and are subset of list states and union is equal to states
        #get the list of indices; i.e. what place do the agent/econ states have in states list
        states_main_ = [s['name'] for s in variables.states_main]

        state_indx_main = []
        for indxmain in states_main_:
            state_indx_main.append(states.index(indxmain))

        #get the list of indices; i.e. what place do the agent/econ policy_states have in policy_states list
        policy_states_main_ = [s['name'] for s in  variables.policies_main]

        x_in = keras.Input(shape=(state_space_dim_,)) #define input vector variable with same dimension as state_space_dim

        #Define agents' net layer by layer
        rate = 0.2

    ##### First net for main policy states
        x_main = x_in #select subset of states to send to agent net
        hidden_layer_main_out = tf.keras.layers.Dense(
            units = 128, 
            activation = 'selu', 
            kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.01, 
                                                                    mode='fan_avg', 
                                                                    distribution='uniform', seed=1))(x_main)
        # hidden_layer_main_out = tf.keras.layers.Dropout(rate)(hidden_layer_main_out)
        hidden_layer_main_out = tf.keras.layers.Dense(
            units = 128, 
            activation = 'selu', 
            kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.01, 
                                                                    mode='fan_avg', 
                                                                    distribution='uniform', seed=2))(hidden_layer_main_out)
        # hidden_layer_main_out = tf.keras.layers.Dropout(rate)(hidden_layer_main_out)
        #output layer
        out_main = tf.keras.layers.Dense(
            units = len(policy_states_main_), 
            activation = 'linear', 
            kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.01, 
                                                                    mode='fan_avg', 
                                                                    distribution='uniform', seed=5))(hidden_layer_main_out)
    ##### Second net for func policy states
        # policy_states_func_ = [s['name'] for s in  variables.policies_func]
        # x_func = x_in #select subset of states to send to agent net
        # hidden_layer_func_out = tf.keras.layers.Dense(
        #     units = 512, 
        #     activation = 'selu', 
        #     kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.1, 
        #                                                             mode='fan_in', 
        #                                                             distribution='truncated_normal', seed=1))(x_func)
        # hidden_layer_func_out = tf.keras.layers.Dropout(rate)(hidden_layer_func_out)
        # hidden_layer_func_out = tf.keras.layers.Dense(
        #     units = 512, 
        #     activation = 'selu', 
        #     kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.1, 
        #                                                             mode='fan_in', 
        #                                                             distribution='truncated_normal', seed=2))(hidden_layer_func_out)
        # hidden_layer_func_out = tf.keras.layers.Dropout(rate)(hidden_layer_func_out)
        # #output layer
        # out_func = tf.keras.layers.Dense(
        #     units = len(policy_states_func_), 
        #     activation = 'linear', 
        #     kernel_initializer = tf.keras.initializers.VarianceScaling(scale=0.1, 
        #                                                             mode='fan_in', 
        #                                                             distribution='truncated_normal', seed=5))(hidden_layer_func_out)


        #joint the final outputs together so they have the same dimension as policy_states vector
        # output = ConcatLayer()([out_func, out_main])
        output = out_main

        #define the model
        self.policy_net = keras.Model(inputs=x_in, outputs=output, name="full_model")

        self.list_of_models = [
            self.policy_net
        ]

        lr = 1e-5
        self.optim_lst = [
            tf.keras.optimizers.Adam(
                learning_rate=lr, amsgrad=False,
                clipvalue=1.0)
        ]

        assert len(self.list_of_models) == len(self.optim_lst), "Number of models must match number of optimizers"        


    # Evaluate the neural net
    @tf.function
    def policy(self,s):
        raw_policy = self.policy_net(s[...,:len(self.states_main)])
        for i, pol in enumerate(self.policy_dict):
            if 'activation' in pol.keys():
                activation_str = pol['activation']
                if pol['activation'] == 'implied':
                    if 'lower' in pol['bounds'].keys() and 'upper' in pol['bounds'].keys():
                        activation_str = 'lambda x: {l} + ({u} - {l}) * tf.math.sigmoid(x)'.format(l=str(pol['bounds']['lower']), u=str(pol['bounds']['upper']))
                raw_policy = tf.tensor_scatter_nd_update(raw_policy,[[j,i] for j in range(s.shape[0])],eval(activation_str)(raw_policy[:,i]))        
                            
        if self.config["keras_precision"] == 'float64':
            return tf.cast(raw_policy, dtype=self.config.get('keras_precision','float32'))
        
        return raw_policy