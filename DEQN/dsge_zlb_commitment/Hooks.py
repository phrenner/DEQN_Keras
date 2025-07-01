from Parameters import policy, states, policy_states, definitions, global_default_dtype, sigma_a_x, sigma_xi_x, rho_a_x
from Parameters import rho_xi_x, sigma_g_x, rho_g_x, ENABLE_TB
import PolicyState
import State
import Definitions
import tensorflow as tf
import Parameters
import keras
import numpy as np
import importlib

Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")

def LearningRateScheduler(epoch,lr):
    return 1e-5
    
class TensorboardCallback(keras.callbacks.Callback):
    def __init__(self, last_state, penalty_bounds_policy):
        super().__init__()    
        self.last_state = last_state
        self.penalty_bounds_policy = penalty_bounds_policy
    def on_epoch_end(self, epoch, logs=None):
        epoch_loss, net_epoch_loss = logs["epoch_loss"], logs["net_epoch_loss"]
        if ENABLE_TB:
            with Parameters.model.writer.as_default():
                i = int(Parameters.current_episode.numpy())
                state = self.last_state
                policy_state = policy(state)
                # for s in states:
                #     tf.summary.histogram("hist_s_" + s, getattr(State,s)(state), step=i)
                    
                # for p in policy_states:
                #     tf.summary.histogram("hist_p_" + p, getattr(PolicyState,p)(policy_state), step=i)
                    
                for d in definitions:
                    tf.summary.histogram("hist_d_" + d, getattr(Definitions,d)(state, policy_state), step=i)

                n_batches = state.shape[0]
                losses = Equations.equations(state, policy_state)
                
                for eq_f in losses.keys():
                    tf.summary.scalar('dev_' + eq_f, tf.math.abs(tf.reduce_mean(losses[eq_f])))

                    
                penalty_bounds, no_bounds = self.penalty_bounds_policy(state, policy_state, ENABLE_TB)
                
                tf.summary.scalar(f'dev_loss', epoch_loss)
                tf.summary.scalar(f'dev_net_loss', net_epoch_loss)    

def post_init():

    vartheta_old = -0.019182
    rho_old = 0.016500
    c_old = 0.921336
    varphi_old = -0.000012
    i_nom_old = 0.002461

    nB = Parameters.starting_state.shape[0]
    normal_noise = tf.random.normal(
        [nB,6],
        mean=0.0,
        stddev=tf.constant([[0.027,0.023,0.052,0.0001,0.001,0.001]],dtype=global_default_dtype), # vartheta, rho, c, disp, i_nom, varphi
        dtype=global_default_dtype)

    init_state = Parameters.starting_state
    init_state = State.update(init_state,'vartheta_old_x', 
                tf.constant(vartheta_old,shape=(nB,),dtype=global_default_dtype) + normal_noise[:,0] )
    init_state =State.update(init_state,'rho_old_x', 
                tf.constant(rho_old,shape=(nB,),dtype=global_default_dtype) + normal_noise[:,1])
    init_state =State.update(init_state,'c_old_x', 
                tf.constant(c_old,shape=(nB,),dtype=global_default_dtype) + normal_noise[:,2])
    init_state = State.update(init_state,'disp_old_x', 
                 tf.constant(1.,shape=(nB,),dtype=global_default_dtype) + normal_noise[:,3])
    init_state = State.update(init_state,'i_nom_old_x', 
                tf.maximum(tf.zeros(nB,dtype=global_default_dtype),tf.constant(i_nom_old,shape=(nB,),dtype=global_default_dtype) + normal_noise[:,4] ))
    init_state =State.update(init_state,'varphi_old_x', 
                tf.minimum(tf.zeros(nB,dtype=global_default_dtype),tf.constant(varphi_old,shape=(nB,),dtype=global_default_dtype) + normal_noise[:,5]))
    
    regime_rand = tf.cast(tf.random.categorical(tf.math.log([[0.5,0.5]]),nB),dtype=global_default_dtype)
    
    if Parameters.ckpt.current_episode < 2:    
        nB = Parameters.starting_state.shape[0]
        normal_noise = tf.random.normal(
            [nB,3],
            mean=0.0,
            stddev=tf.constant([[sigma_a_x**2/(1-rho_a_x**2),sigma_xi_x**2/(1-rho_xi_x**2),sigma_g_x**2/(1-rho_g_x**2)]],dtype=global_default_dtype), # a, xi,  g
            dtype=global_default_dtype)
        init_state =State.update(init_state,'log_a_x', 
                    tf.constant(0.,shape=(nB,),dtype=global_default_dtype) + normal_noise[:,0])
        init_state =State.update(init_state,'log_xi_x', 
                    tf.constant(0.,shape=(nB,),dtype=global_default_dtype) + normal_noise[:,1])
        init_state =State.update(init_state,'log_g_x', 
                    tf.constant(0.,shape=(nB,),dtype=global_default_dtype) + normal_noise[:,2])

    
        Parameters.starting_state.assign(init_state)

        Parameters.starting_state.assign(State.update(Parameters.starting_state,'regime_x', 
                                                    regime_rand[0,:]))


