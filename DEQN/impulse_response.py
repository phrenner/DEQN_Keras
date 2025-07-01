import tensorflow as tf
"""
Disabling GPU for post processing
"""
try:
    # Disable all GPUS
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'

    print("Disabled GPU")
except:
    # Invalid device or cannot modify virtual devices once initialized.
    visible_devices = tf.config.get_visible_devices()
    print(f"Disableing GPU failed; visible devices are {visible_devices}")
    pass

import importlib
import Parameters
import pandas as pd
import State
import PolicyState
import Definitions
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os
import sys
from Parameters import global_default_dtype

plot_enabled = False

Dynamics = importlib.import_module(Parameters.MODEL_NAME + ".Dynamics")

@tf.function
def do_step_no_shock(current_state):
    return Dynamics.total_step_no_shock(current_state, Parameters.policy(current_state))

@tf.function
def do_step_shock(current_state):
    return Dynamics.total_step_shock(current_state, Parameters.policy(current_state))


def run_episode(state_episode,shock_indx):
    """ Runs an episode starting from the begging of the state_episode. Results are saved into state_episode."""
    current_state = state_episode[0,:,:]
    n_batches = state_episode.shape[1]
    regimes = tf.concat([
        tf.zeros((n_batches - 2)//2,dtype=global_default_dtype),
        tf.ones((n_batches - 2)//2,dtype=global_default_dtype),
        tf.constant([0.],dtype=global_default_dtype),
        tf.constant([1.],dtype=global_default_dtype)],0)        
    current_state = State.update(current_state,"regime_x",regimes)
    
    # run optimization
    for i in range(1, state_episode.shape[0]):
        if i == shock_indx:
            current_state = do_step_shock(current_state)
        else:
            current_state = do_step_no_shock(current_state)
        # replace above for deterministic results
        # current_state = BatchState.total_step_spec_shock(current_state, Parameters.policy(current_state),0)
        state_episode = tf.tensor_scatter_nd_update(state_episode, tf.constant([[ i ]]), tf.expand_dims(current_state, axis=0))
    
    return state_episode


folder_name = f'results_{Parameters.MODEL_NAME}/IRS'
try:
    os.makedirs(folder_name)
except FileExistsError:
    print("Folder %s already exists, delete it and recreate it" % folder_name)
    shutil.rmtree(folder_name)
    os.makedirs(folder_name)

## Hooks allows to load in a starting state for the simulation (if a particular choice is needed)
Hooks = importlib.import_module(Parameters.MODEL_NAME + ".Hooks")
starting_policy = Parameters.policy(Parameters.starting_state)

## Import EE to potentially compute 
Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")

## simulate from a single starting state which is the mean across the N_sim_batch individual trajectory starting states
simulation_starting_state = tf.math.reduce_mean(Parameters.starting_state, axis = 0, keepdims=True)

no_steps = 1000 #burn-in-steps
no_steps_decay = 400 #iteration steps after the pulse

print("Model with regime switching")
n_batches = 2*12 + 2

state_episode = tf.tile(tf.expand_dims(simulation_starting_state,0),[no_steps+no_steps_decay,n_batches,1])

presteps = 5 #burn in steps to keep

state_episode = run_episode(state_episode,no_steps)[no_steps-presteps:,:,:]

flat_state_episode = tf.reshape(state_episode,[(no_steps_decay + presteps)*n_batches,state_episode.shape[-1]])
flat_policy_episode = Parameters.policy(flat_state_episode)
        
policy_episode = tf.reshape(
    flat_policy_episode,
        [no_steps_decay + presteps,n_batches,len(Parameters.policy_states)])


def_lst = ["cons_y","disp_y","i_nom_y","h_work_y","pi_tot_y","pi_aux_y","p_star_y","p_star_aux_y","r_real_y","tau_x","a_x","xi_x", "g_x", "y_tot_y", "out_gap_y", "i_flex_y", "cons_flex_y", "regime_x"]


shock_lst = [["3sigA_NT","1sigA_NT","-1sigA_NT","-3sigA_NT"],
                ["3sigT_NT","1sigT_NT","-1sigT_NT","-3sigT_NT"],
                ["3sigG_NT","1sigG_NT","-1sigG_NT","-3sigG_NT"],                 
                ["3sigA_SS","1sigA_SS","-1sigA_SS","-3sigA_SS"],
                ["3sigT_SS","1sigT_SS","-1sigT_SS","-3sigT_SS"],
                ["3sigG_SS","1sigG_SS","-1sigG_SS","-3sigG_SS"],
                ["NT","SS"]                 
            ]


def flatten_extend(matrix):
    flat_list = []
    for row in matrix:
        flat_list.extend(row)
    return flat_list

flat_shock_lst = flatten_extend(shock_lst)
out_dict = {}
out_dict_states = {}
for indxb in range(n_batches):
    ##==========================================
    ## output to pandas:
    # state_episode_df = pd.DataFrame({s:getattr(State,s)(state_episode[:,indxb,:]) for s in Parameters.states})
    # state_episode_df.to_csv(folder_name + f"/simulated_states_{flat_shock_lst[indxb]}.csv", index=False)
    out_dict_states = out_dict_states | {flat_shock_lst[indxb]: {s:getattr(State,s)(state_episode[:,indxb,:]) for s in Parameters.states}}
    # policy_episode_df = pd.DataFrame({ps:getattr(PolicyState,ps)(policy_episode[:,indxb,:]) for ps in Parameters.policy_states})
    # policy_episode_df.to_csv(folder_name + f"/simulated_policies_{flat_shock_lst[indxb]}.csv", index=False)

    out_dict = out_dict | {flat_shock_lst[indxb]: {d:getattr(Definitions,d)(state_episode[:,indxb,:], policy_episode[:,indxb,:]) for d in def_lst}}
    # definition_episode_df = pd.DataFrame({d:getattr(Definitions,d)(state_episode[:,indxb,:], policy_episode[:,indxb,:]) for d in def_lst})
    # definition_episode_df.to_csv(folder_name + f"/simulated_definitions_{flat_shock_lst[indxb]}.csv", index=False)

    # EE_episode_df = pd.DataFrame(Equations.equations(state_episode[:,indxb,:], policy_episode[:,indxb,:]))
    # EE_episode_df.to_csv(folder_name + f"/simulated_euler_error_{flat_shock_lst[indxb]}.csv", index=False)

np.savez_compressed(folder_name + "/IR_definitions.npz", **out_dict)
np.savez_compressed(folder_name + "/IR_states.npz", **out_dict_states)

