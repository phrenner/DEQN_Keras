import tensorflow as tf
import numpy as np
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
import pandas as pd
import Parameters
import matplotlib.pyplot as plt
import State
import PolicyState
import Definitions
import os
# from Graphs import run_episode
from Parameters import MODEL_NAME, global_default_dtype
import sys
import shutil
import scipy

plot_enabled = False

folder_name = f'results_{MODEL_NAME}/simulation'
try:
    os.makedirs(folder_name)
except FileExistsError:
    print("Folder %s already exists, delete it and recreate it" % folder_name)
    shutil.rmtree(folder_name)
    os.makedirs(folder_name)

Hooks = importlib.import_module(Parameters.MODEL_NAME + ".Hooks")
Dynamics = importlib.import_module(Parameters.MODEL_NAME + ".Dynamics")

@tf.function
def do_random_step(current_state):
    return Dynamics.total_step_random(current_state, Parameters.policy(current_state))

def run_episode(state_episode):
    """ Runs an episode starting from the begging of the state_episode. Results are saved into state_episode."""
    current_state = state_episode[0,:,:]
    
    # run optimization
    for i in range(1, state_episode.shape[0]):
        current_state = do_random_step(current_state)
        current_state = State.update(
            current_state, 
            "regime_x",
            tf.concat([
                State.regime_x(current_state)[:1], 
                tf.expand_dims(tf.constant(0.0,dtype=global_default_dtype), axis=0), 
                tf.expand_dims(tf.constant(1.0,dtype=global_default_dtype), axis=0),
                State.regime_x(current_state)[3:4],
                tf.expand_dims(tf.constant(1.0,dtype=global_default_dtype), axis=0),
                ], axis=0))
        current_state = State.update(
            current_state, 
            "log_a_x",
            tf.concat([
                State.log_a_x(current_state)[:3], 
                tf.expand_dims(tf.constant(0.0,dtype=global_default_dtype), axis=0),
                tf.expand_dims(tf.constant(0.0,dtype=global_default_dtype), axis=0),
                ], axis=0))        
        current_state = State.update(
            current_state, 
            "log_xi_x",
            tf.concat([
                State.log_xi_x(current_state)[:3], 
                tf.expand_dims(tf.constant(0.0,dtype=global_default_dtype), axis=0),
                State.log_xi_x(current_state)[3:4], 
                ], axis=0))        
        current_state = State.update(
            current_state, 
            "log_g_x",
            tf.concat([
                State.log_g_x(current_state)[:3], 
                tf.expand_dims(tf.constant(0.0,dtype=global_default_dtype), axis=0),
                tf.expand_dims(tf.constant(0.0,dtype=global_default_dtype), axis=0)
                ], axis=0))        


        # replace above for deterministic results
        # current_state = BatchState.total_step_spec_shock(current_state, Parameters.policy(current_state),0)
        state_episode = tf.tensor_scatter_nd_update(state_episode, tf.constant([[ i ]]), tf.expand_dims(current_state, axis=0))
    
    return state_episode

tf.get_logger().setLevel('CRITICAL')

pd.set_option('display.max_columns', None)
starting_policy = Parameters.policy(Parameters.starting_state)

Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")

        
Parameters.initialize_each_episode = False        
if not Parameters.initialize_each_episode:
    ## simulate from a single starting state which is the mean across the N_sim_batch individual trajectory starting states
    simulation_starting_state = Parameters.starting_state[0:1,:]
    ## simulate a long range and calculate variable bounds + means from it for plotting
    print("Running a long simulation path")
    N_simulated_episode_length = 10000 #Parameters.N_simulated_episode_length or 10000
    N_simulated_batch_size = 5# Parameters.N_simulated_batch_size or 1
        
state_episode = tf.tile(tf.expand_dims(simulation_starting_state, axis = 0), [N_simulated_episode_length, N_simulated_batch_size, 1])

print("Running episode to get range of variables...")

state_episode = run_episode(state_episode)
state_episode_sim = tf.reshape(state_episode[:,0,:], [N_simulated_episode_length,len(Parameters.states)])
state_episode_NT = tf.reshape(state_episode[:,1,:], [N_simulated_episode_length,len(Parameters.states)])
state_episode_SS = tf.reshape(state_episode[:,2,:], [N_simulated_episode_length,len(Parameters.states)])
state_episode_ss_only = tf.reshape(state_episode[:,3,:], [N_simulated_episode_length,len(Parameters.states)])
state_episode_xi_only = tf.reshape(state_episode[:,4,:], [N_simulated_episode_length,len(Parameters.states)])

print("Finished plots. Calculating Euler discrepancies...")

## calculate euler deviations
policy_episode_sim = Parameters.policy(state_episode_sim)
euler_discrepancies = pd.DataFrame(Equations.equations(state_episode_sim, policy_episode_sim))
# print("Euler discrepancy (absolute value) metrics")
# print(euler_discrepancies.abs().describe(include='all'))

# save all relevant quantities along the trajectory 
# euler_discrepancies.to_csv(folder_name + "/simulated_euler_discrepancies.csv", index=False)

state_episode_df = pd.DataFrame({s:(getattr(State,s)(state_episode_sim)).numpy() for s in Parameters.states})
# state_episode_df.to_csv(folder_name + "/simulated_states.csv", index=False)

policy_episode_df = pd.DataFrame({ps:getattr(PolicyState,ps)(policy_episode_sim) for ps in Parameters.policy_states})
# policy_episode_df.to_csv(folder_name + "/simulated_policies.csv", index=False)

out_def_lst = ["cons_y","disp_y","i_nom_y","h_work_y","pi_tot_y","pi_aux_y","p_star_y","p_star_aux_y","r_real_y","tau_x","a_x","xi_x", "g_x", "y_tot_y", "out_gap_y", "i_flex_y", "cons_flex_y", "regime_x"]

defs_dict = {d:(getattr(Definitions,d)(state_episode_sim, policy_episode_sim)).numpy() for d in out_def_lst}
definition_episode_df = pd.DataFrame(defs_dict)
# definition_episode_df.to_csv(folder_name + "/simulated_definitions.csv", index=False)
np.savez_compressed(folder_name + "/simulated_definitions.npz", **defs_dict)

f = open(f"{folder_name}/simulation_stats.txt", "w")

f.write("Euler discrepancy (absolute value) metrics\n")
out_0 = euler_discrepancies.abs().describe(include='all')
f.write(f'{out_0}')

f.write("\nState metrics\n")
out_1 = state_episode_df.describe(include='all')
f.write(f'{out_1}')

f.write("\nPolicy metrics\n")
out_2 = policy_episode_df.describe(include='all')
f.write(f'{out_2}')

f.write("\nDefinition metrics\n")
out_3 = definition_episode_df.describe(include='all')
f.write(f'{out_3}')

f.close()

print("Euler discrepancy (absolute value) metrics")
print(euler_discrepancies.abs().describe(include='all'))
print("State metrics")
print(state_episode_df.describe(include='all'))
print("Policy metrics")
print(policy_episode_df.describe(include='all'))
print("Definition metrics")
print(definition_episode_df.describe(include='all'))

policy_episode_NT = Parameters.policy(state_episode_NT)
defs_dict = {d:(getattr(Definitions,d)(state_episode_NT, policy_episode_NT)).numpy() for d in out_def_lst}
definition_episode_df = pd.DataFrame(defs_dict)
np.savez_compressed(folder_name + "/simulated_definitions_NT.npz", **defs_dict)
print("Definition metrics")
print(definition_episode_df.describe(include='all'))

policy_episode_SS = Parameters.policy(state_episode_SS)
defs_dict = {d:(getattr(Definitions,d)(state_episode_SS, policy_episode_SS)).numpy() for d in out_def_lst}
definition_episode_df = pd.DataFrame(defs_dict)
np.savez_compressed(folder_name + "/simulated_definitions_SS.npz", **defs_dict)
print("Definition metrics")
print(definition_episode_df.describe(include='all'))


policy_episode_ss_only = Parameters.policy(state_episode_ss_only)
defs_dict = {d:(getattr(Definitions,d)(state_episode_ss_only, policy_episode_ss_only)).numpy() for d in out_def_lst}
definition_episode_df = pd.DataFrame(defs_dict)
np.savez_compressed(folder_name + "/simulated_definitions_ss_only.npz", **defs_dict)
print("Definition metrics")
print(definition_episode_df.describe(include='all'))

policy_episode_xi_only = Parameters.policy(state_episode_xi_only)
defs_dict = {d:(getattr(Definitions,d)(state_episode_xi_only, policy_episode_xi_only)).numpy() for d in out_def_lst}
definition_episode_df = pd.DataFrame(defs_dict)
np.savez_compressed(folder_name + "/simulated_definitions_xi_only.npz", **defs_dict)
print("Definition metrics")
print(definition_episode_df.describe(include='all'))

del sys.modules['Parameters']
