# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

################################################
# constants from config file

config = {
    "N_sim_batch": 1024,
    "N_epochs_per_episode": 1,
    "N_minibatch_size": 128,
    "N_episode_length": 10,
    "N_episodes": 10000000,
    "n_quad_pts": 3,
    # "expectation_pseudo_draws": 5,
    "expectation_type": "gauss-hermite",
    "sorted_within_batch": False,
    "error_filename": "error_log.txt",
    "keras_precision": "float32",
}

constants = {'beta': 0.9975,
            'gamma': 2,    
            'epsilon': 7,
            'omega': 1.,
            'xi': 1.233,
            'theta': 1 - 0.25,
            'eps_CC': 0.,
            #exogenous processes
            'p_12': 1/28,
            'p_21': 1/24,
            'sigma_a_x': 0.009,
            'sigma_xi_x': 0.0014,
            'sigma_g_x': 0.0052,
            'rho_a_x': 0.99,
            'rho_xi_x': 0.9,
            'rho_g_x': 0.97,
            'g_bar': 0.2,
            'sim_mult': 1.0, # makes sim noise sim_mult times larger
            }  
    

#################################################


epsilon = constants['epsilon']
low_pi = -0.1
high_pi = 0.1
low_pi_aux = (1 + low_pi)**epsilon
high_pi_aux = (1 + high_pi)**epsilon
high_p_star = 1.1
low_p_star = 0.9
low_p_star_aux = high_p_star**(-epsilon)
high_p_star_aux = low_p_star**(-epsilon)

constants['M_cal'] = constants['epsilon']/(constants['epsilon'] - 1)
constants['tau_bar'] = -1/constants['epsilon']

### states 

# total endogenous state space
end_state = []
end_state.append({'name':'disp_old_x'})

# exogenous states
ex_state = []
ex_state.append({'name':'log_a_x'})
ex_state.append({'name':'log_xi_x'})
ex_state.append({'name':'log_g_x'})
ex_state.append({'name':'regime_x'})

# aux states
aux_state = []
aux_state.append({'name':'vartheta_old_x'})
aux_state.append({'name':'rho_old_x'})
aux_state.append({'name':'c_old_x'})
aux_state.append({'name':'i_nom_old_x'})
aux_state.append({'name':'varphi_old_x'})


###############################

### total state space
states_main = end_state + ex_state + aux_state

states = states_main
n_states = len(states)
print("number of states", n_states)


#################################################
### Policies
policies_main = []

policies_main.append({'name':'cons_y'})
policies_main.append({'name':'num_y','activation': 'tf.nn.softplus'})
policies_main.append({'name':'den_y','activation': 'tf.nn.softplus'})
policies_main.append({'name':'i_nom_y','activation': 'tf.nn.softplus'})
policies_main.append({'name':'p_star_aux_y'})

policies_main.append({'name':'eta_y'})
policies_main.append({'name':'varphi_y','activation': 'tf.nn.softplus'})

policies = policies_main

### total number of policies
print("number of policies", len(policies))


##################################################
### definitions

definitions = []
definitions.append({'name':'i_nom_y', 'bounds': {'upper': 0.1, 'penalty_upper': 1.}})
definitions.append({'name':'pi_tot_y'})
definitions.append({'name':'cons_y', 'bounds': {'upper': 1.4, 'penalty_upper': 1., 'lower': 0.6, 'penalty_lower': 1}})
definitions.append({'name':'cons_flex_y'})
definitions.append({'name':'i_flex_y'})
definitions.append({'name':'out_gap_y'})
definitions.append({'name':'regime_x'})

definitions.append({'name':'disp_y', 'bounds': {'upper': 1.4, 'penalty_upper': 1., 'lower': 0.6, 'penalty_lower': 1}})
definitions.append({'name':'p_star_y', 'bounds': {'upper': high_p_star, 'penalty_upper': 1., 'lower': low_p_star, 'penalty_lower': 1}})
definitions.append({'name':'r_real_y'})


definitions.append({'name':'h_work_y'})
definitions.append({'name':'num_y', 'bounds': {'upper': 7.0, 'penalty_upper': 1.}})
definitions.append({'name':'den_y', 'bounds': {'upper': 7.0, 'penalty_upper': 1.}})
definitions.append({'name':'y_tot_y'})
definitions.append({'name':'wage_y'})
definitions.append({'name':'lambda_y'})

definitions.append({'name':'p_star_aux_y', 'bounds': {'lower': low_p_star_aux, 'penalty_lower': 1,'upper': high_p_star_aux, 'penalty_upper': 1.}})
definitions.append({'name':'pi_aux_y', 'bounds': {'lower': low_pi_aux, 'penalty_lower': 1,'upper': high_pi_aux, 'penalty_upper': 1.}})
definitions.append({'name':'zeta_y'})
definitions.append({'name':'mu_y'})
definitions.append({'name':'rho_y', 'bounds': {'lower': -1, 'penalty_lower': 1,'upper': 1, 'penalty_upper': 1.}})
definitions.append({'name':'vartheta_y', 'bounds': {'lower': -1, 'penalty_lower': 1,'upper': 1, 'penalty_upper': 1.}})
definitions.append({'name':'chi_y', 'bounds': {'upper': 1, 'penalty_upper': 1.,'lower': 0., 'penalty_lower': 1}})
definitions.append({'name':'varphi_y', 'bounds': {'lower': -2, 'penalty_lower': 1}})

definitions.append({'name':'tau_x'})
definitions.append({'name':'tau_bar_x'})
definitions.append({'name':'a_x'})
definitions.append({'name':'xi_x'})
definitions.append({'name':'g_x'})

definitions.append({'name':'u_y'})

### total number of definitions
print("number of definitions", len(definitions)) 