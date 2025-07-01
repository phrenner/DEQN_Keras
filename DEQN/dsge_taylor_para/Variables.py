# -*- coding: utf-8 -*-
import tensorflow as tf

################################################
# constants from config file

config = {
    "N_sim_batch": 512,
    "N_epochs_per_episode": 1,
    "N_minibatch_size": 128,
    "N_episode_length": 10,
    "N_episodes": 10000000,
    "n_quad_pts": 3,
    # "expectation_pseudo_draws": 5,
    "expectation_type": "gauss-hermite",
    "sorted_within_batch": False,
    "error_filename": "error_log.txt",
    "keras_precision": "float64",
}

constants = {'beta': 0.9975,
            'g_bar': 0.2,
            'gamma': 2,    
            'epsilon': 7,
            'omega': 1.,
            'xi': 1.233,
            'psi': 2.0,
            'theta': 1 - 0.25,
            'pi_bar': 0.0 / 4.0,
            #exogenous processes
            'p_12': 1/48,
            'p_21': 1/24,
            'p_21_l': 1/60,
            'p_21_u': 1,
            'sigma_a_x': 0.009,
            'sigma_xi_x': 0.0014,
            'sigma_g_x': 0.0052,
            'rho_i_x': 0.,
            'rho_a_x': 0.99,
            'rho_xi_x': 0.9,
            'rho_g_x': 0.97,
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

lower_cons = 0.6
upper_cons = 1.4

constants['M_cal'] = constants['epsilon']/(constants['epsilon'] - 1)
constants['tau_bar_low'] = -1/constants['epsilon']
constants['tau_bar_high'] = -1/constants['epsilon']

theta = constants['theta']
upper_price_adj_prob = 1.0 - theta + 0.2
lower_price_adj_prob = 0.232

### states 

# total endogenous state space
end_state = []
end_state.append({'name':'disp_old_x'})
end_state.append({'name':'i_old_x'})

# exogenous states
ex_state = []
ex_state.append({'name':'log_a_x'})
ex_state.append({'name':'log_xi_x'})
ex_state.append({'name':'log_g_x'})
ex_state.append({'name':'regime_x'})

# parameter
para_state = []
para_state.append({'name':'p_21_x'})


###############################

### total state space
states_main = end_state + ex_state + para_state
n_states = len(states_main)
print("number of states", n_states)

states = states_main


#################################################
### Policies
policies_main = []
policies_main.append({'name':'num_y'})
policies_main.append({'name':'den_y'})
policies_main.append({'name':'disp_y'})
policies_main.append({'name':'p_star_aux_y'})
policies_main.append({'name':'cons_y'})
policies_main.append({'name':'i_nom_y'})


policies = policies_main

### total number of policies
print("number of policies", len(policies))


##################################################
### definitions

definitions = []
definitions.append({'name':'i_nom_y'})
definitions.append({'name':'pi_tot_y'})
definitions.append({'name':'cons_y', 'bounds': {'upper': upper_cons, 'penalty_upper': 1, 'lower': lower_cons, 'penalty_lower': 1.}})
definitions.append({'name':'log_cons_y'})
definitions.append({'name':'cons_flex_y'})
definitions.append({'name':'i_flex_y'})

definitions.append({'name':'disp_y', 'bounds': {'upper': 1.1, 'penalty_upper': 1, 'lower': 0.9, 'penalty_lower': 1.}})

definitions.append({'name':'p_star_y', 'bounds': {'upper': high_p_star, 'penalty_upper': 1, 'lower': low_p_star, 'penalty_lower': 1.}})
definitions.append({'name':'r_real_y'})

definitions.append({'name':'h_work_y'})
definitions.append({'name':'y_tot_y'})
definitions.append({'name':'wage_y'})
definitions.append({'name':'tau_x'})


definitions.append({'name':'num_y', 'bounds': {'lower': 1., 'penalty_lower': 1., 'upper': 8.0, 'penalty_upper': 1.}})
definitions.append({'name':'den_y', 'bounds': {'lower': 1., 'penalty_lower': 1., 'upper': 8.0, 'penalty_upper': 1.}})
definitions.append({'name':'lambda_y'})

definitions.append({'name':'pi_aux_y', 'bounds': {'lower': low_pi_aux, 'penalty_lower': 1.,'upper': high_pi_aux, 'penalty_upper': 1.}})
definitions.append({'name':'p_star_aux_y'})

definitions.append({'name':'tau_x'})
definitions.append({'name':'a_x'})
definitions.append({'name':'xi_x'})
definitions.append({'name':'g_x'})
definitions.append({'name':'tau_bar_x'})

definitions.append({'name':'out_gap_y'})
definitions.append({'name':'regime_x'})

definitions.append({'name':'p_21_x'})

definitions.append({'name':'u_y'})

### total number of definitions
print("number of definitions", len(definitions)) 
