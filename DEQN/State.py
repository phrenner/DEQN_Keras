# TF Module containing state variables

import importlib
import math
import sys
import tensorflow as tf
import numpy as np
from Parameters import expectation_pseudo_draws, expectation_type, MODEL_NAME, policy, states, state_bounds_hard, global_default_dtype

Dynamics = importlib.import_module(MODEL_NAME + ".Dynamics")

for i, state in enumerate(states):
    if (state in state_bounds_hard["lower"]) or (state in state_bounds_hard["upper"]):
        setattr(
            sys.modules[__name__],
            state,
            (
                lambda ind: lambda x: tf.clip_by_value(x[..., ind], state_bounds_hard["lower"].get(states[ind], -np.inf), state_bounds_hard["upper"].get(states[ind], np.inf))
            )(i),
        )
    else:
        setattr(sys.modules[__name__], state, (lambda ind: lambda x: x[..., ind])(i))

    # always add a 'raw' attribute as well - this can be used for penalties
    setattr(sys.modules[__name__], state + "_RAW", (lambda ind: lambda x: x[..., ind])(i))

def E_t_gen(state, policy_state, mode=None):
    if expectation_type == 'pseudo_random':
        next_states = [Dynamics.total_step_random(state, policy_state, mode) for i in range(expectation_pseudo_draws)]
        next_policies = [policy(next_state) for next_state in next_states]
        
        def E_t(evalFun):
            # calculate conditional expectation
            res = tf.zeros(state.shape[0])  
            for i in range(expectation_pseudo_draws):
                res += 1.0 / expectation_pseudo_draws * evalFun(next_states[i], next_policies[i])
            return res
    else:
        old_state,old_pol,next_states,next_policies = Dynamics.total_step_spec_shock(state, policy_state, mode)
        prob_vec = Dynamics.total_step_spec_probs(state, policy_state)
        n_int_samples = Dynamics.n_int_samples
        nB = state.shape[0]

        def E_t(evalFun):
            # calculate conditional expectation
            tmp_calc = tf.reshape(prob_vec * evalFun(old_state,old_pol,next_states,next_policies), [nB,n_int_samples])
            res = tf.reduce_sum(tmp_calc,axis=-1)
            return res

    return E_t

def E_t_gen_sw(state, policy_state, state_sw, policy_sw, mode=None):
    if expectation_type == 'pseudo_random':
        next_states = [Dynamics.total_step_random(state, policy_state, mode) for i in range(expectation_pseudo_draws)]
        next_policies = [policy(next_state) for next_state in next_states]
        
        def E_t(evalFun):
            # calculate conditional expectation
            res = tf.zeros(state.shape[0])  
            for i in range(expectation_pseudo_draws):
                res += 1.0 / expectation_pseudo_draws * evalFun(next_states[i], next_policies[i])
            return res
    else:
        old_state,old_pol,next_states,next_policies = Dynamics.total_step_spec_shock(state, policy_state, state_sw, policy_sw, mode)
        prob_vec = Dynamics.total_step_spec_probs(state, policy_state)
        n_int_samples = Dynamics.n_int_samples
        nB = state.shape[0]

        def E_t(evalFun):
            # calculate conditional expectation
            tmp_calc = tf.reshape(prob_vec * evalFun(old_state,old_pol,next_states,next_policies), [nB,n_int_samples])
            res = tf.reduce_sum(tmp_calc,axis=-1)
            return res

    return E_t

def update(old_states, at, new_vals):
    i = states.index(at)
    return tf.tensor_scatter_nd_update(old_states,[[j,i] for j in range(old_states.shape[0])], new_vals)

def update_dict(old_states, new_vals_dict):
    new_states = old_states
    for s in new_vals_dict:
        new_states = tf.tensor_scatter_nd_update(new_states,[[j,i] for j in range(new_states.shape[0])], new_vals_dict[s])
        
    return new_states