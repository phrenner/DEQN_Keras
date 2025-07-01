# calculate equilibrium conditions, based on state values & estimated policy function

import importlib
import tensorflow as tf
import PolicyState 
import Definitions
from Parameters import definition_bounds_hard, MODEL_NAME, policy_bounds_hard, global_default_dtype, policy_bounds_hard, definition_bounds_hard, ENABLE_TB

Equations = importlib.import_module(MODEL_NAME + ".Equations")

def penalty_bounds_policy(state, policy_state, enable_tb=False):
    nB = state.shape[0]
    dist_to_bounds = tf.constant(0.001,dtype=global_default_dtype)
    penalty_const = tf.constant(1000.00,dtype=global_default_dtype)
    res = tf.constant(0.0,dtype=global_default_dtype)
    counter = tf.constant(0.0,dtype=global_default_dtype) # number of bounds to scale overall loss
    for bound_vars in policy_bounds_hard['lower'].keys():
        counter += 1.0
        # bounded is >= raw when lower bound is violated - we measure how strong this bound is
        if policy_bounds_hard['penalty_lower'][bound_vars] is None:  # if no bound is specified use a barrier function; barrier is negative if violated and positive otherwise
            raw_vs_bounded = - tf.math.minimum(tf.zeros(nB,dtype=global_default_dtype), getattr(PolicyState, bound_vars + "_RAW")(policy_state) - (policy_bounds_hard["lower"][bound_vars] + dist_to_bounds)) # only consider violations of lower bound
            penalty = tf.math.reduce_sum(penalty_const * Huber_loss(raw_vs_bounded, dist_to_bounds))
        else:
            raw_vs_bounded = - tf.math.minimum(tf.zeros(nB,dtype=global_default_dtype), getattr(PolicyState, bound_vars + "_RAW")(policy_state) - policy_bounds_hard["lower"][bound_vars]) # only consider violations of lower bound
            penalty = tf.math.reduce_sum(policy_bounds_hard['penalty_lower'][bound_vars] * Huber_loss(raw_vs_bounded,1.0))
        if enable_tb:        
            tf.summary.scalar('penalty_lower_policy_'+bound_vars, penalty)
        res += penalty
    
    for bound_vars in policy_bounds_hard['upper'].keys():
        counter += 1.0
        # bounded is always <= raw - we measure how strong this bound is
        if policy_bounds_hard['penalty_upper'][bound_vars] is None: # if no bound is specified use a barrier function; barrier is negative if violated and positive otherwise
            raw_vs_bounded = tf.math.maximum(tf.zeros(nB,dtype=global_default_dtype), getattr(PolicyState, bound_vars + "_RAW" )(policy_state) - (policy_bounds_hard["upper"][bound_vars] - dist_to_bounds)) # only consider violations of upper bound
            penalty = tf.math.reduce_sum(penalty_const * Huber_loss(raw_vs_bounded, dist_to_bounds))
        else:
            raw_vs_bounded = tf.math.maximum(tf.zeros(nB,dtype=global_default_dtype), getattr(PolicyState, bound_vars + "_RAW" )(policy_state) - policy_bounds_hard["upper"][bound_vars]) # only consider violations of upper bound
            penalty = tf.math.reduce_sum(policy_bounds_hard['penalty_upper'][bound_vars] * Huber_loss(raw_vs_bounded,1.0))
        if enable_tb:
            tf.summary.scalar('penalty_upper_policy_' + bound_vars, penalty)
        res += penalty
    
    for bound_vars in definition_bounds_hard['lower'].keys():
        counter += 1.0
        # bounded is always >= raw - we measure how strong this bound is
        if definition_bounds_hard['penalty_lower'][bound_vars] is None: # if no bound is specified use a barrier function; barrier is negative if violated and positive otherwise
            raw_vs_bounded = - tf.math.minimum(tf.zeros(nB,dtype=global_default_dtype), getattr(Definitions, bound_vars + "_RAW")(state, policy_state) - (definition_bounds_hard["lower"][bound_vars] + dist_to_bounds)) # only consider violations of lower bound
            penalty = tf.math.reduce_sum(penalty_const * Huber_loss(raw_vs_bounded, dist_to_bounds))
        else:
            raw_vs_bounded = - tf.math.minimum(tf.zeros(nB,dtype=global_default_dtype), getattr(Definitions, bound_vars + "_RAW")(state, policy_state) - definition_bounds_hard["lower"][bound_vars]) # only consider violations of lower bound
            penalty = tf.math.reduce_sum(definition_bounds_hard['penalty_lower'][bound_vars] * Huber_loss(raw_vs_bounded,1.0))
        if enable_tb:
            tf.summary.scalar('penalty_lower_def_'+bound_vars, penalty)
        res += penalty
    
    for bound_vars in definition_bounds_hard['upper'].keys():
        counter += 1.0
        # bounded is always <= raw - we measure how strong this bound is
        if definition_bounds_hard['penalty_upper'][bound_vars] is None: # if no bound is specified use a barrier function; barrier is negative if violated and positive otherwise
            raw_vs_bounded = tf.math.maximum(tf.zeros(nB,dtype=global_default_dtype), getattr(Definitions, bound_vars + "_RAW" )(state, policy_state) - (definition_bounds_hard["upper"][bound_vars] - dist_to_bounds)) # only consider violations of upper bound
            penalty = tf.math.reduce_sum(penalty_const * Huber_loss(raw_vs_bounded, dist_to_bounds))
        else:
            raw_vs_bounded = tf.math.maximum(tf.zeros(nB,dtype=global_default_dtype), getattr(Definitions, bound_vars + "_RAW" )(state, policy_state) - definition_bounds_hard["upper"][bound_vars]) # only consider violations of upper bound
            penalty = tf.math.reduce_sum(definition_bounds_hard['penalty_upper'][bound_vars] * Huber_loss(raw_vs_bounded,1.0))
        if enable_tb:
            tf.summary.scalar('penalty_upper_def_' + bound_vars, penalty)
        res += penalty
    
    return res, counter

# Huber loss
Huber_loss_delta = 1.0
def Huber_loss(delta_y, delta):
    abs = tf.math.abs(delta_y)
    return tf.where(
        tf.less_equal(abs,delta),
        0.5*delta_y**2,
        delta*(abs - 0.5*delta)) 


@tf.function
def loss_specific(state, policy_net, equations,optimizer_,indx):
    policy_state = policy_net(state)
    n_batches = state.shape[0]
    loss_val = tf.constant(0.0,dtype=global_default_dtype)         # total loss
    net_loss_val = tf.constant(0.0,dtype=global_default_dtype)     # net loss (without penalty)
    if ENABLE_TB:
        tf.summary.experimental.set_step(optimizer_.iterations)
    losses = equations(state, policy_state)
    
    eq_f_lst = []
    for eq_f in losses.keys():
        eq_loss = tf.math.reduce_sum(Huber_loss(losses[eq_f], Huber_loss_delta)) #+ tf.math.reduce_sum(((tf.math.exp(policy_state)/tf.stop_gradient(tf.math.exp(policy_state)) - 1)))

        eq_f_lst.append(eq_loss)
        
    eq_f_vec = tf.stack(eq_f_lst)
    loss_val = tf.math.reduce_sum(eq_f_vec)/ (len(eq_f_lst) * n_batches)
    # loss_val = tf.math.reduce_sum(tf.nn.softmax(eq_f_vec) * eq_f_vec)
        
    net_loss_val = loss_val
    penalty_bounds, no_bounds = penalty_bounds_policy(state, policy_state, False)
    loss_val += penalty_bounds
    #normalize loss with number of equations
    
    
    return loss_val, net_loss_val


Dynamics = importlib.import_module(MODEL_NAME + ".Dynamics")

n_look_ahead = 1

@tf.function
def loss_with_lookahead_specific(state, policy_net, equations,optimizer_,indx):
    n_batches = state.shape[0]
    loss_val = tf.constant(0.0,dtype=global_default_dtype)         # total loss
    net_loss_val = tf.constant(0.0,dtype=global_default_dtype)     # net loss (without penalty)
    if ENABLE_TB:
        tf.summary.experimental.set_step(optimizer_.iterations)

    current_state = state
    for indx_horizon in range(n_look_ahead):

        policy_state = policy_net(current_state)
        losses = equations(current_state, policy_state)
        
        eq_f_lst = []
        for eq_f in losses.keys():
            eq_loss = tf.math.reduce_sum(Huber_loss(losses[eq_f], Huber_loss_delta)) #+ tf.math.reduce_sum(((tf.math.exp(policy_state)/tf.stop_gradient(tf.math.exp(policy_state)) - 1)))
                
            if ENABLE_TB:
                tf.summary.scalar('dev_' + eq_f, eq_loss)

            eq_f_lst.append(eq_loss)
            
        eq_f_vec = tf.stack(eq_f_lst)

        loss_val += tf.math.reduce_sum(eq_f_vec)/ (len(eq_f_lst) * n_batches * n_look_ahead)
        # loss_val = tf.math.reduce_sum(tf.nn.softmax(eq_f_vec) * eq_f_vec)
        
        net_loss_val += loss_val
        penalty_bounds, no_bounds = penalty_bounds_policy(current_state, policy_state)
        loss_val += penalty_bounds
        current_state = Dynamics.total_step_random(current_state, policy_state)
    
    if ENABLE_TB:
        tf.summary.scalar(f'dev_loss_{indx}', loss_val)
        tf.summary.scalar(f'dev_net_loss_{indx}', net_loss_val)    
    
    return loss_val, net_loss_val

