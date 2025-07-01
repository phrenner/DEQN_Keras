import tensorflow as tf
import Definitions
import State
import Parameters
from Parameters import sigma_a_x, sigma_xi_x, rho_a_x, rho_xi_x, sigma_xi_x, p_12, p_21, rng, n_quad_pts, sim_mult, global_default_dtype, policy, sigma_g_x, rho_g_x, beta
from scipy.stats import norm
import numpy as np

# shocks ax
x_tmp,w_tmp = np.polynomial.hermite.hermgauss(n_quad_pts)
probs_gauss_hermite = w_tmp/np.sqrt(np.pi)/np.sum(w_tmp/np.sqrt(np.pi))
shocks_gauss_hermite =  np.sqrt(2)*x_tmp

#######################################################################
# Naming convention
# GM: great moderation half of mean sigma value,  regime = 0
# SS: supply shock double mean sigma value,       regime = 1
tmp_trans_prob_regime_NT = np.array([1-p_12,p_12])
tmp_trans_prob_regime_SS = np.array([p_21,1 - p_21])

trans_prob_regime_NT = tf.constant(tmp_trans_prob_regime_NT,dtype=global_default_dtype)
trans_prob_regime_SS = tf.constant(tmp_trans_prob_regime_SS,dtype=global_default_dtype)

tmp_shock_val = np.zeros([n_quad_pts**3 * 2,4])
tmp_shock_prob_NT = np.ones([n_quad_pts**3 * 2])
tmp_shock_prob_SS = np.ones([n_quad_pts**3 * 2])

counter = 0
for indxm in range(2):
    for indx_ax in range(n_quad_pts):
        for indx_xi in range(n_quad_pts):
            for indx_g in range(n_quad_pts):
                tmp_shock_prob_NT[counter] = probs_gauss_hermite[indx_ax] * probs_gauss_hermite[indx_xi] * probs_gauss_hermite[indx_g] * tmp_trans_prob_regime_NT[indxm]
                tmp_shock_prob_SS[counter] = probs_gauss_hermite[indx_ax] * probs_gauss_hermite[indx_xi] * probs_gauss_hermite[indx_g] * tmp_trans_prob_regime_SS[indxm]
                tmp_shock_val[counter,0] = shocks_gauss_hermite[indx_ax] * sigma_a_x
                tmp_shock_val[counter,1] = shocks_gauss_hermite[indx_xi] * sigma_xi_x
                tmp_shock_val[counter,2] = shocks_gauss_hermite[indx_g] *  sigma_g_x
                tmp_shock_val[counter,3] = 1.0 * indxm
                counter += 1

n_int_samples = counter

# combined shocks
shock_values = tf.constant(tmp_shock_val,dtype=global_default_dtype)

shock_prob_NT = tf.constant(tmp_shock_prob_NT,dtype=global_default_dtype)
shock_prob_SS = tf.constant(tmp_shock_prob_SS,dtype=global_default_dtype)

shock_prob_NT  = shock_prob_NT/tf.math.reduce_sum(shock_prob_NT,-1)
shock_prob_SS  = shock_prob_SS/tf.math.reduce_sum(shock_prob_SS,-1)

# shock_probs = tf.ones(n_int_samples,dtype=global_default_dtype)  #dummy initialization

if Parameters.expectation_type == 'monomial':
   raise Exception("not implemented")

def total_step_random(prev_state, policy_state):
    ar = AR_step(prev_state)
    
    shock = shock_step_random(prev_state)

    policy = policy_step(prev_state, policy_state)
    
    total = augment_state(ar + shock + policy)

    return total


# same as above, but non randomized shock, but rather the same shock for each realization
def total_step_spec_shock(prev_state, policy_state, mode=None):
    batch_size = prev_state.shape[0]
    old_state = tf.repeat(prev_state,n_int_samples,axis=0)
    old_pol = tf.repeat(policy_state,n_int_samples,axis=0)

    ar = AR_step(old_state)
    shock = shock_step_spec_shock(old_state,batch_size)
    policy_comp = policy_step(old_state, old_pol)
    
    next_states = augment_state(ar + shock + policy_comp)
    next_policies = policy(next_states)
    return old_state,old_pol,next_states,next_policies

def total_step_spec_probs(prev_state, policy_state, mode=None):
    regime_in = tf.repeat(State.regime_x(prev_state),n_int_samples,axis=0)
    shock_p_vec_1 = tf.tile(shock_prob_NT,[prev_state.shape[0]])
    shock_p_vec_2 = tf.tile(shock_prob_SS,[prev_state.shape[0]])
    return (1 - regime_in) * shock_p_vec_1 + regime_in * shock_p_vec_2

def shock_step_spec_shock(prev_state,batch_size):
    # Use a specific shock - for calculating expectations
    shock_step = tf.zeros_like(prev_state)
    shock_vec = tf.tile(shock_values,[batch_size,1])
    shock_step = State.update(shock_step,"log_a_x",  (shock_vec[:,0]))
    shock_step = State.update(shock_step,"log_xi_x", (shock_vec[:,1]))
    shock_step = State.update(shock_step,"log_g_x",  (shock_vec[:,2]))
    shock_step = State.update(shock_step,"regime_x", (shock_vec[:,3]))

    return shock_step


def augment_state(state):
    state = State.update(state, "log_a_x", (State.log_a_x(state)))
    state = State.update(state, "log_xi_x", (State.log_xi_x(state)))    
    state = State.update(state, "log_g_x", (State.log_g_x(state)))    

    return state

def AR_step(prev_state):
    # only rf, yt and kappa have autoregressive components
    ar_step = tf.zeros_like(prev_state)
    sigma_a = sigma_a_x
    ar_step = State.update(ar_step, "log_a_x", - 0.5 * (1 - rho_a_x) * sigma_a**2 + rho_a_x * (State.log_a_x(prev_state)) )
    sigma_xi = sigma_xi_x
    ar_step = State.update(ar_step, "log_xi_x", - 0.5 * (1 - rho_xi_x) * sigma_xi**2 + rho_xi_x * (State.log_xi_x(prev_state)) )
    sigma_g = sigma_g_x
    ar_step = State.update(ar_step, "log_g_x", - 0.5 * (1 - rho_g_x) * sigma_g**2 + rho_g_x * (State.log_g_x(prev_state)) )

    return ar_step


def shock_step_random(prev_state):
    shock_step = tf.zeros_like(prev_state)
    nB = prev_state.shape[0]
    random_normals_ = tf.cast(rng.normal([nB,3]),dtype=global_default_dtype)
    n_steady_state_batches = tf.cond(tf.constant(nB > 500, dtype=tf.bool), lambda: 50, lambda: 0)
    random_normals = tf.concat([random_normals_[:nB-n_steady_state_batches,:],tf.zeros([n_steady_state_batches,3],dtype=global_default_dtype)],0)
    shock_step = State.update(shock_step, "log_a_x",  random_normals[:,0] * sim_mult * sigma_a_x)
    shock_step = State.update(shock_step, "log_xi_x", random_normals[:,1] * sim_mult * sigma_xi_x)
    shock_step = State.update(shock_step, "log_g_x",  random_normals[:,2] * sim_mult * sigma_g_x)

    current_trans_prob = tf.expand_dims(1 - State.regime_x(prev_state),-1) * tf.expand_dims(trans_prob_regime_NT,0) + tf.expand_dims(State.regime_x(prev_state),-1) * tf.expand_dims(trans_prob_regime_SS,0)
    new_regime = tf.cast(tf.random.categorical(tf.math.log(current_trans_prob),1),dtype=global_default_dtype)
    shock_step = State.update(shock_step, "regime_x", new_regime[:,0])
    return shock_step

def policy_step(prev_state, policy_state):
    # coming from the lagged policy / definition
    policy_step = tf.zeros_like(prev_state)
    policy_step = State.update(policy_step, "disp_old_x", Definitions.disp_y(prev_state,policy_state))
    # policy_step = State.update(policy_step, "i_old_x", Definitions.i_nom_y(prev_state,policy_state))

    return policy_step

#================================================================== 
#   Special section in Dynamics, concerned with impulse-response 
#   experiment 1 - a_x_state
#   HARD-coded -- please add above manually the function call 
 
#------------------------------------------------------------------        

def total_step_no_shock(prev_state, policy_state):
    ar = AR_step(prev_state)
    
    shock = tf.zeros_like(prev_state)

    n_batches = prev_state.shape[0]
    regimes = State.regime_x(prev_state)

    shock = State.update(shock, "regime_x", regimes)  #maintain regime

    policy = policy_step(prev_state, policy_state)
    
    total = augment_state(ar + shock + policy)

    return total

def total_step_shock(prev_state, policy_state, mode=None):
    ar = AR_step(prev_state)
    
    shock = ir_shock(prev_state)

    policy = policy_step(prev_state, policy_state)
    
    total = augment_state(ar + shock + policy)

    return total

def ir_shock(prev_state):
    
    shock_step = tf.zeros_like(prev_state)

    n_batches = prev_state.shape[0]
    regimes = tf.concat([
        tf.zeros((n_batches - 2)//2,dtype=global_default_dtype),
        tf.ones((n_batches - 2)//2,dtype=global_default_dtype),
        tf.constant([1.],dtype=global_default_dtype),
        tf.constant([0.],dtype=global_default_dtype)],0)

    shock_step = State.update(shock_step, "regime_x", regimes)  #maintain regime

    # quantile -- 1 std shock
    # 1 std: 0.84134; 2 std: 0.97725 3 std: 0.99865
    quantile = 0.84134
    IRS_shock = norm.ppf([quantile]) #inverse quantile
    #print(norm.cdf(norm.ppf(quantile)))

    shock_vec_a_x = tf.concat([
        tf.cast(3*sigma_a_x*IRS_shock,dtype=global_default_dtype),
        tf.cast(sigma_a_x*IRS_shock,dtype=global_default_dtype),
        tf.cast(-sigma_a_x*IRS_shock,dtype=global_default_dtype),
        tf.cast(-3*sigma_a_x*IRS_shock,dtype=global_default_dtype),
        tf.zeros(4,dtype=global_default_dtype),
        tf.zeros(4,dtype=global_default_dtype),
        tf.cast(3*sigma_a_x*IRS_shock,dtype=global_default_dtype),
        tf.cast(sigma_a_x*IRS_shock,dtype=global_default_dtype),
        tf.cast(-sigma_a_x*IRS_shock,dtype=global_default_dtype),
        tf.cast(-3*sigma_a_x*IRS_shock,dtype=global_default_dtype),
        tf.zeros(4,dtype=global_default_dtype),
        tf.zeros(4,dtype=global_default_dtype),
        tf.zeros(2,dtype=global_default_dtype)],0)
    shock_step = State.update(shock_step,'log_a_x',shock_vec_a_x)

    shock_vec_xi_x = tf.concat([
        tf.zeros(4,dtype=global_default_dtype),
        tf.cast(3*sigma_xi_x*IRS_shock,dtype=global_default_dtype),
        tf.cast(sigma_xi_x*IRS_shock,dtype=global_default_dtype),
        tf.cast(-sigma_xi_x*IRS_shock,dtype=global_default_dtype),
        tf.cast(-3*sigma_xi_x*IRS_shock,dtype=global_default_dtype),
        tf.zeros(4,dtype=global_default_dtype),
        tf.zeros(4,dtype=global_default_dtype),
        tf.cast(3*sigma_xi_x*IRS_shock,dtype=global_default_dtype),
        tf.cast(sigma_xi_x*IRS_shock,dtype=global_default_dtype),
        tf.cast(-sigma_xi_x*IRS_shock,dtype=global_default_dtype),
        tf.cast(-3*sigma_xi_x*IRS_shock,dtype=global_default_dtype),
        tf.zeros(4,dtype=global_default_dtype),
        tf.zeros(2,dtype=global_default_dtype)],0)
    
    shock_step = State.update(shock_step,'log_xi_x',shock_vec_xi_x)

    shock_vec_g_x = tf.concat([
        tf.zeros(4,dtype=global_default_dtype),
        tf.zeros(4,dtype=global_default_dtype),
        tf.cast(3*sigma_g_x*IRS_shock,dtype=global_default_dtype),
        tf.cast(sigma_g_x*IRS_shock,dtype=global_default_dtype),
        tf.cast(-sigma_g_x*IRS_shock,dtype=global_default_dtype),
        tf.cast(-3*sigma_g_x*IRS_shock,dtype=global_default_dtype),
        tf.zeros(4,dtype=global_default_dtype),
        tf.zeros(4,dtype=global_default_dtype),
        tf.cast(3*sigma_g_x*IRS_shock,dtype=global_default_dtype),
        tf.cast(sigma_g_x*IRS_shock,dtype=global_default_dtype),
        tf.cast(-sigma_g_x*IRS_shock,dtype=global_default_dtype),
        tf.cast(-3*sigma_g_x*IRS_shock,dtype=global_default_dtype),
        tf.zeros(2,dtype=global_default_dtype)],0)

    shock_step = State.update(shock_step,'log_g_x',shock_vec_g_x)
    
    return shock_step