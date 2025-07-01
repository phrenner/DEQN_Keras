import Parameters
import PolicyState
import State
import Definitions
import tensorflow as tf
import sys
from Parameters import gamma, omega, M_cal, beta, theta, epsilon, g_bar, global_default_dtype

def y_tot_y(state, policy_state):
    #derived quantities
    return Definitions.cons_y(state, policy_state) + Definitions.g_x(state,policy_state)

def wage_y(state, policy_state):
    #derived quantities
    lambda_t = Definitions.lambda_y(state,policy_state)
    return (Definitions.h_work_y(state,policy_state))**omega/lambda_t

def p_star_y(state, policy_state):
    return M_cal*Definitions.num_y(state,policy_state)/Definitions.den_y(state, policy_state)

def lambda_y(state, policy_state):
    c_t = Definitions.cons_y(state,policy_state)
    return c_t**(-gamma)

def i_nom_y(state, policy_state):
    return PolicyState.i_nom_y(policy_state)

def h_work_y(state,policy_state):
    return Definitions.y_tot_y(state,policy_state) * Definitions.disp_y(state,policy_state) / (Definitions.a_x(state,policy_state))

def disp_y(state,policy_state):
    Delta_old = State.disp_old_x(state)
    pi_aux_t = Definitions.pi_aux_y(state,policy_state)
    p_star_aux_t = Definitions.p_star_aux_y(state,policy_state)    
    return (theta * (pi_aux_t) * Delta_old + (1 - theta) * p_star_aux_t)

def cons_y(state, policy_state):
    c_diff = PolicyState.cons_y(policy_state)
    return 1 + c_diff

def r_real_y(state, policy_state):    
    return Definitions.i_nom_y(state,policy_state) - Definitions.pi_tot_y(state,policy_state)

def mu_y(state, policy_state):
    zeta_t = Definitions.zeta_y(state,policy_state)
    p_star_aux_t = Definitions.p_star_aux_y(state, policy_state)
    eta_t = PolicyState.eta_y(policy_state)
    Xi_D_t = Definitions.den_y(state,policy_state)
    return (eta_t * (1-theta) * (1-epsilon) * p_star_aux_t + zeta_t * (1-theta) * (-epsilon) * p_star_aux_t**((-epsilon - 1)/(-epsilon)))/(Xi_D_t)

def vartheta_y(state,policy_state):
    mu_t = Definitions.mu_y(state,policy_state)
    vartheta_old = State.vartheta_old_x(state)
    c_old = State.c_old_x(state)
    c_t = Definitions.cons_y(state,policy_state)
    pi_aux_t = Definitions.pi_aux_y(state, policy_state)

    return M_cal * mu_t + (1/beta) * vartheta_old * beta * theta * c_old**(gamma) * c_t**(-gamma)  * pi_aux_t

def rho_y(state, policy_state):
    mu_t = Definitions.mu_y(state,policy_state)
    rho_old = State.rho_old_x(state)
    c_old = State.c_old_x(state)
    c_t = Definitions.cons_y(state,policy_state)
    pi_aux_t = Definitions.pi_aux_y(state, policy_state)
    p_star_t = Definitions.p_star_y(state, policy_state)    

    return (- mu_t * p_star_t + (1/beta) * rho_old * beta * theta * c_old**gamma * c_t**(-gamma)  * pi_aux_t**((epsilon-1)/epsilon))

def zeta_y(state, policy_state):
    eta_t = PolicyState.eta_y(policy_state)
    Delta_old = State.disp_old_x(state)
    Xi_D_t = Definitions.den_y(state,policy_state)
    Xi_N_t = Definitions.num_y(state,policy_state)
    rho_old = State.rho_old_x(state)
    vartheta_old = State.vartheta_old_x(state)
    c_old = State.c_old_x(state)
    c_t = Definitions.cons_y(state,policy_state)
    pi_aux_t = Definitions.pi_aux_y(state, policy_state)
    varphi_old = State.varphi_old_x(state)
    i_nom_old = State.i_nom_old_x(state)

    return -(c_t**(-gamma) * (vartheta_old * epsilon * pi_aux_t**((epsilon+1)/epsilon) * Xi_N_t * c_old**gamma 
                              + (epsilon - 1) * eta_t * c_t**gamma * pi_aux_t
                              + (epsilon - 1) * rho_old * Xi_D_t * c_old**gamma * pi_aux_t
                              - varphi_old * (1 + i_nom_old) / beta)
                              )/(epsilon * pi_aux_t**((epsilon+1)/epsilon) * Delta_old)


def chi_y(state, policy_state):
    E_t = State.E_t_gen(state, policy_state)
    return - beta * Definitions.varphi_y(state,policy_state) * (
        E_t(lambda sold,psold,snew,psnew: Definitions.cons_y(snew,psnew)**(-gamma) * Definitions.pi_aux_y(snew,psnew)**((epsilon-1)/epsilon) / Definitions.pi_aux_y(snew,psnew)))

def varphi_y(state,policy_state):
    return - PolicyState.varphi_y(policy_state)

def num_y(state,policy_state):
    Xi_N_diff = PolicyState.num_y(policy_state)
    return Xi_N_diff

def den_y(state,policy_state):    
    Xi_D_diff = PolicyState.den_y(policy_state)
    return Xi_D_diff

def pi_tot_y(state, policy_state):
    return Definitions.pi_aux_y(state,policy_state)**(1/epsilon) - 1

def pi_aux_y(state, policy_state):
    p_star_aux_t = Definitions.p_star_aux_y(state, policy_state)
    return (-((-1 + p_star_aux_t**(1 - 1/epsilon) - p_star_aux_t**(1 - 1/epsilon) * theta)/theta))**(epsilon/(-1 + epsilon))

def tau_x(s,ps):
    tau_bar = Definitions.tau_bar_x(s,ps)
    return tau_bar + Definitions.xi_x(s,ps)

def tau_bar_x(s,ps):
    regime = Definitions.regime_x(s,ps)
    return (1 - regime) * (-1/epsilon) + regime * 0.
    # return (-1/epsilon) / ((1/p_12) / (1/p_12 + 1/p_21))

def regime_x(s,ps):
    return (State.regime_x(s))
    
def a_x(s,ps):
    return tf.math.exp(State.log_a_x(s))

def xi_x(s,ps):
    return (State.log_xi_x(s))

def g_x(s,ps):
    return g_bar * tf.math.exp(State.log_g_x(s))

def cons_flex_y(state, policy_state):
    A_t = (Definitions.a_x(state,policy_state))
    tau_t = Definitions.tau_x(state,policy_state)
    g_t = Definitions.g_x(state,policy_state)
    return ((A_t**(1+omega))**(1/(omega + gamma)))

def lambda_flex_y(state, policy_state):
    c_t = cons_flex_y(state,policy_state)
    return c_t**(-gamma)

def i_flex_y(state, policy_state):
    E_t = State.E_t_gen(state, policy_state)
    return lambda_flex_y(state,policy_state) / (beta * E_t(lambda sold,psold,snew,psnew: lambda_flex_y(snew,psnew)/(1))) - 1

def out_gap_y(state, policy_state):
    return tf.math.log(Definitions.cons_y(state,policy_state) / Definitions.cons_flex_y(state,policy_state))

def p_star_aux_y(s,ps):
    return 1 + PolicyState.p_star_aux_y(ps)

def u_y(s,ps):
    c_t = Definitions.cons_y(s,ps)
    h_t = Definitions.h_work_y(s,ps)
    return c_t**(1-gamma)/(1-gamma)  -  h_t**(omega+1)/(omega+1)