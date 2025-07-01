import tensorflow as tf
import Definitions
import PolicyState
import State
from Parameters import beta, gamma, epsilon, theta, omega, M_cal, eps_CC

def Fischer_Burrmeister(a,b,eps):
    return (eps + a + b - tf.math.sqrt(a**2 + b**2 + 4*eps**2))/2

def NN_smoothing_func(a,b,eps):
    return b - eps*tf.math.log(1 + tf.math.exp((b-a)/eps))

def uniform_smoothing_func(a,b,eps):
    return tf.where(
        tf.less_equal(tf.math.abs(a-b),eps/2),
        a - (a-b+eps/2)**2/(2*eps),
        tf.minimum(a,b))

def picard_smoothing_func(a,b,eps):
    return tf.where(
        tf.less_equal(a,b),
        a - eps*tf.math.exp((a-b)/eps)/2,
        b - eps*tf.math.exp((b-a)/eps)/2)

def non_smooth_complementarity_1(a,b,eps):
    # return eps - tf.minimum(a,b)
    return eps - (-a - b + tf.maximum(a,b))

def non_smooth_complementarity_2(a,b,eps):
    return eps - a*b

@tf.function
def complementarity_function(a,b,eps):
    return Fischer_Burrmeister(a,b,eps)

@tf.function
def equations(state, policy_state):
   loss_main = equations_main(state, policy_state)
   loss_dict = loss_main

   return loss_dict



@tf.function
def equations_main(state,policy_state):
   E_t = State.E_t_gen(state, policy_state)
   
   loss_dict = {}

   #define vars vor readability
   c_t = Definitions.cons_y(state,policy_state)
   Delta_t = Definitions.disp_y(state,policy_state)
   p_star_t = Definitions.p_star_y(state, policy_state)
   A_t = (Definitions.a_x(state,policy_state))
   mu_t = Definitions.mu_y(state, policy_state)
   zeta_t = Definitions.zeta_y(state, policy_state)
   tau_t = Definitions.tau_x(state, policy_state)
   Xi_N_t = Definitions.num_y(state,policy_state)
   Xi_D_t = Definitions.den_y(state,policy_state)
   Delta_old = State.disp_old_x(state)
   w_t = Definitions.wage_y(state,policy_state)
   y_t = Definitions.y_tot_y(state, policy_state)
   g_t = Definitions.g_x(state,policy_state)
   pi_aux_t = Definitions.pi_aux_y(state,policy_state)
   varphi_t = Definitions.varphi_y(state,policy_state)
   chi_t = Definitions.chi_y(state,policy_state)
   i_nom_t = Definitions.i_nom_y(state,policy_state)
   p_star_aux_t = Definitions.p_star_aux_y(state,policy_state)

#########################
   #normalized equation
   loss_dict['eq_1'] = ((c_t + g_t)**omega*(Delta_t/A_t)**(1+omega))**(-1) * ( #normalizing factor
      c_t**(-gamma) - (c_t + g_t)**omega*(Delta_t/A_t)**(1+omega) + mu_t * (
      p_star_t * (1 + gamma * c_t**(gamma-1) * E_t(lambda sold,psold,snew,psnew: Definitions.F_fun_y(snew,psnew))) -
      M_cal * (((1 + omega) * c_t + gamma * (c_t + g_t)) * (c_t + g_t)**omega * c_t**(gamma - 1) * (1 + tau_t) * A_t**(-1) * (Delta_t/A_t)**omega +
      gamma * c_t**(gamma-1) * E_t(lambda sold,psold,snew,psnew: Definitions.G_fun_y(snew,psnew)))
      ) + varphi_t* gamma * c_t**(-gamma-1))

#########################
   #normalized equation
   loss_dict['eq_2'] = (((c_t + g_t) * Delta_t/A_t)**(1+omega)/Delta_t)**(-1) * ( #normalizing factor
      - ((c_t + g_t) * Delta_t/A_t)**(1+omega)/Delta_t + beta * E_t(lambda sold,psold,snew,psnew: Definitions.dV_dDelta(snew,psnew)) - zeta_t + mu_t * (
      p_star_t * c_t**gamma * E_t(lambda sold,psold,snew,psnew: Definitions.dF_dDelta(snew,psnew)) - M_cal * (
         (c_t + g_t)**(1 + omega) * c_t**(gamma) * (omega/Delta_t) * (Delta_t/A_t)**omega * (1 + tau_t) * A_t**(-1) + c_t**gamma * E_t(lambda sold,psold,snew,psnew: Definitions.dG_dDelta(snew,psnew))
      ))+ varphi_t * beta * (1 + i_nom_t) * E_t(lambda sold,psold,snew,psnew: Definitions.dH_dDelta(snew,psnew)))

#########################
   #normalized equation      

   loss_dict['eq_3'] = (Xi_N_t)**(-1) * ( #normalizing factor
        Xi_N_t - y_t*w_t*(1.0 + tau_t)*A_t**(-1.0) - 
        E_t(lambda sold,psold,snew,psnew:   beta*theta*(Definitions.lambda_y(snew,psnew)/Definitions.lambda_y(sold,psold)) * (Definitions.pi_aux_y(snew,psnew)) * Definitions.num_y(snew,psnew)))

#########################
   #normalized equation          
   loss_dict['eq_4'] = (Xi_D_t)**(-1) * ( #normalizing factor
         Xi_D_t - y_t - 
         E_t(lambda sold,psold,snew,psnew:   beta*theta*(Definitions.lambda_y(snew,psnew)/Definitions.lambda_y(sold,psold)) * (Definitions.pi_aux_y(snew,psnew))**((epsilon - 1.0)/epsilon) * Definitions.den_y(snew,psnew)) )

# #########################
#    #normalized equation   
#    loss_dict['eq_5'] = ((1.0 - theta) * p_star_t**(1-epsilon))**(-1) * (
#          (theta*(pi_aux_t)**((epsilon - 1.0)/epsilon) + (1.0 - theta)*p_star_t**(1.0 - epsilon)) - 1.0)

# #########################
#    #normalized equation
#    loss_dict['eq_6'] = (Delta_t)**(-1) * (  #normalizing factor
#       (theta * (pi_aux_t) * Delta_old + (1 - theta) * p_star_t**(-epsilon)) - Delta_t)

#########################
   #normalized equation
   loss_dict['eq_7'] = (  #normalizing factor
      complementarity_function(chi_t,i_nom_t,eps_CC))

#########################
   #normalized equation
   loss_dict['eq_8'] = (p_star_aux_t)**(-1) * (  #normalizing factor
      p_star_aux_t - p_star_t**(-epsilon))
   
   return loss_dict


list_of_equations = [
   equations_main
]

