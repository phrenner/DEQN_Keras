import tensorflow as tf
import Definitions
import PolicyState
import State
from Parameters import beta, gamma, epsilon, theta, omega

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
    A_t = Definitions.a_x(state,policy_state)
    zeta_t = Definitions.zeta_y(state, policy_state)
    tau_t = Definitions.tau_x(state,policy_state)
    Xi_N_t = Definitions.num_y(state,policy_state)
    Xi_D_t = Definitions.den_y(state,policy_state)
    Delta_old = State.disp_old_x(state)
    vartheta_t = Definitions.vartheta_y(state,policy_state)
    rho_t = Definitions.rho_y(state,policy_state)
    vartheta_old = State.vartheta_old_x(state)
    rho_old = State.rho_old_x(state)
    c_old = State.c_old_x(state)
    w_t = Definitions.wage_y(state,policy_state)
    y_t = Definitions.y_tot_y(state, policy_state)
    g_t = Definitions.g_x(state,policy_state)
    pi_aux_t = Definitions.pi_aux_y(state,policy_state)
    p_star_aux_t = Definitions.p_star_aux_y(state,policy_state)
     
  #########################
    #normalized equation      
 
    loss_dict['eq_1'] = ((c_t + g_t)**omega * (Delta_t/A_t)**(1+omega))**(-1) * ( #normalizing factor
        c_t**(-gamma) - (c_t + g_t)**omega * (Delta_t/A_t)**(1+omega) + 
        vartheta_t * (((1 + omega) * c_t + gamma * (c_t + g_t)) * (c_t + g_t)**omega * c_t**(gamma - 1) * (Delta_t/A_t)**omega * (1 + tau_t) / A_t + 
                      E_t(lambda sold,psold,snew,psnew: beta * theta * gamma * Definitions.cons_y(sold,psold)**(gamma - 1) * Definitions.cons_y(snew,psnew)**(-gamma) * (Definitions.pi_aux_y(snew,psnew)) * Definitions.num_y(snew,psnew))) +
        (1/beta) * vartheta_old * (-gamma) * beta * theta * c_old**gamma * c_t**(-gamma-1) * pi_aux_t * Xi_N_t +
        rho_t * (1 + E_t(lambda sold,psold,snew,psnew: beta * theta * gamma * Definitions.cons_y(sold,psold)**(gamma - 1) * Definitions.cons_y(snew,psnew)**(-gamma) * (Definitions.pi_aux_y(snew,psnew))**((epsilon - 1.0)/epsilon) * Definitions.den_y(snew,psnew))) + 
        (1/beta) * rho_old * (-gamma) * beta * theta * c_old**gamma * c_t**(-gamma - 1) * pi_aux_t**((epsilon - 1.0)/epsilon) * Xi_D_t
    )
   
 #########################
   #normalized equation          
    loss_dict['eq_2'] = (((c_t + g_t) * Delta_t / A_t)**(1 + omega) / Delta_t )**(-1) * ( #normalizing factor
        - ((c_t + g_t) * Delta_t / A_t)**(1 + omega) / Delta_t - zeta_t + beta * E_t(lambda sold,psold,snew,psnew: Definitions.zeta_y(snew,psnew) * theta * (Definitions.pi_aux_y(snew,psnew))) + 
        vartheta_t * omega * (c_t + g_t)**(1 + omega) * c_t**(gamma) * (Delta_t / A_t)**omega / Delta_t * (1 + tau_t) / A_t
    )
 
 #########################
   #normalized equation      
 
    loss_dict['eq_3'] = (Xi_N_t)**(-1) * ( #normalizing factor
        Xi_N_t - y_t*w_t*(1.0 + tau_t)*A_t**(-1.0) - 
        E_t(lambda sold,psold,snew,psnew: beta*theta * (Definitions.lambda_y(snew,psnew)/Definitions.lambda_y(sold,psold)) * (Definitions.pi_aux_y(snew,psnew)) * Definitions.num_y(snew,psnew)))
 
   
 #########################
   #normalized equation          
    loss_dict['eq_4'] = (Xi_D_t)**(-1) * ( #normalizing factor
        Xi_D_t -  y_t - 
        E_t(lambda sold,psold,snew,psnew:   beta*theta * (Definitions.lambda_y(snew,psnew)/Definitions.lambda_y(sold,psold)) * (Definitions.pi_aux_y(snew,psnew))**((epsilon - 1.0)/epsilon) * Definitions.den_y(snew,psnew)) )
 
 
# #########################
#    #normalized equation   
  #  loss_dict['eq_5'] = (
  #        (theta*(pi_aux_t)**((epsilon - 1.0)/epsilon) + (1.0 - theta)*p_star_aux_t**((1-epsilon)/(-epsilon))) - 1.0)

# #########################
#    #normalized equation
#    loss_dict['eq_6'] = (Delta_t)**(-1) * (  #normalizing factor
#       (theta * (pi_aux_t) * Delta_old + (1 - theta) * p_star_aux_t) - Delta_t)


#########################
   #normalized equation
    loss_dict['eq_7'] = (p_star_t)**(1) * (  #normalizing factor
        p_star_aux_t**(1/epsilon) - p_star_t**(-1))

   
    return loss_dict


list_of_equations = [
   equations_main
]


