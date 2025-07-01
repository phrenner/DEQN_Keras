import tensorflow as tf
import Definitions
import PolicyState
import State
from Parameters import beta, rho_i_x, epsilon, pi_bar, psi, theta, gamma, omega


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
   A_t = (Definitions.a_x(state,policy_state))
   tau_t = Definitions.tau_x(state, policy_state)
   p_star_t = Definitions.p_star_y(state, policy_state)
   p_star_aux_t = Definitions.p_star_aux_y(state,policy_state)
   pi_aux_t = Definitions.pi_aux_y(state,policy_state)
   Xi_N_t = Definitions.num_y(state,policy_state)
   Xi_D_t = Definitions.den_y(state,policy_state)
   w_t = Definitions.wage_y(state,policy_state)
   y_t = Definitions.y_tot_y(state, policy_state)
   lambda_t = Definitions.lambda_y(state, policy_state)
   Delta_t = Definitions.disp_y(state, policy_state)
   Delta_old = State.disp_old_x(state)
   i_nom_t = Definitions.i_nom_y(state, policy_state)

#########################
   #normalized equation
   loss_dict['eq_1'] = (lambda_t)**(-1) * (
      lambda_t - beta * E_t(
         lambda sold,psold,snew,psnew: (Definitions.lambda_y(snew,psnew)) * (1 + Definitions.i_nom_y(sold,psold)) * (Definitions.pi_aux_y(snew,psnew))**((epsilon-1)/epsilon) / Definitions.pi_aux_y(snew,psnew))
      )

#########################
   #normalized equation      
   loss_dict['eq_2'] = (Xi_N_t)**(-1) * ( #normalizing factor
      Xi_N_t - w_t*(1.0 + tau_t)*(y_t/A_t) - 
      E_t(lambda sold,psold,snew,psnew:   beta*theta*(Definitions.lambda_y(snew,psnew)/Definitions.lambda_y(sold,psold)) * (Definitions.pi_aux_y(snew,psnew)) * Definitions.num_y(snew,psnew)))

#########################
   #normalized equation          
   loss_dict['eq_3'] = (Xi_D_t)**(-1) * ( #normalizing factor
         Xi_D_t - y_t - 
         E_t(lambda sold,psold,snew,psnew:   beta*theta*(Definitions.lambda_y(snew,psnew)/Definitions.lambda_y(sold,psold)) * (Definitions.pi_aux_y(snew,psnew))**((epsilon - 1.0)/(epsilon)) * Definitions.den_y(snew,psnew)) )

   # #########################
   #    #normalized equation   
   # loss_dict['eq_4'] = ((1-theta)*p_star_aux_t**((1-epsilon)/(-epsilon)))**(-1) * (
   #       (theta*(pi_aux_t)**((epsilon - 1.0)/epsilon) + (1 - theta)*p_star_aux_t**((1-epsilon)/(-epsilon))) - 1.0)

   # #########################
   #    #normalized equation
   loss_dict['eq_5'] = (Delta_t)**(-1) * (  #normalizing factor
         (theta * (pi_aux_t) * Delta_old + (1-theta) * p_star_aux_t) - Delta_t)

   #########################
   #    normalized equation
   loss_dict['eq_7'] = (p_star_t)**(1) * (  #normalizing factor
         p_star_aux_t**(1/epsilon) - p_star_t**(-1))
   
   #########################
   #    normalized equation
   loss_dict['eq_8'] = (
      (1 + (rho_i_x * State.i_old_x(state) + (1 - rho_i_x) * (1/beta*(1 + pi_bar) - 1 + psi * (Definitions.pi_tot_y(state,policy_state) - pi_bar)))) - (1 + i_nom_t))

   
   return loss_dict


list_of_equations = [
   equations_main
]