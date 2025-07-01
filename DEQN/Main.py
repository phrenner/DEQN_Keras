import importlib
import tensorflow as tf
import Equilibrium
import Parameters
import gc
import keras
import numpy as np
from Equilibrium import penalty_bounds_policy

class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def on_test_end(self, logs=None):
        evaluation_loss = logs["pred_loss"]

        tf.print("\n------------------------------------------------------------------------------------------------------------------------")        
        tf.print(f"Testing error: {evaluation_loss:3e}")
        tf.print("\n========================================================================================================================")

    def on_epoch_end(self, epoch, logs=None):
        epoch_loss, net_epoch_loss = logs["epoch_loss"], logs["net_epoch_loss"]
        MSE_epoch_loss = epoch_loss
        Norm_epoch_loss = tf.math.sqrt(epoch_loss)
        MSE_epoch_no_penalty = net_epoch_loss
        Norm_epoch_loss_no_penalty = tf.math.sqrt(net_epoch_loss)   
        if np.isnan(MSE_epoch_loss) or np.isinf(MSE_epoch_loss):
            keras.src.utils.io_utils.print_msg(
                f"Invalid loss, terminating training"
            )
            self.model.stop_training = True   
            raise Exception("Invalid loss, terminating training")    
            
        if Parameters.current_episode.numpy() > 1:     
            file1 = open(Parameters.LOG_DIR + "/" + Parameters.error_filename,"a")
        else:
            file1 = open(Parameters.LOG_DIR + "/" + Parameters.error_filename,"w")
            file1.write("#Episode, MSE, MAE, MSE_no_penalty, MAE_no_penalty\n")
        tf.print("\n------------------------------------------------------------------------------------------------------------------------")
        tf.print(f"In episode: {Parameters.current_episode.numpy()}")
        tf.print(f"Normalized MSE epoch loss: {MSE_epoch_loss:3e}")
        tf.print(f"Net MSE epoch loss: {MSE_epoch_no_penalty:3e}")
        file1.write(f"{Parameters.current_episode.numpy()} {MSE_epoch_loss:3e} {Norm_epoch_loss:3e} {MSE_epoch_no_penalty:3e} {Norm_epoch_loss_no_penalty:3e}\n" )
        file1.close()     


Dynamics = importlib.import_module(Parameters.MODEL_NAME + ".Dynamics")
Hooks = importlib.import_module(Parameters.MODEL_NAME + ".Hooks")
Equations = importlib.import_module(Parameters.MODEL_NAME + ".Equations")

"""
Overview:
    - Episode > Epochs (1 full scan of an episode) > Mini-batches > Adam
"""

 
def run_cycle(state_episode):
    """ Runs an iteration cycle startin from a given BatchState.
    
    It creates an episode and then runs N_epochs_per_episode epochs on the data.
    """
    
    state_episode = Parameters.model.run_episode(state_episode)
    n_states = state_episode.shape[-1]

    effective_size = state_episode.shape[0] * state_episode.shape[1]
    if not Parameters.sorted_within_batch:
        state_episode_batched = tf.reshape(state_episode, [effective_size,n_states])
    else:
        state_episode_batched = tf.reshape(tf.transpose(state_episode,[1,0,2]), [effective_size,n_states])

    batches = tf.data.Dataset.from_tensor_slices(
        state_episode_batched).shuffle(buffer_size=int(effective_size/Parameters.N_minibatch_size)).batch(Parameters.N_minibatch_size, drop_remainder=True)

    Parameters.model.fit(batches, 
                         epochs=1, 
                         verbose=Parameters.VERBOSE,
                         callbacks=[
                             LossAndErrorPrintingCallback(),
                             Hooks.TensorboardCallback(
                                 state_episode[-1,...],
                                 penalty_bounds_policy),
                            keras.callbacks.LearningRateScheduler(Hooks.LearningRateScheduler),])
    # res = Parameters.model.evaluate(
    #     tf.reshape(state_episode, [effective_size,state_episode.shape[-1]]).numpy(),
    #     tf.reshape(policy_state_episode, [effective_size,policy_state_episode.shape[-1]]).numpy(),
    #     verbose=Parameters.VERBOSE,
    #     callbacks=[LossAndErrorPrintingCallback()],
    # )
        
    return state_episode

def run_cycles():


    if "post_init" in dir(Hooks) and Parameters.current_episode.numpy() < 2:
        print("Running post-init hook...")
        Hooks.post_init()
        print("Starting state after post-init:")
        print(Parameters.starting_state)

    state_episode = tf.tile(tf.expand_dims(Parameters.starting_state, axis = 0), [Parameters.N_episode_length, 1, 1])

    start_time = tf.timestamp()

    # compile model
    Parameters.model.compile(Equations.list_of_equations, Equilibrium.loss_specific, Dynamics.total_step_random)

    for i in range(Parameters.N_episodes):       
        tf.print("Running episode: " + str(Parameters.current_episode.numpy()))
        state_episode = run_cycle(state_episode)
        # start again from previous last state
        if Parameters.initialize_each_episode:
            print("Running with states re-drawn after each episode!") 
            # Parameters.starting_state.assign(Parameters.initialize_states())
            Parameters.starting_state.assign(state_episode[Parameters.N_episode_length-1,:,:])
            Hooks.post_init()
        else:
            Parameters.starting_state.assign(state_episode[Parameters.N_episode_length-1,:,:])
            
        state_episode = tf.tile(tf.expand_dims(Parameters.starting_state, axis = 0), [Parameters.N_episode_length, 1, 1])
        
        # create checkpoint
        Parameters.current_episode.assign_add(1)
        
        Parameters.manager.save()
                
        tf.print("Elapsed time since start: ", tf.timestamp() - start_time)

        if i % 10 == 0:
            tf.print("Garbage collecting")
            tf.keras.backend.clear_session()
            gc.collect()
