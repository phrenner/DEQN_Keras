import tensorflow as tf
import importlib
import keras
import os

    
class Base_Model_Class(keras.Model):
    def __init__(self, MODEL_NAME,config,states,policy_states,policy_dict,current_episode,global_default_dtype):
        super().__init__()
        self.config = config
        self.global_default_dtype = global_default_dtype
        self.MODEL_NAME = MODEL_NAME
        self.states = states
        self.policy_states = policy_states
        self.policy_dict = policy_dict
        self.current_episode = current_episode
        if self.config["ENABLE_TB"]:
            self.writer = tf.summary.create_file_writer(os.getcwd())

        self.pred_epoch_loss_tracker = keras.metrics.MeanAbsoluteError(name="pred_loss")
        self.epoch_loss_tracker = keras.metrics.Mean(name="epoch_loss")
        self.net_epoch_loss_tracker = keras.metrics.Mean(name="net_epoch_loss")

        tf.keras.backend.set_floatx(config.get('keras_precision','float32'))




    # Evaluate the neural net
    @tf.function
    def policy(self,s):
        raw_policy = self.policy_net(s[...,:len(self.states)])
        for i, pol in enumerate(self.policy_dict):
            if 'activation' in pol.keys():
                activation_str = pol['activation']
                if pol['activation'] == 'implied':
                    if 'lower' in pol['bounds'].keys() and 'upper' in pol['bounds'].keys():
                        activation_str = 'lambda x: {l} + ({u} - {l}) * tf.math.sigmoid(x)'.format(l=str(pol['bounds']['lower']), u=str(pol['bounds']['upper']))
                raw_policy = tf.tensor_scatter_nd_update(raw_policy,[[j,i] for j in range(s.shape[0])],eval(activation_str)(raw_policy[:,i]))        
                            
        if self.config["keras_precision"] == 'float64':
            return tf.cast(raw_policy, dtype=self.config.get('keras_precision','float32'))
        
        return raw_policy

    @property
    def metrics(self):
        return [self.epoch_loss_tracker, self.net_epoch_loss_tracker, self.pred_epoch_loss_tracker]

    def compile(self, list_of_equations, loss_fn, total_step_random):
        super().compile()
        assert len(list_of_equations) == len(self.list_of_models), "Number of equations must match number of models"
        self.gradient_step_func = []
        self.total_step_random = total_step_random
        for indx in range(len(self.optim_lst)):
            self.gradient_step_func.append(self.wrap_training(self.list_of_models[indx],list_of_equations[indx],self.optim_lst[indx],indx))        

        self.loss_fn = loss_fn

    def wrap_training(self,model,equations,optimizer,indx):
        @tf.function
        def run_specific_grads(state_sample):
            """Runs a single gradient step using Adam for a minibatch"""
            with tf.GradientTape() as tape:
                loss, net_loss = self.loss_fn(state_sample, self.policy, equations, self.optim_lst[indx],indx)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            
            return loss, net_loss
    
        return run_specific_grads


    def train_step(self, mini_batch):
        """Runs a single gradient step using Adam for a minibatch"""
        epoch_loss, net_epoch_loss = 0., 0.
        for indx in range(len(self.optim_lst)):
            epoch_loss_1, net_epoch_loss_1 = self.gradient_step_func[indx](mini_batch)
            epoch_loss += epoch_loss_1
            net_epoch_loss += net_epoch_loss_1

        self.epoch_loss_tracker.update_state(epoch_loss)
        self.net_epoch_loss_tracker.update_state(net_epoch_loss)
        return {
            "epoch_loss": self.epoch_loss_tracker.result(),
            "net_epoch_loss": self.net_epoch_loss_tracker.result(),
        }        

    def test_step(self, data):
        # Unpack the data
        state, policy_state_old = data
        # Compute predictions
        policy_state_new = self.policy(state)

        # Update the metrics.
        self.pred_epoch_loss_tracker.update_state(y_pred=policy_state_new,y_true=policy_state_old)

        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
    
    @tf.function
    def do_random_step(self,current_state):
        policy_state_episode = self.policy(current_state)
        return self.total_step_random(current_state, policy_state_episode)

    def run_episode(self,state_episode):
        """ Runs an episode starting from the begging of the state_episode. Results are saved into state_episode."""
        current_state = state_episode[0,:,:]
        # run optimization
        for i in range(1, state_episode.shape[0]):
            current_state = self.do_random_step(current_state)
            # replace above for deterministic results
            # current_state = BatchState.total_step_spec_shock(current_state, Parameters.policy(current_state),0)
            state_episode = tf.tensor_scatter_nd_update(state_episode, tf.constant([[ i ]]), tf.expand_dims(current_state, axis=0))
        
        return state_episode    