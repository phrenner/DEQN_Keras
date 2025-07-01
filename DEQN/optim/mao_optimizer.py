import tensorflow as tf

class MAOAdam(tf.Module):

    def __init__(self, learning_rate=1e-3, beta_1=0.9, beta_2=0.999, ep=1e-7):
        super(MAOAdam, self).__init__()
        # Initialize the Adam parameters
        self.beta_1 = tf.constant(beta_1)
        self.beta_2 = tf.constant(beta_2)
        self.learning_rate = learning_rate
        self.ep = ep
        self.v_dvar, self.s_dvar = [], []
        self.title = f"Adam: learning rate={self.learning_rate}"
        self.built = False
        # track iterations so we can call .iterations (Keras-like)
        self._iterations = tf.Variable(1, dtype=tf.int64, trainable=False)    

    @property
    def iterations(self):
        """
        Exposes an `iterations` property to mimic standard TF/Keras optimizers.
        """
        return self._iterations

    def apply_gradients(self, grads, vars):
      # Set up moment and RMSprop slots for each variable on the first call
        if not self.built:
            for indx_t,grad in enumerate(grads):
                self.v_dvar.append([])
                self.s_dvar.append([])
                for var in vars:
                    v = tf.Variable(tf.zeros(shape=var.shape))
                    s = tf.Variable(tf.zeros(shape=var.shape))
            
                self.v_dvar[indx_t].append(v)
                self.s_dvar[indx_t].append(s)
          
        self.built = True
        # Perform Adam updates
        t = self._iterations
        for indx_t,grad in enumerate(grads):
            for i, (d_var, var) in enumerate(zip(grad, vars)):
                # Moment calculation
                self.v_dvar[indx_t][i] = self.beta_1*self.v_dvar[indx_t][i] + (1-self.beta_1)*d_var
                # RMSprop calculation
                self.s_dvar[indx_t][i] = self.beta_2*self.s_dvar[indx_t][i] + (1-self.beta_2)*tf.square(d_var)
                # Bias correction
                v_dvar_bc = self.v_dvar[indx_t][i]/(1-(self.beta_1**t))
                s_dvar_bc = self.s_dvar[indx_t][i]/(1-(self.beta_2**t))
                # Update model variables
                var.assign_sub(self.learning_rate*(v_dvar_bc/(tf.sqrt(s_dvar_bc) + self.ep)))
        # Increment the iteration counter
        self._iterations.assign_add(1)


    
    def get_config(self):
        """
        Returns a dictionary of optimizer configuration, similar to Keras built-ins.
        Allows:  tf.print(optimizer.get_config())
        """
        return {
            "name": "MultiAdam",
            "learning_rate": float(self.learning_rate),
            "beta_1": float(self.beta_1),
            "beta_2": float(self.beta_2),
            "ep": float(self.ep),
        }

    def save_own_variables(self, optimizer_weights):
        """
        Save all the relevant internal state into the `optimizer_weights` dict,
        so that FlexCheckpoint can pickle it.
        """
        optimizer_weights["iterations"] = int(self.t)

        v_dvar_data = []
        for v_var in self.v_dvar:
            v_dvar_data.append(v_var.numpy())
        optimizer_weights["v_dvar"] = v_dvar_data

        s_dvar_data = []
        for s_var in self.s_dvar:
            s_dvar_data.append(s_var.numpy())
        optimizer_weights["s_dvar"] = s_dvar_data


    def load_own_variables(self, optimizer_weights):
        """
        Load internal state from a dictionary created by `save_own_variables`.
        """
        self.t.assign(optimizer_weights["iterations"])

        for i, arr in enumerate(optimizer_weights["v_dvar"]):
            self.v_dvar[i].assign(arr)
        for i, arr in enumerate(optimizer_weights["s_dvar"]):
            self.s_dvar[i].assign(arr)
