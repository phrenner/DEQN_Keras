import Parameters
import tensorflow as tf
class ReduceLROnPlateau:
    def __init__(self, factor, patience, min_lr, min_delta, cooldown, verbose):
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.cooldown = cooldown
        self.verbose = verbose
        self.best_loss = float('inf')
        self.wait = 0
        self.best_weights = None
        self.cooldown_counter = 0
        self.name = 'ReduceLROnPlateau'

    def step(self, current_loss):

        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            if self.verbose:
                tf.print(f"Cooldown: {self.cooldown_counter} epochs left")
            return
 
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                current_lr = Parameters.optimizer.learning_rate.numpy()
                # tf.print(f"Current learning rate: {current_lr}")
                if current_lr > self.min_lr:
                    new_lr = max(Parameters.optimizer.learning_rate * self.factor, self.min_lr)
                    Parameters.optimizer.learning_rate.assign(new_lr)
                    if self.verbose:
                        tf.print(f"Reducing learning rate from {current_lr} to {new_lr}")
                else:
                    if self.verbose:
                        tf.print(f"Learning rate is already at minimum {current_lr}")
                self.wait = 0
                self.cooldown_counter = self.cooldown