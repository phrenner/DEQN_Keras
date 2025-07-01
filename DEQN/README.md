# Dependencies

The basic dependencies are tensorflow, hydra-core and tensorboard (for monitoring). 

## Paralellization

The code can be run on GPU - in this case you need to configure the underlying environment by installing the necessary packages.

# Running the code

To run the code, just execute:

```
python run_deepnet.py 
```

## Model selection

The model is selected via the config/config.yaml file by setting 'MODEL_NAME'. You can then run the code by executing:

```
python run_deepnet.py
```

You can also specify where the checkpoints are stored by setting 'hydra: run: dir:'
    
## Speficying the model

### Model.py

This file is used to define the Neural Net and specifiying the optimizer to use. You can define any architecture here.

### Hooks.py

This file is used to define callbacks to Keras: 
- what to plot via Tensorboard,
- set a Learning Rate scheduler,
- set what to do with the initial state each episode

### Variables.py

Needs two dictonaries, one called config which needs to contain the following entries:
```
    "N_sim_batch": 1000,
    "N_epochs_per_episode": 1,
    "N_minibatch_size": 60,
    "N_episode_length": 30,
    "N_episodes": 10000000,
    "n_quad_pts": 3,
    "sorted_within_batch": False,
    "error_filename": "error_log.txt",
    "keras_precision": "float32"
```

Important to note: N_sim_batch is the number of trajectories simulated in parallel, so e.g. 1000 episodes are simulated in parallel. 
Each (parallel) episode is 30 long in this case, so altogether 30000 states are drawn at each episode iteration. 
These are then batched into N_minibatch_size for gradient steps (so in this case 60 minibatches are drawn of size 500, for each episode).

### Equations.py

Specify the equations here via a dictonary. Assuming your equations are the functions $f_i$. Then loss function used will be
$$\sum_i f_i^2 $$

### Dynamics.py

This file is used to define the state transition and how to compute the integral if you want to use a quadrature rule.

## Definitions.py

This file can be used to define auxiliary functions that help make the equations more readable.