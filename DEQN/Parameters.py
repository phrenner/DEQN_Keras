#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 09:07:14 2020

@author: -
"""
import tensorflow as tf
import hydra
import os
import sys
from omegaconf import OmegaConf
from FlexCheckpoint import FlexCheckpoint, FlexCheckpointManager

import Globals
Globals.POST_PROCESSING=False



#### Configuration setup
@hydra.main(config_path="config", config_name="config.yaml")
def set_conf(cfg):
    print(OmegaConf.to_yaml(cfg))

    enable_tb = cfg.get("ENABLE_TB",False)
    if enable_tb:
        print("Enabling Tensorboard...")
    
    setattr(sys.modules[__name__], "ENABLE_TB", enable_tb)

    verbosity = cfg.get("VERBOSE", 0)

    setattr(sys.modules[__name__], "VERBOSE", verbosity)
    
    # debug
    if cfg.get("enable_check_numerics"):
        print("Enabling numerics debugging...")
        tf.debugging.enable_check_numerics(stack_height_limit=30, path_length_limit=50)
    
    # the model we are running
    MODEL_NAME = cfg.MODEL_NAME
    setattr(sys.modules[__name__],"MODEL_NAME", MODEL_NAME)
    
    seed_offset = 0
    
    # RNG       
    tf.random.set_seed(cfg.seed + seed_offset)
    rng_state = ([231, 45, cfg.seed + seed_offset])
    setattr(sys.modules[__name__], "rng", tf.random.Generator.from_state(rng_state, alg='philox'))
    

    # VARIABLES
    try:
        import importlib
        variables = importlib.import_module(MODEL_NAME + ".Variables")
    except ImportError:
        raise Exception("Something went wrong with importing the Variables module and core parameters. Please check the MODEL_NAME in the config file and Variables.py.")
        
    config_states = variables.states
    config_policies = variables.policies
    config_definitions = variables.definitions
    config_constants = variables.constants
    config_n_states = variables.n_states
    # for backward compatibility in case constants are also in yaml
    print("Variables imported from Variables module")
    print(__name__)

    config = variables.config

    # RUN CONFIGURATION
    setattr(sys.modules[__name__], "N_sim_batch", config["N_sim_batch"])
    setattr(sys.modules[__name__], "N_epochs_per_episode", config["N_epochs_per_episode"])
    setattr(sys.modules[__name__], "N_minibatch_size", config["N_minibatch_size"])
    setattr(sys.modules[__name__], "N_episode_length", config["N_episode_length"])
    setattr(sys.modules[__name__], "N_episodes", config["N_episodes"])
    setattr(sys.modules[__name__], "n_quad_pts", config.get('n_quad_pts',3))
    setattr(sys.modules[__name__], "expectation_pseudo_draws", config.get('expectation_pseudo_draws',5))
    setattr(sys.modules[__name__], "expectation_type", config.get('expectation_type','product'))
    setattr(sys.modules[__name__], "sorted_within_batch", config.get('sorted_within_batch',False))
    if sorted_within_batch and N_episode_length < N_minibatch_size:
        print("WARNING: minibatch size is larger than the episode length and sorted batches were requested!")
    # OUTPUT FILE FOR ERROR MEASURES
    setattr(sys.modules[__name__], "error_filename", config["error_filename"])

    setattr(sys.modules[__name__], "states", [s['name'] for s in config_states])
    setattr(sys.modules[__name__], "n_states", config_n_states) #input dimension of NN
    setattr(sys.modules[__name__], "policy_states", [s['name'] for s in config_policies])
    setattr(sys.modules[__name__], "definitions", [s['name'] for s in config_definitions])
    
    
    state_bounds = {"lower": {}, "penalty_lower": {}, "upper": {}, "penalty_upper": {}}

    for s in config_states:
        if "bounds" in s.keys() and "lower" in s["bounds"].keys():
            state_bounds["lower"][s["name"]] = s["bounds"]["lower"]

        if "bounds" in s.keys() and "upper" in s["bounds"].keys():
            state_bounds["upper"][s["name"]] = s["bounds"]["upper"]

    setattr(sys.modules[__name__], "state_bounds_hard", state_bounds)
    
    policy_bounds = {'lower': {}, 'penalty_lower': {}, 'upper': {}, 'penalty_upper': {}}

    for s in config_policies:
        if 'activation' in s.keys():
            if  s['activation'] != 'implied':
                if 'bounds' in s.keys() and 'lower' in s['bounds'].keys():
                    policy_bounds['lower'][s['name']] = s['bounds']['lower']
                    if 'penalty_lower' in s['bounds'].keys():
                        penalty = s["bounds"]["penalty_lower"]
                    else:
                        penalty = None
                    policy_bounds['penalty_lower'][s['name']] = penalty
            
                if 'bounds' in s.keys() and 'upper' in s['bounds'].keys():
                    policy_bounds['upper'][s['name']] = s['bounds']['upper']
                    if 'penalty_upper' in s['bounds'].keys():
                        penalty = s["bounds"]["penalty_upper"]
                    else:
                        penalty = None
                    policy_bounds['penalty_upper'][s['name']] = penalty
        else:
            if 'bounds' in s.keys() and 'lower' in s['bounds'].keys():
                policy_bounds['lower'][s['name']] = s['bounds']['lower']
                if 'penalty_lower' in s['bounds'].keys():
                    penalty = s["bounds"]["penalty_lower"]
                else:
                    penalty = None
                policy_bounds['penalty_lower'][s['name']] = penalty
        
            if 'bounds' in s.keys() and 'upper' in s['bounds'].keys():
                policy_bounds['upper'][s['name']] = s['bounds']['upper']
                if 'penalty_upper' in s['bounds'].keys():
                    penalty = s["bounds"]["penalty_upper"]
                else:
                    penalty = None
                policy_bounds['penalty_upper'][s['name']] = penalty
            

    setattr(sys.modules[__name__], "policy_bounds_hard", policy_bounds)

    definition_bounds = {'lower': {}, 'penalty_lower': {}, 'upper': {}, 'penalty_upper': {}}

    for s in config_definitions:
        if 'bounds' in s.keys() and 'lower' in s['bounds'].keys():
            definition_bounds['lower'][s['name']] = s['bounds']['lower']
            if 'penalty_lower' in s['bounds'].keys():
                penalty = s["bounds"]["penalty_lower"]
            else:
                penalty = None
            definition_bounds['penalty_lower'][s['name']] = penalty
    
        if 'bounds' in s.keys() and 'upper' in s['bounds'].keys():
            definition_bounds['upper'][s['name']] = s['bounds']['upper']
            if 'penalty_upper' in s['bounds'].keys():
                penalty = s["bounds"]["penalty_upper"]
            else:
                penalty = None
            definition_bounds['penalty_upper'][s['name']] = penalty
            
    setattr(sys.modules[__name__], "definition_bounds_hard", definition_bounds)
    
    # NEURAL NET
    tf.keras.backend.set_floatx(config.get('keras_precision','float32'))
    if config.get('keras_precision','float32') == 'float32':
        default_dtype = tf.float32
    elif config.get('keras_precision','float32') == 'float64':
        default_dtype = tf.float64
    elif config.get('keras_precision','float32') == 'float16':
        default_dtype = tf.float16
    else:
        raise Exception("should not happen")
    
    setattr(sys.modules[__name__], "global_default_dtype", default_dtype)

    current_episode_ = tf.Variable(1, dtype=tf.int32, trainable=False)
    setattr(sys.modules[__name__], "current_episode", current_episode_)
    # NEURAL NET
    try:
        config["ENABLE_TB"] = cfg.get("ENABLE_TB",False)
        model_class = importlib.import_module(MODEL_NAME + ".Model")
        setattr(sys.modules[__name__], "model", model_class.Model(MODEL_NAME,config,states,policy_states,config_policies,current_episode,global_default_dtype))

    except ImportError:

        raise Exception("Something went wrong with importing the Net module and core parameters. Please check the MODEL_NAME in the config file and Net.py.")


    #print net summary
    model.policy_net.summary()

    setattr(sys.modules[__name__], "policy", model.policy)
    setattr(sys.modules[__name__], "policy_net", model.policy_net)

    
    # CONSTANTS
    for (key, value) in config_constants.items():
        setattr(sys.modules[__name__], key, value)

    # STATE INITIALIZATION
    def initialize_states(N_batch = N_sim_batch):
        # starting state
        init_val = tf.ones([N_batch, len(states)],dtype=global_default_dtype)
        return init_val

    starting_state = tf.Variable(initialize_states())
    
    setattr(sys.modules[__name__], "starting_state", starting_state)
    setattr(sys.modules[__name__], "initialize_states", initialize_states)
    setattr(sys.modules[__name__], "initialize_each_episode", cfg.get("initialize_each_episode",False)) 
    setattr(sys.modules[__name__], "N_simulated_batch_size", cfg.get("N_simulated_batch_size",None))
    setattr(sys.modules[__name__], "N_simulated_episode_length", cfg.get("N_simulated_episode_length",None)) 

    # LOGGING
    setattr(sys.modules[__name__], "LOG_DIR", os.getcwd())
    
    if cfg.STARTING_POINT == 'NEW':
        for file in os.scandir(os.getcwd()):
            if not ".hydra" in file.path:
                os.unlink(file.path)
            

    ckpt = FlexCheckpoint(current_episode=current_episode, optimizer=model.optim_lst, policy=policy_net, rng_state=rng_state, starting_state=starting_state)
    manager = FlexCheckpointManager(ckpt, os.getcwd(), max_to_keep=cfg.MAX_TO_KEEP_NUMBER, checkpoint_interval=cfg.CHECKPOINT_INTERVAL)

    if cfg.STARTING_POINT == 'LATEST' and manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
        if cfg.get("reload_optimizer",False):
            ckpt.restore(manager.latest_checkpoint,load_optim=True)
        else:
            ckpt.restore(manager.latest_checkpoint)
         
    if cfg.STARTING_POINT != 'LATEST' and cfg.STARTING_POINT != 'NEW':
        print("Restored from {}".format(cfg.STARTING_POINT))
        if cfg.get("reload_optimizer",False):
            ckpt.restore(cfg.STARTING_POINT,load_optim=True)
        else:
            ckpt.restore(cfg.STARTING_POINT)

    tf.print("Optimizer configuration:")
    setattr(sys.modules[__name__], "optimizer_starting_iteration", model.optim_lst[0].iterations.numpy())
    tf.print(model.optim_lst[0].get_config())

    tf.print("Starting state:")
    tf.print(starting_state)    

    setattr(sys.modules[__name__], "ckpt", ckpt)
    setattr(sys.modules[__name__], "manager", manager)    
    

set_conf()
