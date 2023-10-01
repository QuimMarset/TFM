## Hyperparameters:

* use_cuda: By setting it to true, it uses GPU if CUDA is available

* repetitions: Number of times we train the same method with different seeds
  seed: Seed to use when training a mehtod. We can set it as a list of seeds, one for each repetition, or an integer and let the code increase it itself

* t_max: Number of steps the training will take. By steps we refer each time we execute an action in the environment, with each environment we use
  
* num_envs: Number of parallel environments to use to gather data when training. During evaluation it always uses 1 

* test_nepisode: Number of evaluation episodes to do
  
* test_interval: During training, we perform test_nepisode evaluation episodes after a certain number of steps. This parameter sets how many steps between evaluation episodes

* log_interval: How many training steps we wait to print the most recent stats (e.g. returns, losses or environment information)
  
* runner_log_interval: How many training steps we wait to update the logged data related to the training episodes return. We compute an average with the values of the episodes we have completed in those steps
  
* learner_log_interval: The same as the previous one, but logging the training stats like the losses and gradients. 
                        However, we do not compute an average but log the most recent ones

* checkpoint_path: The path containing the weights we want to load in the model. We can use it to fine-tune or to evaluate the learned policy
  
* save_model: If we set it to True, we will save the models each save_model_interval training steps. If False, we won't save the model
  
* save_model_end: If we set it to True, independently of the previous parameter, we will save the models once training ends
  
* save_model_interval: How many training steps we have to wait to save the current model's weights

* evaluate: If we set it to False, means we want to train a method. If we set it to True, means we want to evaluate some learned policy
  
* save_frames: When evaluating, we can save the frames when rendering an episode to see how a method has learned to solve an environment. However, we need to set the render mode of the environment to rbg_array

* l2_reg_coef: Regularization coefficient used in Adam optimizers
  
* optimizer_epsilon: Epsilon used in Adam optimizers

* gamma: Discount factor used to compute the discounted return

* action_selector: In methods working with discrete actions, we output all the action values. Thus, when exploring, we need to select some action. Currently, we only have epsilon-greedy exploration

* n_batches_to_sample: Methods not relying on recurrent layers sample as many mini-batches of random transitions as septs in an episode. However, if we run several parallel environments, we might want to update fewer times the networks
  
* buffer_transitions: If we set it to True, the buffer stores random transitions without caring about the order. If we set it to False, it stores complete episodes keeping the order of the transitions
  
* batch_size: Number of transitions/episodes to sample each time we want to update the networks 
  
* buffer_size: Maximum number of transitions/episodes to store. Usually, we store fewer episodes than transitions

* obs_entity_mode & state_entity_mode: We use these parameters in TransfQMIX to take into account that the observations and state should be processed as matrices instead of vectors

* start_steps: Methods working with continuous actions usually allow to sample random actions to fill the replay buffer, instead of using the one the policy outputs. This parameter determines how many training steps to output random actions

* add_agent_id: If we set it to True, we will add agent identifiers as a one-hot encoding to the inputs to the networks. This parameter applies to value-based methods, and actor-critic, adding the identifiers to the actor
  
* critic_add_agent_id: Same as the previous parameter, but only applies to actor-critic methods, adding the identifiers to the critic. Nonetheless, with MADDPG, which has a monolithic critic, we do not use it

* increase_step_counter: We use this parameter and the next one when sampling noise from a Gaussian in the contniuous methods.  As we decay that noise as training proceeds, we keep a counter to know at which step we are to get the current Gaussian sigma. To increase that counter, we have to consider if we use several parallel environments, because we might want to increase that counter after all the environments do a step, or after any does a step. If we set it to True, we will increase it after each step or any environment, and if we set it to False, we will increase it after all environments do a step. Hence, we will sample the noise with the same sigma with each environment, or a slightly different one. However, there is no significant difference in using it or not

* use_training_steps_to_compute_target_noise: This parameter is similar to the previous one, but only applies to TD3 and JAD3 because they add noise to the target actions. If we set it to True, we will compute the Gaussian using the counter we mentioned in the previous parameter. But if we set it to False, we will use a counter that counts how many times we have updated the networks. Like the previous one, we did not observe significant difference in using it or not

* decay_type: Determines how we decay the Gaussian sigma, and can take three values: linear, exponential, and polynomial
  
* power: Related to decay_type, and only applies if we use polynomial decay to set the power of that polynomial

* sigma_start: The initial value of the Gaussian sigma
  
* sigma_finish: The final value of the Gaussian sigma
  
* sigma_anneal_time: In how many steps we want the initial value to decay until the final one. Mostly, it takes the same value as t_max

* target_sigma_start & target_sigma_finish & target_sigma_anneal_time: The same set of parameters when we need to add noise to the actions we sample with the target networks
  
* target_noise_clipping: The maximum magnitude we allow the target noise to have. In TD3 and JAD3, we clip that noise

* update_actor_targets_freq: The frequency at we update the actor and the target networks in TD3 and JAD3

* num_previous_transitions: Only applies to multi-agent continuous methods, and is used to determine how many previous (observation, action) pairs we want to add as input to the actor and critic together with the current observation critic_use_previous_transitions: By default, we only add previous transitions to the actor. We can set it to True to add them to the critic 

* add_last_action: If we set it to True in multi-agent methods working with recurrent layers, we add the previous action as input together with the observation

* update_actor_with_joint_qs: Only applies to JAD3, and by setting it to True, we update the actor with the joint action-value function instead of the individual ones 
* use_min_to_update_actor: Only applies to JAD3, and by setting it to True, we update the actor computing the minimum between the two approximations of the action-value functions (i.e. like SAC)

* lr: Learning rate used in the Adam optimizer to update the networks in value-based methods, and to update the actor in actor-critic methods
  
* critic_lr: Learning rate used in the Adam optimizer to update the actor-critic methods' critic

* grad_norm_clip_actor: Maximum norm we allow in the gradients before clipping them. We use it when updating the networks in value-based methods, and the actor in actor-critic ones
* grad_norm_clip_critic: Like the previous one, but used when updating the critic in actor-critic methods

* learner & controller & critic_controller: Determine which  Python class they use to explore the environment and update the networks. They are bound to each methods, so do not change them
  
* agent: Determines which type of network to use, according to the `__init__` inside modules/agents. We do not recommend changing the types of layer 
         it uses (i.e. MLP or RNN), but you can change how many networks we train. It only applies to value-based methods and the actor in actor-critic ones
* critic: Determines which critic network architecture to use, according to the `__init__` inside modules/critic. Again, we do not recommend changing the types of layer
  
* mixer: Determines which mixer to use in those factorizing the joint action-value function. Currently, only QMIX and VDN are implemented

* mixing_embed_dim: The number of units per layer in the mixer network. All the layers use the same number
  
* hypernet_layers: The number of hypernetwork layers to compute the weights of the mixer network
  
* hypernet_embed: The number of units in each layer of each hypernetwork
  
* hidden_dim: Number of units to use in the actors, critics, and action-value networks used in value-based methods.
  
* hidden_dim_critic: Only applies to MADDPG, and sets the number of units in the centralized critic

* target_update_mode: If we set it to soft, we update the target networks using polyak average. If we set it to hard, we update them each hard_update_interval training steps
* target_update_tau: The rho value to use in polyak average, usually close to 0
* hard_update_interval: The number of trainig steps (i.e. number of times we update the networks, not steps in the environments) we wait to update the target networks by copying the parameters of the non-target ones

* actions_regularization: In some methods, when we compute the actor loss, we allow to add a regularization term to penalize too large actions. We can set it to True to indicate we want to use that term