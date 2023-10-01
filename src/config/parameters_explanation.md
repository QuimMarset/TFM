## Hyperparameters:

* use_cuda: If we set it to ``True``, it trains using GPU if CUDA is available.

* repetitions: Number of times we train the same method with different seeds.

* seed: Seed to use when training a method. We can set it as a list, one for each repetition, or as an integer and let the code increase it. In most cases, we have used the seed 0. If you encounter the parameter `seeds` in one of the save configurations, we set a list instead, and you should also use them to reproduce our results.

* t_max: Number of training steps. A step is each action we execute in each of the parallel environments.

* num_envs: Number of parallel environments we use to gather data when training. During the evaluation, we always used 1.

* test_nepisode: Number of evaluation episodes to do. We execute them during training, at a particular frequency, and to evaluate the learned policy after training ends.
  
* test_interval: The frequency, expressed in training steps, at which we perform test_nepisode evaluation episodes during training to assess the current policy's performance (i.e. executing the real action, not the exploratory one).

* log_interval: The frequency, expressed in training steps, at which we print the most recent stats (e.g. returns, losses or environment information).
  
* runner_log_interval: The frequency, expressed in training steps, at which we update the episode return. We compute an average of all the episodes finished in that frequency and use it to generate the plots.
  
* learner_log_interval: The frequency, expressed in training steps, at which we log the training stats like the losses and gradients. We do not compute an average but log the most recent ones. We do not generate plots with those stats but save them during the experiment.

* checkpoint_path: The path containing the weights we want to load in the different networks. We can use it to fine-tune or evaluate the learned policy.
  
* save_model: If we set it to `True`, we will save the networks at the frequency specified by save_model_interval training steps. If we set it to  `False`, we won't save the model.
  
* save_model_end: If we set it to `True`, independently of the previous parameter, we will save the models once the training ends.
  
* save_model_interval: The frequency, expressed in training steps, at which we save the current networks' weights.

* evaluate: If we set it to `False`, we will train a method and evaluate some learned policy otherwise.
  
* save_frames: If we set it to `True`, we will save the frames the environment renders when evaluating a policy. However, we need `render_mode` in `rbg_array` mode.

* render_mode: It takes to values: `human` and `rgb_array`. The former will render the environment to see how it behaves, while the latter will save that rendering as frames, not showing any window.

* l2_reg_coef: Regularization coefficient used in Adam optimizers.
  
* optimizer_epsilon: The epsilon used in the Adam optimizers.

* gamma: The discount factor used to compute the discounted return.

* action_selector: Methods working with discrete actions output all the action values, and we need to select one when exploring. Currently, we only have epsilon-greedy exploration.

* n_batches_to_sample: Methods not relying on recurrent layers sample as many mini-batches of random transitions as septs in an episode. However, if we run several parallel environments, we might want to sample fewer.
  
* buffer_transitions: If we set it to `True`, the replay buffer stores random transitions without caring about the order. If we fix it to `False`, it stores complete episodes, keeping the order.
  
* batch_size: Number of transitions/episodes to sample each time we want to update the networks.
  
* buffer_size: Maximum number of transitions/episodes to store. We save fewer episodes than transitions.

* obs_entity_mode & state_entity_mode: We use these parameters in TransfQMIX to treat the observations and state as matrices instead of vectors.

* start_steps: Methods working with continuous actions usually start sampling random ones to fill the replay buffer instead of using the one the policy outputs. This parameter determines how many training steps the method will execute arbitrary actions.

* add_agent_id: If we set it to `True`, we will add agent identifiers as a one-hot encoding to the inputs to the networks. This parameter applies to value-based methods and actor-critic (adding the identifiers to the actor).
  
* critic_add_agent_id: Same as the previous parameter, but only applies to actor-critic methods, adding the identifiers to the critic. Nonetheless, with MADDPG, which has a monolithic critic, we do not use it.

* increase_step_counter: We use this parameter and the next one when sampling noise from a Gaussian in methods working with continuous actions. As we decay that noise as training proceeds, we keep a counter of the training step to get the current Gaussian sigma. To increase that counter, we have to consider how many environments we use because we might want to increase that counter after all do a step or after any does one. If we set it to `True`, we will update it after one step on any environment, and if we set it to `False`, we will increase it after all the environments perform a transition. Hence, we will sample the noise with the same sigma with each environment or a slightly different one. However, there is no significant difference in using it or not.

* use_training_steps_to_compute_target_noise: This parameter is similar to the previous one but only applies to TD3 and JAD3 because they add noise to the target actions. If we set it to `True`, we will compute the Gaussian using the counter mentioned in the previous parameter. But if we set it to `False`, we will use a counter that counts how many times we have updated the networks. Like the previous one, we did not observe a significant difference in using it or not.

* decay_type: Determines how we decay the Gaussian sigma and can take three values: `linear`, `exponential`, and `polynomial`.
  
* power: Related to the previous parameter, it only applies if we use polynomial decay to set the power of that polynomial.

* sigma_start: The initial value of the Gaussian sigma.
  
* sigma_finish: The final value of the Gaussian sigma.
  
* sigma_anneal_time: The frequency, expressed in training steps, at which we want the initial sigma value to decay until the final one. Mostly, it takes the same value as t_max.

* target_sigma_start & target_sigma_finish & target_sigma_anneal_time: The same set of parameters when we need to add noise to the actions we sample with the target networks.
  
* target_noise_clipping: The maximum magnitude we allow the target noise to have. It only applies to TD3 and JAD3.

* update_actor_targets_freq: The frequency at which we update the actor and the target networks in TD3 and JAD3.

* num_previous_transitions: Only applies to multi-agent methods working with continuous actions. Determine how many previous observation-action pairs we want to add as input to the actor and critic together with the current observation.

*critic_use_previous_transitions: By default, we only add previous transitions to the actor. We can set it to `True` to add them to the critic.

* add_last_action: If we set it to `True` in multi-agent methods working with recurrent layers, we add the previous action as input together with the observation.

* update_actor_with_joint_qs: Only applies to JAD3, and by setting it to `True`, we update the actor with the joint action-value function instead of the individual ones.

* use_min_to_update_actor: Only applies to JAD3, and by setting it to `True`, we update the actor by computing the minimum between the two approximations of the action-value functions, as SAC does.

* lr: Learning rate used in the Adam optimizer to update the networks in value-based and actor-critic methods (only applies to the actor).
  
* critic_lr: Learning rate used in the Adam optimizer to update the actor-critic methods' critic.

* grad_norm_clip_actor: Maximum norm we allow the gradients to have before clipping them. We use it when updating the networks in value-based actor-critic methods (only applies to the actor).

* grad_norm_clip_critic: Like the previous one, but used when updating the critic in actor-critic methods.

* learner & controller & critic_controller: Determine which Python class they use to explore the environment and update the networks. They are bound to each method, so do not change them.
  
* agent: Determines which type of network to use, according to the `__init__` inside modules/agents. We do not recommend changing the layer type it uses (i.e. MLP or RNN), but you can change how many networks we train. It only applies to value-based methods and to the actor in actor-critic ones.

* critic: Determines which critic network architecture to use, according to the `__init__` inside modules/critic. Again, we do not recommend changing the layer type.
  
* mixer: Determines which mixer to use when factorizing the joint action-value function. Currently, we have only implemented QMIX and VDN decompositions.

* mixing_embed_dim: The number of units per layer in the mixer network. All the layers use the same number.
  
* hypernet_layers: The number of hypernetwork layers to compute the weights of the mixer network.
  
* hypernet_embed: The number of units in each layer of each hypernetwork.
  
* hidden_dim: Number of units to use in the actors, critics, and action-value networks used in value-based methods.
  
* hidden_dim_critic: Only applies to MADDPG and sets the number of units in the centralized critic.

* target_update_mode: If we set it to `soft`, we update the target networks using the polyak average. If we fix it to `hard`, we update those networks at the frequency specified by `hard_update_interval`.

* target_update_tau: The weight used in polyak average (i.e. the rho), usually close to 0.

* hard_update_interval: The frequency, expressed in network updates, at which we the target networks by copying the parameters of the non-target ones (i.e. hard update).

* actions_regularization: In some methods, when we compute the actor loss, we allow to add a regularization term to penalize too large actions. We can set it to `True` to indicate we want to use that term