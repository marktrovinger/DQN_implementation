# Notes about PPO implementation

This document is a series of notes about PPO implementation, based on a YouTube video by Costa Huang, originally on the W&B channel.

## General

* 60 mins blitz - PyTorch
* Lil's Log - Policy Gradient Algorithms
* Argparse for experiment setup
* Torch deterministic for repro
* CUDA flag for GPU
* Poetry for package
* Tensorboard
* Vector env - multiple stacked envs
* RecordEpisodeStatistics wrapper
* SyncVectorEnv wrapper for Vector envs
* Categorical distribution from Torch
* Function for layer creation
* torch.nn.init.orthogonal for weight
* torch.nn.init.constant for bias
* Critic, 3 linear layers, Tanh() activation
* output layer uses std of 1
* input shape is product of observation space shape
* Actor is similar, init with std of 0.01
* Output is number of available actions
* Adam eps - 1e-5
* Numsteps - 128 per env, 128 x 4 = 512 data points for training
* logits - unnormalized action probabilities
* LR annealing - starts at 1->0 over the course of the training loop
* Step is sent to CPU, reward is sent to GPU, as is next_obs and next_done
* GAE - look up implementation
* Minibatch update
* Advantage norm - 
* Clipped policy - 
* Value loss policy - MSE, but with clipping? Look at this again
* Entropy loss - coefficient for entropy loss, value loss coefficient, minimize policy and value, max entropy loss
* Gradient clipping - max gradient norm, backprop
* KL divergence, John Schulman?
* Early stopping, SpinningUp, target KL variable, batch level or minibatch 

## Atari specific

* NoFrameSkip env
* Pixel env
* No Ops on env reset
* Skip 4 frames by default
* End of life as end of episode
* FireResetEnv - some envs will be stationary until first fire
* Clip Reward Env [-1, 1]
* Image preprocessing, use resize before grayscale, bug in resize
* FrameStack, 4 past obs
* CNN from DQN paper
* Create a shared network for feature extraction
* Scale image from [0,1]
* 10M timesteps
* Create report at W&B
