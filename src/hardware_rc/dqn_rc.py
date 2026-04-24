#!/usr/bin/env python3

"""
Filename: DQN_RC.py
Author: Andrew Carr
Description: 
    A class that implements PPO for a MEMS-based reservoir computer for 
    discrete reinforcement learning tasks.

    MEMS dynamics simulation is done with a rk4 DDE solver implemented 
    with JAX for speed.

    Objects are created with, optionally, a list of hyperparameters,
    environment from Farama Gymnasium, previously trained models, and 
    singular or specific hyperparameter values.
"""
import os, multiprocessing as mp

OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
# from matplotlib.colormap import get_cmap
from IPython import display
import gymnasium as gym
from collections import deque


from dataclasses import dataclass, asdict, replace, field
from typing import Any, Mapping, Optional, List
from datetime import datetime
import wandb
import time
import json
import os

import tempfile


from ray.air import session

from ray.train import Checkpoint
from ray.air.integrations.wandb import WandbLoggerCallback, wandb
from datetime import datetime

import jax
import jax.numpy as jnp
from functools import partial

from .reservoir import Reservoir



@dataclass(frozen=True)
class DQNConfig:
    # ---- model/solver ---
    # DEFAULT HYPERPARAMETERS

    rc_seed: int = 1
    mask_seed: int = 1
    weight_seed: int = 1
    env_seed: int = 1
    general_seed: int = 0
    N: int = 200
    bufferLength: int = 30*N
    learning_rate: float = 1e-4
    learning_rate_decay: float = 1
    learning_rate_min: float = 1e-5
    epsilon: float = .01
    epsilon_min: float = 0.01
    epsilon_decay: float = 1
    gamma: float = 0.97
    beta: float = 100
    vel_weight: float = .5
    rewardNormalizationFactor: float = 1
    TargetUpdateRate: int = 4
    batch_size: int = 16
    theta: float = 0.8
    T: float = 6*np.pi
    h: float = 0.2
    SampleDelay: int = 3
    InputConnectivity: float = 0.2
    NormalizationFactor: List[float] = field(default_factory=lambda: [4.8, 3, 0.418, 2])
    NormalizationOffset: List[float] = field(default_factory=lambda: [4.8, 4, 0.5, 4])
    amplification: int = 9
    tau: int = 75
    fb_gain: float = 0.25
    tau_N: float = 0.0  # if non-zero, tau is set to tau_N * N
    trials: int = 10
    val_size: int = 10


    def validate(self) -> None:
        if self.N <= 0:
            raise ValueError(f"N must be > 0: {self.N}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be > 0: {self.batch_size}")
        if self.epsilon < self.epsilon_min:
            raise ValueError("epsilon must be >= epsilon_min")
        if self.epsilon_decay < 0:
            raise ValueError(f"epsilon_decay must be >= 0: {self.epsilon_decay}")
        if self.gamma < 0:
            raise ValueError(f"gamma must be >= 0: {self.gamma}")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DQNConfig":
        # ignore unknown keys gracefully
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        cfg = cls(**known)  # type: ignore[arg-type]
        cfg.validate()
        return cfg

    def updated(self, **overrides: Any) -> "DQNConfig":
        # ignore unknown override keys as well
        known_overrides = {k: v for k, v in overrides.items()
                           if k in self.__dataclass_fields__}
        cfg = replace(self, **known_overrides)
        cfg.validate()
        return cfg

class DQN_RC:
    def __init__(self, env, reward_function=None,
                 config: Optional[DQNConfig | Mapping[str, Any]] = None,
                 model: Optional[str] = None,
                 **overrides: Any) -> None:
        """
        A class that implements a DQN Reservoir Computer for reinforcement learning tasks.

        Args:
            env (gym.Env): The environment to train the model on.
            DQNConfig (dict): Hyparameter values
            Mapping (dict): Hyperparameter values
            overrides (dict): Individual hyperparameter values

        Attributes:
            rand (np.random.RandomState): Random number generator with seed.
            N (int): Number of internal nodes in the reservoir.
            env (gym.Env): The environment to train the model on.
            bufferLength (int): Length of the memory buffer.
            memory (deque): Memory buffer for experience replay.
            learning_rate (float): Learning rate for training.
            learning_rate_decay (float): Learning rate decay after each episode.
            learning_rate_min (float): Minimum learning rate.
            epsilon (float): Initial epsilon value for epsilon-greedy policy.
            epsilon_min (float): Minimum epsilon value.
            epsilon_decay (float): Decay rate for epsilon.
            gamma (float): Discount factor for future rewards.
            rewardNormalizationFactor (float): Normalization factor for reward.
            TargetUpdateRate (int): When to update the target method.
            ClearWhenSampled (bool): Clears memory after each training if True.
            batch_size (int): Batch size for training.
            updateCounter (int): Counter of episodes for target model updating.
            state_shape (tuple): Shape of the gym state space.
            loss (float): Loss value, squared difference between target and predicted Q-values.

            theta (float): Length of one sample of reservoir state [non-dimensional].
            T (float): Total simulation time for one reservoir state [non-dimensional].
            h (float): Integration step size.
            tau (float): unused as of 9/28/2025.
            T_final (float): Total simulation time for one reservoir state [non-dimensional].
            SampleDelay (int): How long to wait after input before starting to sample the reservoir state (in number of h steps).
            MEMS_IC (np.ndarray): MEMS initial conditions (position, velocity).
            time_array (np.ndarray): Array of simulation time values.
            MEMS_state (np.ndarray): Current position and velocity of beam.
            NormalizationFactor (np.ndarray): Normalization divisor for input.
            NormalizationOffset (np.ndarray): Normalization offset for input.
            W_internal (np.ndarray): Internal weights of the reservoir.
            spectralRadius (float): Spectral radius of the internal weights.
            spectralRadius_after_mod (float): Spectral radius after normalization.
            spectral_radius_scalar (float): Scalar for spectral radius normalization.
            W_in (np.ndarray): Input weights of the reservoir.
            X_int (np.ndarray): unused as of 9/28/2025.
            Vin (np.ndarray): Input voltage sequence.
            W_out (np.ndarray): Output weights for the action space.
            W_out_target (np.ndarray): Target output weights for the action space.
            MEMS_neurons (np.ndarray): Neuron values based on input to MEMS simulation.

        """
        # start from defaults
        base = DQNConfig()
        # merge user config (dataclass or dict)
        if isinstance(config, DQNConfig):
            base = config
        elif isinstance(config, Mapping):
            base = DQNConfig.from_dict(config)
        elif config is not None:
            raise TypeError("config must be a DQNConfig, dict-like, or None")
        # apply overrides (highest priority)
        self.config = base.updated(**overrides)

        # future generalization addition
        # if self.config.NormalizationFactor is None and env.__class__.__name__ == '':
        #   self.config.NormalizationFactor

        # ----------------------------------------------------------------------------------------
        ### RL parameters
        if self.config.general_seed != 0:
            self.rc_seed = self.config.general_seed
            self.mask_seed = self.config.general_seed
            self.env_seed = self.config.general_seed
            self.weight_seed = self.config.general_seed
        else:
            self.rc_seed = self.config.rc_seed
            self.mask_seed = self.config.mask_seed
            self.env_seed = self.config.env_seed
            self.weight_seed = self.config.weight_seed
        self.rand = np.random.RandomState(self.rc_seed)
        self.N = self.config.N  # N^2 = number of internal nodes in the reservoir
        self.env = env  # ale environment
        self.test_env = None
        self.bufferLength = self.config.bufferLength  # memory length, not sure why N is used -> in progress learning
        self.memory = deque(maxlen=self.bufferLength) # memory buffer in form of a queue for experience replay
        self.trials = self.config.trials
        self.val_size = self.config.val_size

        self.learning_rate = self.config.learning_rate  # learning rate of training
        self.learning_rate_decay = self.config.learning_rate_decay  # learning rate decay after each episode, 1 means no decay
        self.learning_rate_min = self.config.learning_rate_min  # minimum learning rate

        self.epsilon = self.config.epsilon  # initial epsilon for epsilon-greedy policy
        self.epsilon_min = self.config.epsilon_min
        self.epsilon_decay = self.config.epsilon_decay
        self.gamma = self.config.gamma  # discount factor for future rewards

        self.rewardNormalizationFactor = self.config.rewardNormalizationFactor  # normalization factor for reward if needed
        self.TargetUpdateRate = self.config.TargetUpdateRate  # when to update the target method

        self.batch_size = self.config.batch_size  # batch size for training

        self.updateCounter = 0  # keep track of episodes for target model updating

        self.state_shape  = self.env.observation_space.shape
        if self.state_shape is None:
            print(self.env.observation_space)
            self.state_shape = [len(self.env.observation_space)]
            print(self.state_shape)
        try:
            self.action_shape = self.env.action_space.n
        except:
            self.action_shape = self.env.action_space.shape[0]
        print(f'action shape: {self.action_shape}')
            
        self.loss = 0 # Initialize loss to 0

        if reward_function is None:
            def r_func(*, reward, cur_state=None, new_state=None, action=None, done=None, hparams=None, validate=False):
                return reward
            self.reward_function = r_func
        else:
            self.reward_function = reward_function

        # ----------------------------------------------------------------------------------------
        ### MEMS RC parameters

        self.theta = self.config.theta  # length of one sample of reservoir state [non-dimensional]
        self.T = self.config.T  # total simulation time for one reservoir state [non-dimensional]

        self.h = self.config.h  # integration step size
        self.tau = self.config.tau
        self.fb_gain = self.config.fb_gain

        self.T_final = self.theta * self.N  # total simulation time for one reservoir state [non-dimensional]
        self.SampleDelay = self.config.SampleDelay # how long to wait before starting to sample the reservoir state (in number of h steps)

        self.MEMS_IC = np.array([0.0,0.0])  # MEMS initial conditions (position, velocity)
        
        if self.config.tau_N != 0:
            self.tau = int(self.config.tau_N * self.N) # craft tau based on N if specified
            print('setting tau based on N: ', self.tau)
        
        # hardcoding tau = N+1 for this experiment
        # self.tau = self.N + 1
        # print(self.tau)

        self.tau_steps = int(self.tau*jnp.floor(self.theta/self.h))
        m = self.tau_steps
        buf_len = max(m+2, 2)
        # Create tau buffers for feedback
        self.pos_buf = jnp.full((buf_len,), 0.0, dtype=jnp.float32)
        self.vel_buf = jnp.full((buf_len,), 0.0, dtype=jnp.float32)
        self.buf_idx = jnp.int32(0)
        self.buf_cnt = jnp.int32(0) # buffer count
        # start everything with IC: x(0) = 0, x_dot(0) = 0

        self.time_array = np.array([0])  # start with initial time of zero
        self.MEMS_state = np.array([[0.0, 0.0]])  # initialize the array of MEMS state array (stores the MEMS dynamical response)


        self.NormalizationFactor = self.config.NormalizationFactor
        self.NormalizationOffset = self.config.NormalizationOffset

        self.amplification = self.config.amplification

        # ----------------------------------------------------------------------------------------
        # Code Optimization Metrics
        self.mems_sim_times = np.array([])

        # ----------------------------------------------------------------------------------------
        # check if model is provided after defining all hyperparams

        if model is not None:
            print('transferring over model')
            root, extension = os.path.splitext(model)
            if extension.lower() != '.npz':
              raise TypeError('File extension must be .npz')
            self.load_reservoir(model)
            self.W_out_target = self.W_out.copy()
            return

        # ----------------------------------------------------------------------------------------
        # input weight matrix aka mask

        self.reservoir = Reservoir(h=self.h,
                              theta=self.theta,
                              N=self.N,
                              tau=self.tau,
                              fb_gain=self.fb_gain,
                              sd=self.SampleDelay,
                              amp=self.amplification,
                              norm_factor=self.NormalizationFactor,
                              norm_offset=self.NormalizationOffset,
                              state_shape=self.state_shape,
                              input_connectivity=self.config.InputConnectivity,
                              mask=True,
                              mask_seed=self.mask_seed,
                              normalize_mask=True
                              )

        weight_rand = np.random.RandomState(self.weight_seed)
        scale = 1/np.sqrt(self.N)
        self.W_out = weight_rand.normal(0, scale, size=(self.N+1+self.state_shape[0], self.action_shape))
        
        self.W_out_target = self.W_out.copy()
    
    def zero_reservoir(self):
        self.reservoir.zero_reservoir()

    
    def readout(self, neurons, W_out, *, analyze=False):
        high_Q = float('-inf')
        Q_vals = np.array([])
        for i in range(W_out.shape[1]):
            Q_val = np.dot(neurons, W_out[:,i])
            if analyze:
                Q_vals = np.append(Q_vals, Q_val)
            if Q_val > high_Q:
                high_Q = Q_val
                action = i
        if analyze:
            return high_Q, action, Q_vals.tolist()
        else:
            return high_Q, action, None

    def optimize(self, target, Q_current, neurons, W_old):       #There will be a different readout circuit training for each action
      """
      Run one step of gradient descent to update the readout weights.

      Args:
          target (float): Target Q-value.
          Q_current (float): Current Q-value prediction.
         neurons (np.ndarray): Reservoir neurons state including input feedback.
          W_old (np.ndarray): Current readout weights to be updated.
      Returns:
          W_updated (np.ndarray): Updated readout weights after one step of gradient descent.
          loss (float): The squared difference between target and predicted Q-value.
      """
      loss = np.square(target - Q_current)

      # Gradient = (-2) * np.multiply(np.transpose(Z), (target - Q_current)).squeeze()
      Gradient = -2 * (target - Q_current) *neurons.flatten()
      W_updated = W_old - self.learning_rate*Gradient
      # print(f'max_w_old: {np.max(W_old)}, max_w_new: {np.max(W_updated)}, Gradient: {np.max(Gradient)}, learn_rate: {self.learning_rate}')
      return W_updated, loss

    def act(self, state, opt=False, *, analyze=False):
        """
        Run the reservoir computer to get action and apply epsilon-greedy policy.


        Args:
            state (np.ndarray): Current state of the environment.

        Returns:
            action (int): Chosen action.
           neurons (np.ndarray): Reservoir neurons state including input feedback.
        """
        if opt:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon)

        # Action_out = self.predictRC(state)
        time_start = time.time_ns()

        neuron_vals = self.reservoir.sim(obs=state, direct_fb=True)

        self.mems_sim_times = np.append(self.mems_sim_times, time.time_ns() - time_start) if opt else self.mems_sim_times

        self.MEMS_neurons = np.array(neuron_vals).reshape(1, -1)          # shape (N, 1)

        _, action, Q_vals = self.readout(self.MEMS_neurons, self.W_out, analyze=analyze)

        if opt:
            if self.rand.random() < self.epsilon:
                action = int(self.rand.randint(0,self.action_shape)) # using rc_seed instead of env
                return action, self.MEMS_neurons, Q_vals
                # return self.env.action_space.sample(), self.MEMS_neurons

        return action, self.MEMS_neurons, Q_vals

    def remember(self, action, reward, done, neurons, future_neurons):
        """
        Add experience to the memory buffer for experience replay-based training.
        """
        self.memory.append([action, reward, done, neurons, future_neurons])

    def replay(self):
        """
        Run training based on a batch of experiences from memory.

        1. Takes a random batch of experiences from memory.
        2. For each experience, runs target reservoir computer to get target Q-value.
        3. Trains the readout matrix (optimize) depending on what action the actual model took.
          - Gets actual Q-Values by multiplying neuron states,neurons, with readout weights.

        """

        self.loss = 0
        batch_loss = 0 # Initialize batch loss


        if (len(self.memory) < self.batch_size):
          return

        # changed the sampling technique to allow for the same random number generator to be used
        sample_indices = self.rand.choice(len(self.memory), self.batch_size, replace=False)
        samples = [self.memory[i] for i in sample_indices]

        for sample in samples:
            action, reward, done, neurons, future_neurons = sample
            # Do things related to the readout circuit of action 0
            if done:
                target = reward/self.rewardNormalizationFactor
                # print(f'target: {target}, normfactor: {self.rewardNormalizationFactor}')
            else:
                Q_future = self.readout(future_neurons, self.W_out_target)[0]
                target = (reward + Q_future * self.gamma)/self.rewardNormalizationFactor
                # print(f'target: {target}, normfactor: {self.rewardNormalizationFactor}')

            Q_current = np.dot(neurons, self.W_out[:,action])
            new_weights, loss = self.optimize(target, Q_current, neurons, self.W_out[:,action])
            self.W_out[:,action] = new_weights
            batch_loss += loss # Accumulate loss for the batch

        # Decay learning rate
        self.learning_rate *= self.learning_rate_decay
        self.learning_rate = max(self.learning_rate, self.learning_rate_min)      #Annealing


        # Update target network every few episodes
        self.updateCounter = self.updateCounter+1


        if(self.updateCounter%self.TargetUpdateRate == 0):
          self.W_out_target = self.W_out.copy()


        average_batch_loss = batch_loss / self.batch_size # Calculate average batch loss
        self.loss = average_batch_loss # Return average batch loss

    def play_env(self, obs, env, *, opt=False):

        done = False
        total_reward = 0
        trial_length = 0
        trial_loss = 0
        tot_neuron_sat = 0

        self.zero_reservoir()
        action, neurons, _= self.act(obs, opt)
        
        while not done:

            new_state, reward, terminated, truncated, info = env.step(action)
            
            done = terminated or truncated
            # ------------------------------------------------------------------
            # Reward Shaping (if necessary)
            # reward = reward + 1
            if opt:
                reward = self.reward_function(reward=reward, cur_state=obs, new_state=new_state, action=action, done=done, hparams=self.config)
            
            obs = new_state
            # ------------------------------------------------------------------
            future_action, future_neurons, _ = self.act(new_state, opt)
            
            if opt:
                self.remember(action, reward, done, neurons, future_neurons)
                self.replay()
                trial_loss += self.loss

            neurons = future_neurons
            action = future_action
            total_reward += reward
            tot_neuron_sat += np.sum(np.abs(neurons)>=0.89)*100/neurons.size
            trial_length += 1
                
        returns = {
            "total_reward": total_reward,
            "trial_length": trial_length,
            "trial_loss": trial_loss if opt else None,
            "avg_neuron_sat": tot_neuron_sat/trial_length
        }

        return returns


    def train(self, *, env=None, meta, trials=None, 
              folder_path=None, wandb_on=True, save_model=True,
              ray_on=False, ray_metric='avg_reward', ray_mode='max',
              ray_report_freq=1, full_validate=True, val_size_min=1, threed_it=False):
        self.env_name = meta['env'] if 'env' in meta else 'unknown_env'
        if env is not None:
            self.env = env
        if trials is None:
            trials = self.trials

        required_fields = ["project", "group", "job_type", "run_name", "tags"]
        for field in required_fields:
            if field not in meta:
                meta[field] = "not_provided"
        self.run_name = meta["run_name"]
        if folder_path is not None:
            self.folder_path = folder_path
        elif save_model:
            raise ValueError("A folder path is required for saving models during training")
        
        if ray_on and ray_metric is None:
            ray_metric = 'avg_reward'

        # setup wandb
        if wandb_on:
            run = wandb.init(
                project=meta["project"],
                group=meta["group"],
                job_type=meta["job_type"],
                name=meta["run_name"],
                tags=meta['tags'],
                config={"hp": self.config, "meta": meta},
            )

        eval_metric = float('-inf')
        if ray_on and ray_mode == 'min':
            eval_metric = float('inf')
        reward_metric = float('-inf')
        length_metric = float('inf')

        trial_times = np.array([])
        if threed_it:
            images =[]

        best_avg_reward = float('-inf')
        
        obs = self.env.reset(seed=self.env_seed)[0]

        for trial in range(trials):
            if trial != 0:
                obs = self.env.reset()[0]
                

            self.mems_sim_times = np.array([])
            time_start = time.time_ns()

            returns = self.play_env(obs, self.env, opt=True)

            trial_loss = returns["trial_loss"]
            total_reward = returns["total_reward"]
            trial_length = returns["trial_length"]

            
            
            val_time_start = time.time_ns()
            if full_validate or ray_on:
                val_results = self.validate()
                if val_results['avg_reward'] >= best_avg_reward or threed_it:
                    if not ray_on:
                        print(f'New best reward: {val_results["avg_reward"]}')
                    best_avg_reward = val_results['avg_reward']
                    if save_model and not ray_on:
                        folder_path = folder_path if folder_path is not None else f"DQN_RC/models/{meta['env']},{meta['group']}"
                        os.makedirs(folder_path, exist_ok=True)
                        best_model_path = f"{folder_path}/{meta['run_name']}.npz"
                        self.save_reservoir(best_model_path)
            elif total_reward > .9*best_avg_reward or total_reward > 1.1*best_avg_reward: # validate fully if reward is close to best
                print('sampling')
                val_results = self.validate()
                if val_results['avg_reward'] >= best_avg_reward :
                    print(f'New best reward: {val_results["avg_reward"]}')
                    best_avg_reward = val_results['avg_reward']
                    if save_model and not ray_on:
                        folder_path = folder_path if folder_path is not None else f"DQN_RC/models/{meta['env']},{meta['group']}"
                        os.makedirs(folder_path, exist_ok=True)
                        best_model_path = f"{folder_path}/{meta['run_name']}.npz"
                        self.save_reservoir(best_model_path)
            else: # if full_validation is false and not close to best, validate for val_size_min
                val_results = self.validate(val_size=val_size_min)

            val_time = time.time_ns() - val_time_start

            
            if threed_it:
                from analyze.analyze_run import AnalyzeRun
                ar = AnalyzeRun(best_model_path)
                img = ar.visualize_state_action(num_steps=10,
                                          ele_offset=-10,
                                          azim_offset=-90,
                                          folder_path=f'{folder_path}/3d_trials',
                                          file_name=f'3d_trial{trial}',
                                          show=False,
                                        #   save=False,
                                        #   to_bytes=True,
                                          title_addon=f'- Trial {trial}: reward={int(best_avg_reward)}')
                # images.append(img)
                # np.savez_compressed(f'{folder_path}/{meta["run_name"]}_images.npz', 
                #             images=images,)

            

            
            

            if ray_on:
                # if ray_mode is 'max', we want to maximize eval_metric, vice versa for 'min'
                if val_results[ray_metric] >= eval_metric and ray_mode == 'max' or val_results[ray_metric] <= eval_metric and ray_mode == 'min':
                    eval_metric = val_results[ray_metric]
                    with tempfile.TemporaryDirectory() as tmpdir:
                        # 2) save your artifact(s) into that folder
                        filename = f"reservoir_best-{meta['run_name']}.npz"
                        out_path = os.path.join(tmpdir, filename)
                        self.save_reservoir(path_npz=out_path)
                    
                        session.report({
                            'avg_reward': val_results['avg_reward'],
                            'avg_trial_length': val_results['avg_trial_length'],
                            'neuron_sat': val_results['avg_neuron_sat'],
                            'eval_metric': eval_metric,
                            'avg_trial_loss': trial_loss,
                            'eval_metric_cur': val_results[ray_metric],
                            'trial': trial,
                            'epsilon': self.epsilon,
                            'learning_rate': self.learning_rate,
                            'trial_length': trial_length,
                            
                        }, checkpoint=Checkpoint.from_directory(tmpdir))

                elif trial % ray_report_freq == 0:
                    session.report({
                        'avg_reward': val_results['avg_reward'],
                        'avg_trial_length': val_results['avg_trial_length'],
                        'neuron_sat': val_results['avg_neuron_sat'],
                        'eval_metric': eval_metric,
                        'avg_trial_loss': trial_loss,
                        'eval_metric_cur': val_results[ray_metric],
                        'trial': trial,
                        'epsilon': self.epsilon,
                        'learning_rate': self.learning_rate,
                        'trial_length': trial_length,
                    })
            
            trial_time = time.time_ns() - time_start
            trial_times = np.append(trial_times, trial_time/1_000_000_000) # should be in seconds

            if wandb_on:
                wandb.log({
                    "reward": total_reward,
                    "avg_loss_this_trial": trial_loss/trial_length,
                    "trial_length": trial_length,
                    "epsilon": self.epsilon,
                    "learning_rate": self.learning_rate,
                    "Neuron Saturation Percentage": val_results['avg_neuron_sat'],
                    "Average Trial Time per N [ms]": trial_time/trial_length/1000000/self.N,
                    'val_reward': val_results['avg_reward'],
                    'best_avg_reward': best_avg_reward,
                    'trial': trial,
                })
                    
            if not ray_on:
                print(f"Trial {trial+1}/{trials}, Trial Reward: {total_reward:.1f}, best_avg_reward: {best_avg_reward:.1f}, Trial Length: {trial_length}, Avg Loss: {np.max(trial_loss/trial_length):.2f}, Avg neuron sat: {val_results['avg_neuron_sat']:.1f} [%]")
                print(f"Trial time = {trial_time/1000000000:.2f} [s], MEMS sim time per node = {np.mean(self.mems_sim_times)/self.N/1000000:.3f} [ms], Validation time = {val_time/1000000000:.2f} [s]")

        hours = np.floor(np.sum(trial_times)/60/60)
        if not ray_on:
            print(f'done, trained for {int(np.sum(trial_times))} [s] or {int(hours)} hours and {np.sum(trial_times/60)-hours*60:.2f} minutes')
        if wandb_on and not ray_on:
            wandb.finish()
        if ray_on:
            return
        # elif threed_it:
        #     return image_path
        else:
            return best_model_path
    
    def validate(self, val_size=None):
        # if self.test_env is None:
        # print(self.env_name)
        if self.env_name == 'tron':
            from games.tron import TronEnv
            self.test_env = TronEnv()
            
        else:
            self.test_env = gym.make(f'{self.env.unwrapped.spec.id}')

        avg_neuron_sats = []
        rewards = []
        trial_lengths = []
        trial_losses = []

        self.zero_reservoir()

        if val_size is None:
            val_size = self.val_size
        
        start = 20
        end = start + val_size
        
        for i in range(start,end,1):
            obs = self.test_env.reset(seed=i)[0]
            returns = self.play_env(obs, self.test_env)

            rewards.append(returns["total_reward"])
            trial_lengths.append(returns["trial_length"])
            avg_neuron_sats.append(returns["avg_neuron_sat"])
        
        val_results = {
            'avg_reward': np.mean(rewards),
            'avg_trial_length': np.mean(trial_lengths),
            'avg_neuron_sat': np.mean(avg_neuron_sats),
        }

        return val_results

    def save_reservoir(self, path_npz):
        self.hparams = {
            'algorithm': 'DQN_RC',
            'environment': self.env.unwrapped.spec.id if self.env_name != 'tron' else 'tron',
        }
        self.hparams.update(asdict(self.config)) # Convert dataclass fields to dict and update hparams
        meta = json.dumps(self.hparams)
        np.savez_compressed(path_npz, 
                            reservoir=self.reservoir, 
                            W_out=self.W_out, 
                            meta=meta)

    def load_reservoir(self, path_npz):
        with np.load(path_npz, allow_pickle=True) as data:
            mask = data['mask'] if 'mask' in data else None
            self.reservoir = data['reservoir'].item() if 'reservoir' in data else None
            self.W_out = data['W_out']

            # Decode JSON string from the NPZ entry
            meta_dict = json.loads(data['meta'].item())

            # Set attributes dynamically
            for key, value in meta_dict.items():
                setattr(self, key, value)
            
            if self.reservoir is None:
                self.reservoir = Reservoir(h=self.h,
                              theta=self.theta,
                              N=self.N,
                              tau=self.tau,
                              fb_gain=self.fb_gain,
                              sd=self.SampleDelay,
                              amp=self.amplification,
                              norm_factor=self.NormalizationFactor,
                              norm_offset=self.NormalizationOffset,
                              state_shape=self.state_shape,
                              load=True, 
                              mask=mask)

            # Store as a Python dict
            self.hparams = meta_dict

    