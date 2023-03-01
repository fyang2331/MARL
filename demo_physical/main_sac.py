# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 16:10:43 2021

"""

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
import os
from rl_env import SeaBattleEnv
from stable_baselines3.common.results_plotter import load_results, ts2xy
import numpy as np
import time
import matplotlib.pyplot as plt
import torch


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level.
    """

    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 110:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                print(y[-5:])
                if self.verbose > 0:
                    print(f"Num timesteps: {self.num_timesteps}")
                    print(
                        f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward and y[-1:] >np.mean(y[-2:]):
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

        return True

# from typing import Callable
# def linear_schedule(initial_value:float) -> Callable[[float],float]:
#     def func(process_remaining:float) -> float:
#     	return initial_value*0.99**(1-process_remaining)/0.1
#     return func
def train_ddpg():
    np.random.seed(0)
    log_dir = f"model_ppo/"
    env = SeaBattleEnv.SeaBattleEnv("json_file/env.json", "json_file/dd_info.json")
    env.seed(0)
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    #env = VecNormalize(env, norm_obs=True, norm_reward=False,training= True)
    env = VecNormalize.load('model_ppo/env',env)
    n_actions = env.action_space.shape[-1]
    policy_kwargs = dict(
        activation_fn = torch.nn.ReLU,
        net_arch = dict(pi = [256,128],qf = [256,128])
    )
    action_noise = NormalActionNoise(mean=np.zeros(n_actions),sigma=0.3*np.ones(n_actions))
    model = SAC("MlpPolicy",
                env,
                policy_kwargs = policy_kwargs,
                verbose=1,
                learning_starts=20000,
                buffer_size=200000,
                tau = 0.01,
                train_freq=32,
                gradient_steps=32,
                batch_size = 8192,
                learning_rate= 1e-5,
                action_noise=action_noise,
                gamma = 0.9999,
                seed = 0,
                tensorboard_log='sac')
    model.set_parameters('model_ppo/best_model.zip')
    print(model.policy)
    print(model.learning_rate)
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
    model.learn(total_timesteps=int(5e6), callback=callback, log_interval=10)
    env.save('model_ppo/env')
    model.save('model_ppo/ppo')
    # with open('enegy_ddpg', 'wb') as file_pi:
    #     pickle.dump(env.envs[0].total_enegy, file_pi)
    # with open('finfish_ddpg', 'wb') as file_pi:
    #     pickle.dump(env.envs[0].finish, file_pi)


def test_ddpg():
    log_dir = f"model_ppo/best_model.zip"
    env = SeaBattleEnv.SeaBattleEnv("json_file/env.json", "json_file/dd_info.json")

    # env.render = True
    env = Monitor(env, log_dir)
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True,clip_obs=10.)
    # env = VecNormalize.load('model_ppo/env',env)
    env.training = False
    model = SAC.load(log_dir)
    model.set_env(env)
    # plot_results(f"model_ddpg/")
    # with open('enegy_ddpg', 'rb') as file_pi:
    #     result = pickle.load(file_pi)
    # with open('finfish_ddpg', 'rb') as file_pi:
    #     finish = pickle.load(file_pi)
    # plt.figure()
    # plt.plot(result)
    # plt.title('energy')
    # # plt.ylabel('Returns')
    # # plt.xlabel('time step')
    # plt.show()
    #
    # plt.figure()
    # plt.plot(finish)
    # plt.title('finish')
    # # plt.ylabel('Returns')
    # # plt.xlabel('time step')
    # plt.show()
    state = env.reset()
    r = 0
    while True:
        action = model.predict(state)
        next_state, reward, done, info = env.step(action[0])
        r += reward
        state = next_state
        time.sleep(0.0001)
        if done:
            env.reset()
            print('finish')
            print(r)
            r = 0
            break
            


def plot_results(log_folder):
    R = load_results(log_folder)['r']
    T = load_results(log_folder)['t']
    # _w = 7
    # _window_size = len(R) // _w if (len(R) // _w) % 2 != 0 else len(R) // _w + 1
    # filtered = savgol_filter(R, _window_size, 1)

    plt.title('smoothed returns')
    plt.ylabel('Returns')
    plt.xlabel('time step')
    plt.plot(T, R)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # log_dir = "tmp/"
    #
    # os.makedirs(log_dir, exist_ok=True)
    # train_ddpg()  # whether train
    test_ddpg()
