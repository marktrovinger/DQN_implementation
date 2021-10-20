from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import A2C
from stable_baselines3 import DQN
import wandb
from wandb.integration.sb3 import WandbCallback

# There already exists an environment generator
# that will make and wrap atari environments correctly.
# Here we are also multi-worker training (n_envs=4 => 4 environments)

config = {
    "policy_type":"CnnPolicy",
    "total_timesteps":25000,
    "env_name":"PongNoFrameskip-v4"
}

run = wandb.init(
    project="dqn_project",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)

def make_env():
    env = make_atari_env('PongNoFrameskip-v4', n_envs=1, seed=0)
    env = Monitor(env)
    return env

env = make_env()
env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)

# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)

model = DQN('CnnPolicy', env, verbose=1, tensorboard_log=f'runs/{run.id}')
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)
run.finish()
