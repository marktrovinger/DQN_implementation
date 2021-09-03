import gym
from gym.wrappers import AtariPreprocessing
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback

# create a list of environments to test
envs = []

# config, paper says 10M timesteps, testing on 1M
config = {
    "policy_type":"CnnPolicy",
    "total_timesteps":1000000,
    "env_name":"Pong-v0",
}

# run configuration
run = wandb.init(
    project="dqn-project",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True
)


def make_env():
    """
    A function for creating a gym environment and registering
    that environment
    :env - gym environment
    """
    env = gym.make(config["env_name"])
    env = Monitor(env)
    # gym has a nice module for preprocessing Atari images to the specification of
    # the Mnih paper, however Pong-v0 has built in frame skip, so we need to handle it
    # a different way
    if config["env_name"] != "Pong-v0":
        env = AtariPreprocessing(env, grayscale_obs=True)
    return env

env = DummyVecEnv([make_env])
env = VecVideoRecorder(env, f"videos/{run.id}", record_video_trigger=lambda x: x % 2000 == 0,
                       video_length=200)

# the default policy for CNN doesn't use quite the same settings as the paper,
# so we create a cnnpolicy

   #cnn_paper_policy = DQN.cnnpolicy(env.observation_space, env.action_space, optimizer_class=torch.# optim.RMSProp(lr=0.00025, ))
# matching the original paper will likely require some fiddling

model = DQN(config['policy_type'], env, verbose=1,
            tensorboard_log=f"runs/{run.id}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,

    ),

)

