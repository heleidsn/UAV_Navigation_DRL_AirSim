import gym
import gym_env
import time

env = gym.make("airsim-env-v0")
# env.read_config('gym_airsim_multirotor/envs/config.ini')

env.reset()

step = 0
start_time = time.time()
x = 1  # displays the frame rate every 1 second
counter = 0

for i in range(500):
    action = [5, 0]
    obs, reward, done, info = env.step(action)
    step += 1

    counter += 1
    if (time.time() - start_time) > x:
        print("FPS: ", counter / (time.time() - start_time))
        counter = 0
        start_time = time.time()
