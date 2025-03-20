import torch
import multiprocessing as mp
from bird import Bird
from FlappyBirdEnv import FlappyBirdEnv
from model import BirdModel as NeuralBird
def simulate_bird(args):
    state_dict, env_params = args
    env = FlappyBirdEnv(**env_params, render_mode=False)
    bird = NeuralBird()  # używamy wersji z siecią
    bird.model.load_state_dict(state_dict)
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = bird.get_action(state)
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state
    env.close()
    return total_reward