import numpy as np
import gymnasium
from gymnasium import spaces
from game import Game
from bird import Bird
from pipe import Pipe

class FlappyBirdEnv(gymnasium.Env):
    def __init__(self, width=800, height=600, render_mode=True, num_birds=1):
        super(FlappyBirdEnv, self).__init__()
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.num_birds = num_birds
        # Tworzymy domyślną populację, jeśli nie zostanie podana przez reset()
        birds = [Bird(400, 100) for _ in range(num_birds)]
        self.game = Game(width, height, birds, render_mode=render_mode)
        self.prev_scores = [0] * num_birds

        # Akcje – dla każdego ptaka (0: brak skoku, 1: skok)
        self.action_space = spaces.MultiDiscrete([2] * num_birds)
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_birds, 4), dtype=np.float32)

    def reset(self, seed=None, options=None, birds=None):
        if birds is None:
            birds = self.game.birds
        self.game.reset(birds)
        self.prev_scores = [0] * len(birds)
        state = self.get_state()
        return state, {}

    def step(self, actions):
        if not isinstance(actions, (list, np.ndarray)):
            actions = [actions]
        for i, action in enumerate(actions):
            if action == 1 and not self.game.birds[i].dead:
                self.game.birds[i].jump()
        done, _ = self.game.update()
        rewards = []
        for i, bird in enumerate(self.game.birds):
            if bird.dead:
                reward = -30
            else:
                reward = 10 if bird.score > self.prev_scores[i] else 0.1
            self.prev_scores[i] = bird.score
            rewards.append(reward)
        state = self.get_state()
        truncated = False
        info = {}
        done = all(bird.dead for bird in self.game.birds)
        return state, np.array(rewards, dtype=np.float32), done, truncated, info

    def get_state(self):
        states = []
        for bird in self.game.birds:
            if bird.dead:
                states.append([0, 0, 0, 0])
            else:
                bird_y = bird.y / self.height
                bird_velocity = bird.velocity / Bird.max_fall_speed
                nearest_pipe_distance_x = self.width
                nearest_pipe_distance_y = 0
                for pipe in self.game.pipes:
                    pipe_end_x = pipe.x + Pipe.width
                    if pipe_end_x - bird.x >= 0 and (pipe_end_x - bird.x) < nearest_pipe_distance_x:
                        nearest_pipe_distance_x = pipe_end_x - bird.x
                        nearest_pipe_distance_y = pipe.gap_y - bird.y
                nearest_pipe_distance_x /= self.width
                nearest_pipe_distance_y /= self.height
                states.append([nearest_pipe_distance_x, nearest_pipe_distance_y, bird_y, bird_velocity])
        return np.array(states, dtype=np.float32)

    def render(self):
        if self.render_mode:
            self.game.draw()

    def close(self):
        self.game.close()
