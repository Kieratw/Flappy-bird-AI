import gymnasium as gym
import numpy as np
import cv2
import pygame

from game import Game

class FlappyBirdEnv():
    def __init__(self, width=800, height=600):
        self.game=Game(width, height)
        self.action_space=gym.spaces.Discrete(2)
        self.observation_space=gym.spaces.Box(low=0, high=1,shape=(84,84),dtype=np.float32)

    def get_frame(self):
        frame = pygame.surfarray.array3d(self.game.screen)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame,(84,84))
        frame = frame/255.0
        return frame.astype(np.float32)

    def reset(self,seed=None):
        self.game.reset()
        state=self.get_frame()
        return state , {}

    def step(self,action):
        if action == 1:
            self.game.bird.jump()

        done ,score=self.game.update()

        self.game.draw()

        reward=0.1
        if done:
            reward = -100

        truncated=False
        state=self.get_frame()
        return state, reward, done,truncated, {}

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.game.draw()
        pygame.display.flip()
        self.game.clock.tick(60)

