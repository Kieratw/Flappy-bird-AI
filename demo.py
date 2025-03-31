import pygame
import torch
from FlappyBirdEnv import FlappyBirdEnv
from bird import Bird
from model import BirdModel as NeuralBird
def run_demo_loop():
    pygame.init()

    best_bird = Bird(400, 100, model=NeuralBird())
    best_bird.brain.load_state_dict(torch.load("best_model.pt"))
    demo_env = FlappyBirdEnv(width=800, height=600, render_mode=True, num_birds=1)
    demo_env.reset(birds=[best_bird])
    done = False
    print("Best Bird")

    while not done:
        state = demo_env.get_state()[0]
        action = best_bird.get_action(state)
        _, _, done, _, _ = demo_env.step([action])
        demo_env.render()
        pygame.display.flip()
        demo_env.game.clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

    demo_env.close()
    pygame.quit()
