import os
import numpy as np
import pygame
import torch
from FlappyBirdEnv import FlappyBirdEnv
from bird import Bird
from model import BirdModel as NeuralBird

def run_training_loop():
    main()
def run_episode(env):
    total_rewards = np.zeros(len(env.game.birds))
    max_alive = 0
    done = False
    while not done:
        states = env.get_state()
        actions = []
        for i, bird in enumerate(env.game.birds):
            if getattr(bird, 'dead', False):
                actions.append(0)
            else:
                actions.append(bird.get_action(states[i]))
        _, rewards, done, _, _ = env.step(actions)
        total_rewards += rewards
        current_alive = sum(1 for bird in env.game.birds if not bird.dead)
        max_alive = max(max_alive, current_alive)

        if env.render_mode:
            env.render()
            pygame.display.flip()
            env.game.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
    return total_rewards, max_alive

def main():
    pygame.init()
    num_birds = 20
    generations = 20
    mutation_rate = 0.1
    mutation_std = 0.1

    if os.path.exists("best_model.pt"):
        print("Kontynuowanie treningu z wykorzystaniem zapisanego modelu...")
        best_bird = Bird(400, 100, model=NeuralBird())
        best_bird.brain.load_state_dict(torch.load("best_model.pt"))
        population = [best_bird]
        for _ in range(num_birds - 1):
            agent = Bird(400, 100, model=NeuralBird())
            population.append(agent)
    else:
        print("Rozpoczynanie nowego treningu...")
        population = []
        for _ in range(num_birds):
            agent = Bird(400, 100, model=NeuralBird())
            population.append(agent)

    best_bird = None
    best_score = -np.inf

    for gen in range(1, generations + 1):
        env = FlappyBirdEnv(width=800, height=600, render_mode=True, num_birds=num_birds)
        env.reset(birds=population)
        total_rewards, max_alive = run_episode(env)
        env.close()

        gen_best_idx = int(np.argmax(total_rewards))
        gen_best_score = total_rewards[gen_best_idx]
        gen_avg_score = np.mean(total_rewards)

        print(f"Generacja {gen}: najlepszy wynik = {gen_best_score:.2f}, średni wynik = {gen_avg_score:.2f}, maks. żywych ptaków = {max_alive}")

        if gen_best_score > best_score:
            best_score = gen_best_score
            best_bird = Bird(400, 100, model=NeuralBird())
            best_bird.brain.load_state_dict(population[gen_best_idx].brain.state_dict())
            torch.save(best_bird.brain.state_dict(), "best_model.pt")
            print("Zapisano nowy najlepszy model do pliku 'best_model.pt'.")

        # Selekcja + mutacja
        sorted_indices = np.argsort(total_rewards)[::-1]
        new_population = [population[sorted_indices[0]]]
        if len(sorted_indices) > 1:
            new_population.append(population[sorted_indices[1]])
        while len(new_population) < num_birds:
            parent_idx = sorted_indices[0 if np.random.rand() < 0.5 else 1]
            parent = population[parent_idx]
            child = Bird(400, 100, model=NeuralBird())
            child.brain.load_state_dict(parent.brain.state_dict())
            child.mutate(mutation_rate=mutation_rate, std=mutation_std)
            new_population.append(child)
        population = new_population

    print(f"\nNajlepszy wynik osiągnięty przez sieć: {best_score:.2f} punktów")
