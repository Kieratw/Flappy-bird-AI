import numpy as np
import pygame
from FlappyBirdEnv import FlappyBirdEnv
from bird import Bird
from model import BirdModel as NeuralBird

def run_episode(env):
    """
    Uruchamia epizod w środowisku, w którym symulowane są wszystkie ptaki.
    Zwraca: wektor całkowitych nagród dla każdego ptaka.
    """
    total_rewards = np.zeros(len(env.game.birds))
    done = False
    while not done:
        states = env.get_state()  # Stan dla każdego ptaka
        actions = []
        for i, bird in enumerate(env.game.birds):
            if getattr(bird, 'dead', False):
                actions.append(0)
            else:
                actions.append(bird.get_action(states[i]))
        state, rewards, done, _, _ = env.step(actions)
        total_rewards += rewards
        if env.render_mode:
            env.render()
            pygame.display.flip()
            env.game.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
    return total_rewards

if __name__ == "__main__":
    pygame.init()
    num_birds = 50
    generations = 50
    mutation_rate = 0.2
    mutation_std = 0.2

    # Inicjalizacja populacji – każdy ptak to instancja sieci neuronowej
    population = []
    for _ in range(num_birds):
        brain = NeuralBird()  # Tworzymy model sieci
        agent = Bird(400, 100, model=brain)  # Tworzymy obiekt ptaka z fizyką i grafiką, do którego dołączamy model
        population.append(agent)

    best_bird = None
    best_score = -np.inf

    for gen in range(1, generations + 1):
        # Tworzymy środowisko (bez renderowania dla szybkości treningu)
        env = FlappyBirdEnv(width=800, height=600, render_mode=True, num_birds=num_birds)
        env.reset(birds=population)
        total_rewards = run_episode(env)
        env.close()

        gen_best_idx = int(np.argmax(total_rewards))
        gen_best_score = total_rewards[gen_best_idx]
        gen_avg_score = np.mean(total_rewards)
        print(f"Generacja {gen}: najlepszy wynik = {gen_best_score:.2f}, średni wynik = {gen_avg_score:.2f}")
        print(type(population[gen_best_idx]))
        print(hasattr(population[gen_best_idx], 'brain'))
        print(type(population[gen_best_idx].brain) if hasattr(population[gen_best_idx],
                                                              'brain') else 'No brain attribute')
        if gen_best_score > best_score:
            best_score = gen_best_score
            best_bird = Bird(400, 100, model=NeuralBird())
            best_bird.brain.load_state_dict(population[gen_best_idx].brain.state_dict())

        # Selekcja: zachowujemy dwóch najlepszych, a resztę uzupełniamy mutowanymi kopiami
        sorted_indices = np.argsort(total_rewards)[::-1]
        new_population = []
        new_population.append(population[sorted_indices[0]])
        if len(sorted_indices) > 1:
            new_population.append(population[sorted_indices[1]])
        while len(new_population) < num_birds:
            parent_idx = sorted_indices[0] if np.random.rand() < 0.5 else sorted_indices[1]
            parent = population[parent_idx]
            child = Bird(400, 100, model=NeuralBird())
            # Poprawione wywołanie:
            child.brain.load_state_dict(parent.brain.state_dict())
            child.mutate(mutation_rate=mutation_rate, std=mutation_std)
            new_population.append(child)
        population = new_population

    print(f"\nNajlepszy wynik osiągnięty przez sieć: {best_score:.2f} punktów")

    # Demonstracja – uruchamiamy środowisko z renderowaniem i pojedynczym (najlepszym) ptakiem
    demo_env = FlappyBirdEnv(width=800, height=600, render_mode=True, num_birds=1)
    demo_env.reset(birds=[best_bird])
    done = False
    print("Uruchamianie demonstracji... (zamknij okno, aby zakończyć)")
    while not done:
        state = demo_env.get_state()[0]  # tylko jeden ptak
        action = best_bird.get_action(state)
        state, reward, done, _, _ = demo_env.step([action])
        demo_env.render()
        pygame.display.flip()
        demo_env.game.clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
    demo_env.close()
    pygame.quit()
