import os
import numpy as np
import pygame
import torch
from FlappyBirdEnv import FlappyBirdEnv
from bird import Bird
from model import BirdModel as NeuralBird

def run_episode(env):
    """
    Uruchamia epizod w środowisku, w którym symulowane są wszystkie ptaki.
    Zwraca: wektor całkowitych nagród dla każdego ptaka oraz maksymalną liczbę żywych ptaków, jaka wystąpiła.
    """
    total_rewards = np.zeros(len(env.game.birds))
    max_alive = 0
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
        # Liczba żywych ptaków w tej klatce
        current_alive = sum(1 for bird in env.game.birds if not bird.dead)
        if current_alive > max_alive:
            max_alive = current_alive
        if env.render_mode:
            env.render()
            pygame.display.flip()
            env.game.clock.tick(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
    return total_rewards, max_alive

if __name__ == "__main__":
    pygame.init()
    num_birds = 20
    generations = 50
    mutation_rate = 0.1
    mutation_std = 0.1

    # Jeśli istnieje zapisany model, pytamy czy chcesz trenować dalej, czy tylko zademonstrować.
    if os.path.exists("best_model.pt"):
        mode = input("Wykryto zapisany model 'best_model.pt'. Wybierz tryb: [T]renowanie dalej, [D]emonstracja: ").strip().lower()
        if mode == "d":
            print("Ładowanie modelu i uruchamianie demonstracji...")
            best_bird = Bird(400, 100, model=NeuralBird())
            best_bird.brain.load_state_dict(torch.load("best_model.pt"))
            demo_env = FlappyBirdEnv(width=800, height=600, render_mode=True, num_birds=1)
            demo_env.reset(birds=[best_bird])
            done = False
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
            exit()
        elif mode == "t":
            print("Kontynuowanie treningu z wykorzystaniem zapisanego modelu...")
            # Tworzymy początkową populację – ustawiamy jednego ptaka z najlepszym modelem,
            # a reszta to nowe instancje (mogą być również zainicjalizowane tym modelem, jeśli chcesz).
            best_bird = Bird(400, 100, model=NeuralBird())
            best_bird.brain.load_state_dict(torch.load("best_model.pt"))
            population = [best_bird]
            for _ in range(num_birds - 1):
                brain = NeuralBird()
                agent = Bird(400, 100, model=brain)
                population.append(agent)
        else:
            print("Nieprawidłowy wybór. Uruchamiam demonstrację domyślnie.")
            best_bird = Bird(400, 100, model=NeuralBird())
            best_bird.brain.load_state_dict(torch.load("best_model.pt"))
            demo_env = FlappyBirdEnv(width=800, height=600, render_mode=True, num_birds=1)
            demo_env.reset(birds=[best_bird])
            done = False
            while not done:
                state = demo_env.get_state()[0]
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
            exit()
    else:
        # Jeśli nie ma zapisanego modelu, inicjujemy nową populację
        population = []
        for _ in range(num_birds):
            brain = NeuralBird()  # Tworzymy model sieci
            agent = Bird(400, 100, model=brain)  # Tworzymy obiekt ptaka z fizyką i grafiką, do którego dołączamy model
            population.append(agent)

    # Trening – niezależnie czy kontynuujemy czy zaczynamy od nowa
    best_bird = None
    best_score = -np.inf

    for gen in range(1, generations + 1):
        # Tworzymy środowisko (render_mode=True, żeby widzieć postęp treningu)
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
            # Zapisz najlepszy model do pliku – można później załadować i kontynuować trening
            torch.save(best_bird.brain.state_dict(), "best_model.pt")
            print("Zapisano nowy najlepszy model do pliku 'best_model.pt'.")

        # Selekcja: zachowujemy dwóch najlepszych, a resztę uzupełniamy krzyżowaniem lub mutacją
        sorted_indices = np.argsort(total_rewards)[::-1]
        new_population = []
        new_population.append(population[sorted_indices[0]])
        if len(sorted_indices) > 1:
            new_population.append(population[sorted_indices[1]])
        while len(new_population) < num_birds:
            parent_idx = sorted_indices[0] if np.random.rand() < 0.5 else sorted_indices[1]
            parent = population[parent_idx]
            child = Bird(400, 100, model=NeuralBird())
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
