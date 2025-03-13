#from game_manager import GameManager

#if __name__ == "__main__":
 #   manager = GameManager()
  #  manager.run()


import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from collections import deque
import random
from FlappyBirdEnv import FlappyBirdEnv  # Twoja klasa środowiska
from model import FlappyBirdModel            # Twój model DQN
import pygame

device = "cuda" if torch.cuda.is_available() else "cpu"

env = FlappyBirdEnv()

modelv1 = FlappyBirdModel().to(device)
optimizer = optim.Adam(modelv1.parameters(), lr=0.01)
memory  = deque(maxlen=1000000)

episodes = 2000
batch_size = 32
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01


for episode in range(episodes):
    state, _ = env.reset()
    state=torch.tensor(state).unsqueeze(0).unsqueeze(0).float().to(device)
    done = False
    total_reward = 0

    while not done:
        # wybór akcji (epsilon-greedy)
        if np.random.random() < epsilon:
            action = random.randint(0, 1)
        else:
            with torch.no_grad():
                q_values = modelv1(state)
                action = q_values.argmax().item()

        # Wykonanie akcji w środowisku
        next_state, reward, done, truncated, _ = env.step(action)
        next_state_tensor = torch.tensor(next_state).unsqueeze(0).unsqueeze(0).float().to(device)

        # Zapisz doświadczenie
        memory.append((state, action, reward, next_state_tensor, done))

        state = next_state_tensor
        if done or truncated:
            state, _ = env.reset()
            state = torch.tensor(state).unsqueeze(0).unsqueeze(0).float().to(device)

        # Trening sieci
        if len(memory) > batch_size:
            batch = random.sample(memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.cat(states).to(device)
            actions = torch.tensor(actions).to(device)
            rewards = torch.tensor(rewards).to(device)
            next_states = torch.cat(next_states).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).to(device)

            current_q = modelv1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            max_next_q = modelv1(next_states).max(1)[0]
            expected_q = rewards + (1 - dones) * gamma * max_next_q

            loss = nn.MSELoss()(current_q, expected_q.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        env.render()
    # Zmniejszanie epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Informacje o postępie
    if episode % 50 == 0:
        print(f"Episode {episode}, epsilon: {epsilon:.3f}")

torch.save(modelv1.state_dict(), "flappy_bird_model.pth")

