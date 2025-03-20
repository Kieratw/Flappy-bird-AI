import torch
import torch.nn as nn
import random


class BirdModel(nn.Module):
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()  # Inicjalizacja klasy bazowej nn.Module
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(4, 24),  # Warstwa wejściowa: 4 wejścia -> 24 neurony
            nn.ReLU(),
            nn.Linear(24, 24),  # Warstwa ukryta: 24 -> 24 neurony
            nn.ReLU(),
            nn.Linear(24, 2)  # Warstwa wyjściowa: 24 -> 2 neurony (akcje)
        ).to(self.device)
        self.score = 0

    def get_action(self, state):
        """
        Uzyskaj akcję (skok lub brak skoku) na podstawie stanu gry.

        Parametry:
        - state: stan gry (np. pozycja ptaka, rur itd.)

        Zwraca:
        - 0 (brak skoku) lub 1 (skok)
        """
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return q_values.argmax().item()  # Wybierz akcję z największą wartością Q

    def mutate(self, mutation_rate=0.1, std=0.1):
        """
        Wprowadź mutacje do wag sieci neuronowej.

        Parametry:
        - mutation_rate: prawdopodobieństwo mutacji dla każdego parametru
        - std: odchylenie standardowe szumu gaussowskiego dodawanego do wag
        """
        for param in self.model.parameters():
            if random.random() < mutation_rate:
                param.data += torch.randn_like(param, device=self.device) * std