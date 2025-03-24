import pygame
import random
class Pipe:

    gap=150
    width=80
    speed=3
    platform_height=130
    def __init__(self,x,screen_height):
        self.x=x
        self.screen_height=screen_height
        self.gap_y=random.randint(Pipe.gap,screen_height-Pipe.gap-200)
        self.passed=False

        # Ładowanie obrazu rury
        try:
           base_image = pygame.image.load("assets/pipe.png").convert_alpha()

        except pygame.error as e:
            print(f"Nie można załadować obrazu rury: {e}. Używam prostokąta.")
            base_image = pygame.Surface((Pipe.width, 200))  # Fallback na prostokąt
            base_image.fill((0, 255, 0))

        top_height=self.gap_y-Pipe.gap/2
        self.top_image=pygame.transform.scale(base_image, (Pipe.width, top_height))
        self.top_rect = self.top_image.get_rect(bottomleft=(self.x, self.gap_y - Pipe.gap // 2))

        # Dolna rura (od końca szczeliny do dołu ekranu)
        bottom_start_y = self.gap_y + Pipe.gap // 2
        bottom_height = self.screen_height - Pipe.platform_height - bottom_start_y
        self.bottom_image = pygame.transform.scale(base_image, (Pipe.width, bottom_height))
        self.bottom_image = pygame.transform.flip(self.bottom_image,False,True)
        self.bottom_rect = self.bottom_image.get_rect(topleft=(self.x, self.gap_y + Pipe.gap // 2))

        # Tworzenie masek dla kolizji
        self.top_mask = pygame.mask.from_surface(self.top_image)
        self.bottom_mask = pygame.mask.from_surface(self.bottom_image)

    def update(self):
        # Przesuwanie rur w lewo
        self.x -= Pipe.speed
        self.top_rect.x = self.x
        self.bottom_rect.x = self.x

    def draw(self, screen):
        # Rysowanie górnej i dolnej rury
        screen.blit(self.top_image, self.top_rect)
        screen.blit(self.bottom_image, self.bottom_rect)

    def get_rects(self):
        # Zwraca prostokąty rur do wykrywania kolizji
        return [self.top_rect, self.bottom_rect]

    def get_mask(self):
        # Zwraca maski rur do dokładnego wykrywania kolizji
        return [self.top_mask, self.bottom_mask]