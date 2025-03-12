import pygame
from bird import Bird
from pipe import Pipe

class Game:
    def __init__(self, width, height):
        pygame.init()

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        # Inicjalizacja tła (statyczne)
        try:
            self.background_image = pygame.image.load('assets/background.jpg').convert()
            self.background_image = pygame.transform.scale(self.background_image, (self.width, self.height))
        except pygame.error as e:
            print("Failed to load background image")
            self.background_image = pygame.Surface((self.width, self.height))
            self.background_image.fill((255, 255, 255))

        # Inicjalizacja platformy (przesuwanej na dole)
        self.platform_speed = 3
        self.platform_x = 0
        try:
            self.platform_image = pygame.image.load('assets/platform.jpg').convert()
            self.platform_image = pygame.transform.scale(self.platform_image, (self.width * 2, 130))
        except pygame.error as e:
            print("Failed to load platform image")
            self.platform_image = pygame.Surface((self.width * 2, 130))
            self.platform_image.fill((139, 69, 19))  # Brązowy kolor jako fallback

        self.platform_rect = self.platform_image.get_rect()
        self.platform_rect.bottomleft = (0, self.height)

        # Inicjalizacja ptaka
        self.bird = Bird(400, 100)

        # Inicjalizacja rur
        self.pipes = []
        self.pipe_frequency = 90
        self.frame_count = 0
        self.score = 0

        # Czcionka do wyniku
        self.font = pygame.font.SysFont(None, 48)

    def update(self):
        self.frame_count += 1

        # Aktualizacja ptaka
        self.bird.update()

        # Generowanie nowych rur
        if self.frame_count % self.pipe_frequency == 0:
            self.pipes.append(Pipe(self.width, self.height))

        # Aktualizacja rur
        for pipe in self.pipes:
            pipe.update()
            if not pipe.passed and pipe.x + Pipe.width < self.bird.x:
                pipe.passed = True
                self.score += 1

        # Usuwanie rur poza ekranem
        self.pipes = [pipe for pipe in self.pipes if pipe.x + Pipe.width > 0]

        # Przesuwanie platformy
        self.platform_x -= self.platform_speed
        if self.platform_x <= -self.platform_rect.width // 2:
            self.platform_x = 0

        # Aktualizacja pozycji prostokąta platformy
        self.platform_rect.bottomleft = (self.platform_x, self.height)
        self.platform_rect_2 = self.platform_rect.copy()
        self.platform_rect_2.bottomleft = (self.platform_x + self.platform_rect.width // 2, self.height)

        # Wykrywanie kolizji
        bird_rect = self.bird.get_rect()
        bird_mask = self.bird.get_mask()
        game_over = False

        # Kolizja z sufitem
        if bird_rect.top <= 0:
            game_over = True

        # Kolizja z platformą
        if bird_rect.colliderect(self.platform_rect) or bird_rect.colliderect(self.platform_rect_2):
            game_over = True

        # Kolizja z rurami
        for pipe in self.pipes:
            pipe_rects = pipe.get_rects()
            pipe_masks = pipe.get_mask()
            for i, pipe_rect in enumerate(pipe_rects):
                if bird_rect.colliderect(pipe_rect):
                    offset_x = pipe_rect.x - bird_rect.x
                    offset_y = pipe_rect.y - bird_rect.y
                    if bird_mask.overlap(pipe_masks[i], (offset_x, offset_y)):
                        game_over = True
                        break

        return game_over, self.score

    def draw(self):
        self.screen.blit(self.background_image, (0, 0))
        self.screen.blit(self.platform_image, (self.platform_x, self.height - self.platform_rect.height))
        self.screen.blit(self.platform_image, (self.platform_x + self.platform_rect.width // 2, self.height - self.platform_rect.height))
        for pipe in self.pipes:
            pipe.draw(self.screen)
        self.bird.draw(self.screen)

        # Wyświetlanie wyniku
        score_text = self.font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

    def reset(self):
        self.bird = Bird(400, 100)
        self.pipes = []
        self.frame_count = 0
        self.score = 0
        self.platform_x = 0