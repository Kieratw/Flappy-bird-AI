import pygame
from pipe import Pipe

class Game:
    def __init__(self, width, height, birds, render_mode=True):
        pygame.init()
        self.width = width
        self.height = height
        self.render_mode = render_mode
        self.birds = birds

        self.screen = None
        if self.render_mode:
            self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        # Inicjalizacja tła
        try:
            self.background_image = pygame.image.load('assets/background.jpg').convert()
            self.background_image = pygame.transform.scale(self.background_image, (self.width, self.height))
        except pygame.error as e:
            print("Failed to load background image")
            self.background_image = pygame.Surface((self.width, self.height))
            self.background_image.fill((255, 255, 255))

        # Inicjalizacja platformy
        self.platform_speed = 3
        self.platform_x = 0
        try:
            self.platform_image = pygame.image.load('assets/platform.jpg').convert()
            self.platform_image = pygame.transform.scale(self.platform_image, (self.width * 2, 130))
        except pygame.error as e:
            print("Failed to load platform image")
            self.platform_image = pygame.Surface((self.width * 2, 130))
            self.platform_image.fill((139, 69, 19))
        self.platform_rect = self.platform_image.get_rect()
        self.platform_rect.bottomleft = (0, self.height)

        # Lista rur i pozostałych elementów
        self.pipes = []
        self.pipe_frequency = 90
        self.frame_count = 0
        self.font = pygame.font.SysFont(None, 48)

    def update(self):
        self.frame_count += 1

        # Aktualizacja wszystkich ptaków (tylko tych żywych)
        for bird in self.birds:
            if not getattr(bird, 'dead', False):
                bird.update()

        # Generowanie nowych rur
        if self.frame_count % self.pipe_frequency == 0:
            self.pipes.append(Pipe(self.width, self.height))

        # Aktualizacja rur i przydzielanie punktów ptakom (jeśli minią przeszkodę)
        for pipe in self.pipes:
            pipe.update()
            for bird in self.birds:
                if (not getattr(bird, 'dead', False)) and (not getattr(bird, 'passed', False)) and (pipe.x + Pipe.width < bird.x):
                    bird.passed = True  # Oznaczamy, że ptak miniął tę rurę
                    # Przydzielamy punkt – zakładamy, że każdy ptak ma atrybut score
                    bird.score = getattr(bird, 'score', 0) + 1

        # Usuwanie rur poza ekranem
        self.pipes = [pipe for pipe in self.pipes if pipe.x + Pipe.width > 0]

        # Przesuwanie platformy
        self.platform_x -= self.platform_speed
        if self.platform_x <= -self.platform_rect.width // 2:
            self.platform_x = 0
        self.platform_rect.bottomleft = (self.platform_x, self.height)
        self.platform_rect_2 = self.platform_rect.copy()
        self.platform_rect_2.bottomleft = (self.platform_x + self.platform_rect.width // 2, self.height)

        # Sprawdzanie kolizji dla każdego ptaka
        for bird in self.birds:
            if getattr(bird, 'dead', False):
                continue
            bird_rect = bird.get_rect()
            bird_mask = bird.get_mask()
            if bird_rect.top <= 0 or bird_rect.colliderect(self.platform_rect) or bird_rect.colliderect(self.platform_rect_2):
                bird.dead = True
            for pipe in self.pipes:
                pipe_rects = pipe.get_rects()
                pipe_masks = pipe.get_mask()
                for i, pipe_rect in enumerate(pipe_rects):
                    if bird_rect.colliderect(pipe_rect):
                        offset_x = pipe_rect.x - bird_rect.x
                        offset_y = pipe_rect.y - bird_rect.y
                        if bird_mask.overlap(pipe_masks[i], (offset_x, offset_y)):
                            bird.dead = True
                            break

        # Zakończenie epizodu: jeśli wszyscy ptacy są martwi
        all_dead = all(getattr(bird, 'dead', False) for bird in self.birds)
        return all_dead, max([getattr(bird, 'score', 0) for bird in self.birds])

    def draw(self):
        if not self.render_mode:
            return
        self.screen.blit(self.background_image, (0, 0))
        self.screen.blit(self.platform_image, (self.platform_x, self.height - self.platform_rect.height))
        self.screen.blit(self.platform_image, (self.platform_x + self.platform_rect.width // 2, self.height - self.platform_rect.height))
        for pipe in self.pipes:
            pipe.draw(self.screen)
        for bird in self.birds:
            if not getattr(bird, 'dead', False):
                bird.draw(self.screen)
        score_text = self.font.render(f"Score: {max([getattr(bird, 'score', 0) for bird in self.birds])}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

    def reset(self, birds=None):
        # Resetowanie gry – jeśli podamy nową listę ptaków, użyj jej; w przeciwnym razie resetuj aktualnych
        if birds is not None:
            self.birds = birds
        else:
            for bird in self.birds:
                bird.dead = False
                bird.score = 0
                bird.passed = False
                bird.x = 400
                bird.y = 100
                bird.velocity = 0
                bird.rect.center = (bird.x, bird.y)
        self.pipes = []
        self.frame_count = 0
        self.platform_x = 0

    def close(self):
        pygame.quit()
