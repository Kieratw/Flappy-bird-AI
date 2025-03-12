import pygame

class GameManager:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()

        from game import Game  # Importujemy Game tutaj, aby uniknąć cyklicznych importów
        self.game = Game(self.width, self.height)

        # Czcionki
        self.font = pygame.font.SysFont(None, 48)
        self.large_font = pygame.font.SysFont(None, 72)

        # Stan gry
        self.state = "menu"

    def draw_menu(self):
        self.screen.blit(self.game.background_image, (0, 0))
        self.screen.blit(self.game.platform_image, (self.game.platform_x, self.height - self.game.platform_rect.height))
        self.screen.blit(self.game.platform_image, (self.game.platform_x + self.game.platform_rect.width // 2, self.height - self.game.platform_rect.height))

        title_text = self.large_font.render("Flappy Bird", True, (255, 255, 255))
        start_text = self.font.render("Start Game", True, (255, 255, 255))

        title_rect = title_text.get_rect(center=(self.width // 2, self.height // 3))
        start_rect = start_text.get_rect(center=(self.width // 2, self.height // 2))

        self.screen.blit(title_text, title_rect)
        pygame.draw.rect(self.screen, (0, 0, 255), start_rect.inflate(20, 20))  # Niebieski przycisk
        self.screen.blit(start_text, start_rect)

        return start_rect

    def draw_game_over(self):
        self.screen.blit(self.game.background_image, (0, 0))
        self.screen.blit(self.game.platform_image, (self.game.platform_x, self.height - self.game.platform_rect.height))
        self.screen.blit(self.game.platform_image, (self.game.platform_x + self.game.platform_rect.width // 2, self.height - self.game.platform_rect.height))

        game_over_text = self.large_font.render("Game Over", True, (255, 0, 0))
        score_text = self.font.render(f"Score: {self.game.score}", True, (255, 255, 255))
        restart_text = self.font.render("Restart", True, (255, 255, 255))

        game_over_rect = game_over_text.get_rect(center=(self.width // 2, self.height // 3))
        score_rect = score_text.get_rect(center=(self.width // 2, self.height // 2 - 20))
        restart_rect = restart_text.get_rect(center=(self.width // 2, self.height // 2 + 20))

        self.screen.blit(game_over_text, game_over_rect)
        self.screen.blit(score_text, score_rect)
        pygame.draw.rect(self.screen, (0, 255, 0), restart_rect.inflate(20, 20))  # Zielony przycisk
        self.screen.blit(restart_text, restart_rect)

        return restart_rect

    def run(self):
        while True:
            if self.state == "menu":
                start_rect = self.draw_menu()
                pygame.display.flip()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if start_rect.collidepoint(event.pos):
                            self.state = "game"

            elif self.state == "game":
                game_over, score = self.game.update()
                self.game.draw()
                pygame.display.flip()
                self.clock.tick(60)

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_SPACE:
                            self.game.bird.jump()

                if game_over:
                    self.state = "game_over"

            elif self.state == "game_over":
                restart_rect = self.draw_game_over()
                pygame.display.flip()

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if restart_rect.collidepoint(event.pos):
                            self.game.reset()
                            self.state = "game"