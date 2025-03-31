import pygame
import sys
from Training import main as run_training_loop
from demo import run_demo_loop
from play_manual import run_manual_loop

pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Flappy Bird AI - Menu")
font = pygame.font.SysFont(None, 48)

def draw_button(text, y, color=(0, 128, 255)):
    text_surface = font.render(text, True, (255, 255, 255))
    rect = text_surface.get_rect(center=(400, y))
    pygame.draw.rect(screen, color, rect.inflate(20, 20))
    screen.blit(text_surface, rect)
    return rect

def main_menu():
    while True:
        screen.fill((0, 0, 0))
        train_rect = draw_button("Trenuj AI", 200)
        demo_rect = draw_button("Nauczony model", 300)
        play_rect = draw_button("Zagraj sam", 400)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                if train_rect.collidepoint(event.pos):
                    run_training_loop()
                elif demo_rect.collidepoint(event.pos):
                    run_demo_loop()
                elif play_rect.collidepoint(event.pos):
                    run_manual_loop(screen)

if __name__ == "__main__":
    main_menu()
