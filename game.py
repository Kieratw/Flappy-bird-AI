import pygame
from bird import Bird
class Game:
    def __init__(self,width,height):
        pygame.init()

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width,self.height))
        self.clock = pygame.time.Clock()

        try:
            self.background_image = pygame.image.load('assets/background.jpg')
        except pygame.error as e:
            print("Failed to load background image")
            self.background_image = pygame.Surface((self.width,self.height))
            self.background_image.fill((255,255,255))
        self.bird=Bird(400,100)
    def run(self):

        running = True
        while running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.bird.jump()


            self.screen.blit(self.background_image,(0,0))

            self.bird.update()
            self.bird.draw(self.screen)
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()