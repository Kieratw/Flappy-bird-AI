import pygame
import torch
import random

class Bird:
    gravity = 0.5
    lift = -4
    max_fall_speed=9
    max_rise_speed=10
    def __init__(self,x,y,model=None):
        self.x=x
        self.y=y
        self.velocity=0
        self.brain = model
        self.score = 0
        self.dead = False
        try:
            self.image= pygame.image.load('assets/bird.png')
            self.image=pygame.transform.scale(self.image,(30,40))
        except pygame.error as e:
            print("Failed to load bird image")
            self.image=pygame.Surface((30,40))
            self.image.fill((0,0,0))

        self.mask= pygame.mask.from_surface(self.image)
        self.rect=self.image.get_rect(center=(self.x,self.y))
    def update(self):
        self.velocity+=Bird.gravity
        if self.velocity>Bird.max_fall_speed:
            self.velocity=Bird.max_fall_speed
        self.y+=self.velocity
        self.rect.center=(self.x,self.y)


        if self.y>=555:
            self.velocity=0
            self.y=555
        elif self.y<=18:
            self.velocity=0
            self.y=18


    def jump(self):
        if self.max_rise_speed>self.velocity:
            self.velocity=self.lift

    def draw(self,screen):

        screen.blit(self.image,self.rect)

    def get_rect(self):
        return self.rect

    def get_mask(self):
        return self.mask

    def get_action(self, state):
        # Jeśli model nie jest przypisany, domyślnie nie wykonujemy skoku
        if self.brain is None:
            return 0
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.brain.device)
        with torch.no_grad():
            q_values = self.brain.model(state_tensor)
        return q_values.argmax().item()

    def mutate(self, mutation_rate=0.1, std=0.1):
        if self.brain is None:
            return
        for param in self.brain.model.parameters():
            if random.random() < mutation_rate:
                param.data += torch.randn_like(param) * std
