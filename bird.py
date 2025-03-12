import pygame

class Bird:
    gravity = 0.5
    lift = -8
    max_fall_speed=9
    max_rise_speed=10
    def __init__(self,x,y):
        self.x=x
        self.y=y
        self.velocity=0

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

