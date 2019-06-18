desktop=r'c:/users/yelei/desktop'
import os
background_image_filename =os.path.join(desktop,'back.jpg')
sprite_image_filename =os.path.join(desktop,'fulu.png')


import pygame
from pygame.locals import *
from sys import exit

# pygame.init()
# screen=pygame.display.set_mode((640,480),0,32)
#
# backgroud=pygame.image.load(background_image_filename)
# spirte=pygame.image.load(sprite_image_filename)
#
# clock=pygame.time.Clock()
# speed=250  ####单位是像素每秒
#
# x=0
# while True:
#     for event in pygame.event.get():
#         if event.type==QUIT:
#             exit()
#     screen.blit(backgroud,(0,0))
#     screen.blit(spirte,(x,100))
#
#     time_passed=clock.tick()  ##返回单位是毫秒
#     time_passed_seconds=time_passed/1000.0
#     distance_moved=time_passed_seconds*speed
#
#     x+=distance_moved
#     if x>640:
#         x=0
#     pygame.display.update()


pygame.init()

screen = pygame.display.set_mode((640, 480), 0, 32)

background = pygame.image.load(background_image_filename).convert()
sprite = pygame.image.load(sprite_image_filename).convert_alpha()

clock = pygame.time.Clock()

x, y = 100., 100.
speed_x, speed_y = 133., 170.

while True:
    for event in pygame.event.get():
        if event.type==QUIT:
            exit()

    screen.blit(background,(0,0))
    screen.blit(sprite,(x,y))

    time_passed=clock.tick()
    time_passed_secondes=time_passed/1000.0

    x+=speed_x*time_passed_secondes
    y+=speed_y*time_passed_secondes

    if x>640-sprite.get_width():
        speed_x=-speed_x
        x=640-sprite.get_width()
    elif x<0:
        speed_x=-speed_x
        x=0
    if y>480-sprite.get_height():
        speed_y=-speed_y
        y=480-sprite.get_height()
    if y<0:
        speed_y=-speed_y
        y=0
    pygame.display.update()


