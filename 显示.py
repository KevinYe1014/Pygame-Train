import pygame
from pygame.locals import *
from sys import exit


###下面代码显示你的机器支持几种分辨率显示模式
# pygame.init()
# print(pygame.display.list_modes())
##[(1920, 1080), (1680, 1050), (1600, 900),.....]

backgound_image_filenme=r'c:/users/yelei/desktop/back.jpg'

# pygame.init()
# screen=pygame.display.set_mode((640,480),0,32)
# background=pygame.image.load(backgound_image_filenme)
#
# Fullscreen=False
#
# while True:
#     for event in pygame.event.get():
#         if event.type==QUIT:
#             exit()
#         if event.type==KEYDOWN:
#             if event.key==K_f:
#                 Fullscreen=not Fullscreen
#                 if Fullscreen:
#                     screen=pygame.display.set_mode((640,480),FULLSCREEN,32)
#                 else:
#                     screen=pygame.display.set_mode((640,480),0,32)
#     screen.blit(background,(0,0))
#     pygame.display.update()


screen_size=(640,480)
pygame.init()
screen=pygame.display.set_mode(screen_size,RESIZABLE,32)

background=pygame.image.load(backgound_image_filenme)

while True:
    event=pygame.event.wait()
    if event.type==QUIT:
        exit()
    if event.type==VIDEORESIZE:
        screen_size=event.size
        screen=pygame.display.set_mode(screen_size,RESIZABLE,32)
        pygame.display.set_caption('window resized to '+str(event.size))

    screen_width,screen_height=screen_size
    for y in range(0,screen_height,background.get_height()):
        for x in range(0,screen_width,background.get_width()):
            screen.blit(background,(x,y))
    pygame.display.update()


