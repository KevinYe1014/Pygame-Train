
import pygame
import pprint

##获得支持的字体  其font不需要initialized  所以直接可以print
##但是display得需要initialized  所以 为了方便都initialized
# pprint.pprint(pygame.font.get_fonts())
# my_font = pygame.font.Font("my_font.ttf", 16)
# print(pygame.display.list_modes())

# my_name='will mcgugan你好'.encode('utf-8')
# pygame.init()
# my_font=pygame.font.SysFont('宋体',64)
# name_surface=my_font.render(my_name,True,(0,0,0),(255,255,255))
# pygame.image.save(name_surface,'name2.png')

# from  pygame.locals import *
# from sys import exit
#
# backgound_image_filenme=r'c:/users/yelei/desktop/back.jpg'
# pygame.init()
# screen=pygame.display.set_mode((640,480),0,32)
#
# ##该字体可以显示中文
# font=pygame.font.SysFont('simsunnsimsun',40)
# text_surface=font.render(u'你好',True,(0,0,255))
#
# x=0
# y=(480-text_surface.get_height())/2
#
# background=pygame.image.load(backgound_image_filenme)
#
# while True:
#     for event in pygame.event.get():
#         if event.type==QUIT:
#             exit()
#     screen.blit(background,(0,0))
#
#     x-=1
#     if x < -text_surface.get_width():
#         x = 640 - text_surface.get_width()
#
#     screen.blit(text_surface, (x, y))
#
#     pygame.display.update()

import pygame
SCREEN_SIZE=(640,-1)

try:
    screen = pygame.display.set_mode(SCREEN_SIZE)
except pygame.error as e:
    print("Can't create the display :-(")
    print(e)
    exit()

