# import pygame
# from pygame.locals import *
# from sys import exit
#
# pygame.init()
# SCREEN_SIZE = (640, 480)
# screen = pygame.display.set_mode(SCREEN_SIZE, 0, 32)
#
# font = pygame.font.SysFont("arial", 16);
# font_height = font.get_linesize()
# event_text = []
#
# while True:
#
#     event = pygame.event.wait()
#     event_text.append(str(event))
#     #获得时间的名称
#     event_text = event_text[-SCREEN_SIZE[1]//font_height:]
#     #这个切片操作保证了event_text里面只保留一个屏幕的文字
#
#     if event.type == QUIT:
#         exit()
#
#     screen.fill((255, 255, 255))
#
#     y = SCREEN_SIZE[1]-font_height
#     #找一个合适的起笔位置，最下面开始但是要留一行的空
#     for text in reversed(event_text):
#         screen.blit( font.render(text, True, (0, 0, 0)), (0, y) )
#         #以后会讲
#         y-=font_height
#         #把笔提一行
#
#     pygame.display.update()

backgound_image_filenme=r'c:/users/yelei/desktop/back.jpg'

import pygame
from pygame.locals import *
from sys import exit

pygame.init()
screen=pygame.display.set_mode((640,480),0,32)
background=pygame.image.load(backgound_image_filenme)

x,y=0,0
move_x,move_y=0,0
while True:
    for event in pygame.event.get():
        ##长按和短按 事件都是一个  按住是一个  当然松开也是一个
        # print(str(event))

        if event.type==QUIT:
            exit()
        if event.type==KEYDOWN:
            if event.key==K_LEFT:
                move_x=-1
            elif event.key==K_RIGHT:
                move_x=1
            elif event.key==K_UP:
                move_y=1
            elif event.key==K_DOWN:
                move_y=-1
        elif event.type==KEYUP:
            move_x=0
            move_y=0
    x+=move_x   ###this is important indent
    y+=move_y

    screen.fill((0,0,0))
    screen.blit(background,(x,y))

    pygame.display.update()



