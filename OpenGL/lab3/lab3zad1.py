#!/usr/bin/env python3
import sys
import math
from glfw.GLFW import *


from OpenGL.GL import *
from OpenGL.GLU import *


  

def jajko(n):
   punkty = [[[0] * 3 for i in range(n)] for j in range(n)]
   glBegin(GL_POINTS)
   for i in range(n):
       for j in range(n):
           u = i/n
           v = j/n
           x = (-90*u*u*u*u*u+225*u*u*u*u-270*u*u*u+180*u*u-45*u)*math.cos(v*math.pi)
           y = 160*u*u*u*u-320*u*u*u+160*u*u
           z = (-90*u*u*u*u*u + 225*u*u*u*u-270*u*u*u+180*u*u-45*u)*math.sin(math.pi*v)
           punkty[i][j][0]= x
           punkty[i][j][1]= y
           punkty[i][j][2]= z
           glColor3f(0.2,0.4,0.5)
           glVertex3f(punkty[i][j][0],punkty[i][j][1],punkty[i][j][2])
   glEnd()
  

def startup():
   update_viewport(None, 400, 400)
   glClearColor(0.0, 0.0, 0.0, 1.0)
   glEnable(GL_DEPTH_TEST)




def shutdown():
   pass




def axes():
   glBegin(GL_LINES)


   glColor3f(1.0, 0.0, 0.0)
   glVertex3f(-5.0, 0.0, 0.0)
   glVertex3f(5.0, 0.0, 0.0)


   glColor3f(0.0, 1.0, 0.0)
   glVertex3f(0.0, -5.0, 0.0)
   glVertex3f(0.0, 5.0, 0.0)


   glColor3f(0.0, 0.0, 1.0)
   glVertex3f(0.0, 0.0, -5.0)
   glVertex3f(0.0, 0.0, 5.0)


   glEnd()




def render(time):
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
   glLoadIdentity()


   axes()
   jajko(100)
   glColor3f(1.0, 0.5, 0.0)
   glFlush()




def update_viewport(window, width, height):
   if width == 0:
       width = 1
   if height == 0:
       height = 1
   aspect_ratio = width / height


   glMatrixMode(GL_PROJECTION)
   glViewport(0, 0, width, height)
   glLoadIdentity()


   if width <= height:
       glOrtho(-7.5, 7.5, -7.5 / aspect_ratio, 7.5 / aspect_ratio, 7.5, -7.5)
   else:
       glOrtho(-7.5 * aspect_ratio, 7.5 * aspect_ratio, -7.5, 7.5, 7.5, -7.5)


   glMatrixMode(GL_MODELVIEW)
   glLoadIdentity()




def main():
   if not glfwInit():
       sys.exit(-1)


   window = glfwCreateWindow(400, 400, __file__, None, None)
   if not window:
       glfwTerminate()
       sys.exit(-1)


   glfwMakeContextCurrent(window)
   glfwSetFramebufferSizeCallback(window, update_viewport)
   glfwSwapInterval(1)
  
   startup()
   while not glfwWindowShouldClose(window):
       render(glfwGetTime())
       glfwSwapBuffers(window)
       glfwPollEvents()
   shutdown()


   glfwTerminate()




if __name__ == '__main__':
   main()





