
import sys
import math

from glfw.GLFW import *

from OpenGL.GL import *
from OpenGL.GLU import *
def trojkat(x,y,a,p):
    if (p==0):
        glColor3f(1, 0.5, 0.25)
        glBegin(GL_TRIANGLES)
        glVertex2f(x, y)
        glVertex2f(x+a/2, y+a* math.sqrt(3)/2)
        glVertex2f(x+a, y)
        glEnd()
    else:
        a=a/2
        for i in range(5):
            if(i==0):
                trojkat(x+2*a/4,y+2*a*math.sqrt(3)/4,a,p-1)
            elif(i==1):
                glColor3f(1.0, 1.0, 1.0)
                glBegin(GL_TRIANGLES)
                glVertex2f(x+2*a/4,y+2*a*math.sqrt(3)/4)
                glVertex2f(x+6*a/4,y+2*a*math.sqrt(3)/4)
                glVertex2f((x+a), (y))
                glEnd()
            elif (i==2):
                trojkat(x,y,a,p-1)
            elif (i==3):
                trojkat(x+a,y,a,p-1)

            
def dywan(x,y,a,b,p): 
    if (p==0):
        glColor3f(1, 0.5, 0.25)
        glBegin(GL_TRIANGLES)
        glVertex2f(x, y)
        glVertex2f(x, y+b)
        glVertex2f(x+a, y)
        glEnd()
        
        
        glColor3f(1, 0.5, 0.25)
        glBegin(GL_TRIANGLES)
        glVertex2f(x+a, y+b)
        glVertex2f(x, y+b)
        glVertex2f(x+a, y)
        glEnd()
    else:
        a=a/3
        b=b/3
        for i in range(4):
            for j in range(4):
                if (i==1) and (j==1):
                    dywan(x,y,a,b,p-1)
                elif (i==1) and (j==1):
                    dywan(x+a,y,a,b,p-1)
                elif (i==1) and (j==3):
                    dywan(x+2*a,y,a,b,p-1)
                elif (i==2) and (j==1):
                    dywan(x,y+b,a,b,p-1) 
                elif (i==2) and (j==2):
                        glColor3f(1.0, 1.0, 1.0)
                        glBegin(GL_TRIANGLES)
                        glVertex2f((x+a), (y+b))
                        glVertex2f((x+2*a), (y+b))
                        glVertex2f((x+2*a), (y+2*b))
                        glEnd()

                        glColor3f(1.0, 1.0, 1.0)
                        glBegin(GL_TRIANGLES)
                        glVertex2f((x+a), (y+b))
                        glVertex2f((x+a), (y+2*b))
                        glVertex2f((x+2*a), (y+2*b))
                        glEnd()
                elif (i==2) and (j==3):
                    dywan(x+2*a,y+a,a,b,p-1)
                elif (i==3) and (j==1):
                    dywan(x,y+2*a,a,b,p-1)
                elif (i==3) and (j==2):
                    dywan(x+a,y+2*a,a,b,p-1)
                elif (i==3) and (j==3):
                    dywan(x+2*a,y+2*a,a,b,p-1)

                   


def startup():
    update_viewport(None, 400, 400)
    glClearColor(0.5, 0.5, 0.5, 1.0)


def shutdown():
    pass


def render(time):
    glClear(GL_COLOR_BUFFER_BIT)

    #dywan(0,0,100,100,3)
    trojkat(0,0,100,3)
 
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
        glOrtho(-100.0, 100.0, -100.0 / aspect_ratio, 100.0 / aspect_ratio,
                1.0, -1.0)
    else:
        glOrtho(-100.0 * aspect_ratio, 100.0 * aspect_ratio, -100.0, 100.0,
                1.0, -1.0)

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