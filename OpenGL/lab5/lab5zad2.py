#!/usr/bin/env python3
import sys

from glfw.GLFW import *

from OpenGL.GL import *
from OpenGL.GLU import *


viewer = [0.0, 0.0, 10.0]

theta = 0.0
alfa= 0.0
pix2angle = 1.0

left_mouse_button_pressed = 0
mouse_x_pos_old = 0
delta_x = 0
mouse_y_pos_old = 0
delta_y = 0
lightchoice = 0
keypressed = 0
keyvalue = 0.0

mat_ambient = [1.0, 1.0, 1.0, 1.0]
mat_diffuse = [1.0, 1.0, 1.0, 1.0]
mat_specular = [1.0, 1.0, 1.0, 1.0]
mat_shininess = 20.0

light_ambient = [0.1, 0.1, 0.0, 1.0]
light_diffuse = [0.0, 0.8, 0.0, 1.0]
light_specular = [1.0, 1.0, 1.0, 1.0]
light_position = [0.0, 0.0, 10.0, 1.0]


att_constant = 1.0
att_linear = 0.05
att_quadratic = 0.001


def startup():
    update_viewport(None, 400, 400)
    glClearColor(0.0, 0.0, 0.0, 1.0)
    glEnable(GL_DEPTH_TEST)

    glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient)
    glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse)
    glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
    glMaterialf(GL_FRONT, GL_SHININESS, mat_shininess)

    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)

    

    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, att_constant)
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, att_linear)
    glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, att_quadratic)



    glShadeModel(GL_SMOOTH)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)


def shutdown():
    pass


def render(time):
    global theta
    global alfa
    global light_ambient
    global light_diffuse
    global light_specular

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    gluLookAt(viewer[0], viewer[1], viewer[2],
              0.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    if left_mouse_button_pressed:
        theta += delta_x * pix2angle
        alfa += delta_y * pix2angle
    if keypressed ==1:
        match lightchoice:
            case 0:
                light_ambient[0] += keyvalue
                if light_ambient[0] >1:
                    light_ambient[0] =1
                if light_ambient[0] <0:
                    light_ambient[0] =0
            case 1:
                light_ambient[1] += keyvalue
                if light_ambient[1] >1:
                    light_ambient[1] =1
                if light_ambient[1] <0:
                    light_ambient[1] =0
            case 2:
                light_ambient[2] += keyvalue
                if light_ambient[2] >1:
                    light_ambient[2] =1
                if light_ambient[2] <0:
                    light_ambient[2] =0
            case 3:
                light_diffuse[0] += keyvalue
                if light_diffuse[0] >1:
                    light_diffuse[0] =1
                if light_diffuse[0] <0:
                    light_diffuse[0] =0
            case 4:
                light_diffuse[1] += keyvalue
                if light_diffuse[1] >1:
                    light_diffuse[1] =1
                if light_diffuse[1] <0:
                    light_diffuse[1] =0
            case 5:
                light_diffuse[2] += keyvalue
                if light_diffuse[2] >1:
                    light_diffuse[2] =1
                if light_diffuse[2] <0:
                    light_diffuse[2] =0
            case 6:
                light_specular[0] += keyvalue
                if light_specular[0] >1:
                    light_specular[0] =1
                if light_specular[0] <0:
                    light_specular[0] =0
            case 7:
                light_specular[1] += keyvalue
                if light_specular[1] >1:
                    light_specular[1] =1
                if light_specular[1] <0:
                    light_specular[1] =0
            case 8:
                light_specular[2] += keyvalue
                if light_specular[2] >1:
                    light_specular[2] =1
                if light_specular[2] <0:
                    light_specular[2] =0
        

    glRotatef(theta, 0.0, 1.0, 0.0)
    glRotatef(alfa,1.0,0,0)

    quadric = gluNewQuadric()
    gluQuadricDrawStyle(quadric, GLU_FILL)
    gluSphere(quadric, 3.0, 10, 10)
    gluDeleteQuadric(quadric)
    glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient)
    glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse)
    glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular)
    glLightfv(GL_LIGHT0, GL_POSITION, light_position)

    

    glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, att_constant)
    glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, att_linear)
    glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, att_quadratic)

    glFlush()


def update_viewport(window, width, height):
    global pix2angle
    pix2angle = 360.0 / width

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    gluPerspective(70, 1.0, 0.1, 300.0)

    if width <= height:
        glViewport(0, int((height - width) / 2), width, width)
    else:
        glViewport(int((width - height) / 2), 0, height, height)

    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()


def keyboard_key_callback(window, key, scancode, action, mods):
    global lightchoice
    global keypressed
    global keyvalue
    if key == GLFW_KEY_ESCAPE and action == GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)
    if key == GLFW_KEY_1 and action == GLFW_PRESS:
        lightchoice += 1
        lightchoice = lightchoice%9
    if key == GLFW_KEY_2 and action == GLFW_PRESS:
        keypressed = 1
        keyvalue = -0.1
    elif key == GLFW_KEY_3 and action == GLFW_PRESS:
        keypressed = 1
        keyvalue = 0.1
    else:
        keypressed = 0
           
        



def mouse_motion_callback(window, x_pos, y_pos):
    global delta_x
    global mouse_x_pos_old
    global delta_y
    global mouse_y_pos_old

    delta_x = x_pos - mouse_x_pos_old
    mouse_x_pos_old = x_pos
    delta_y = y_pos - mouse_y_pos_old
    mouse_y_pos_old = y_pos


def mouse_button_callback(window, button, action, mods):
    global left_mouse_button_pressed

    if button == GLFW_MOUSE_BUTTON_LEFT and action == GLFW_PRESS:
        left_mouse_button_pressed = 1
    else:
        left_mouse_button_pressed = 0


def main():
    if not glfwInit():
        sys.exit(-1)

    window = glfwCreateWindow(400, 400, __file__, None, None)
    if not window:
        glfwTerminate()
        sys.exit(-1)

    glfwMakeContextCurrent(window)
    glfwSetFramebufferSizeCallback(window, update_viewport)
    glfwSetKeyCallback(window, keyboard_key_callback)
    glfwSetCursorPosCallback(window, mouse_motion_callback)
    glfwSetMouseButtonCallback(window, mouse_button_callback)
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
