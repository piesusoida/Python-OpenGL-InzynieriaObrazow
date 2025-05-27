import cv2
from matplotlib import pyplot
import math
import numpy


# Ustawienie rozmarów wyświetlanych obrazów
pyplot.rcParams["figure.figsize"] = (18, 10)

image_from_file = cv2.imread('images/arch.png')
#image_from_file = cv2.imread('images/gacław-na-kuchni.jpg')
image_gray = cv2.cvtColor(image_from_file, cv2.COLOR_BGR2GRAY)
image_color = cv2.cvtColor(image_from_file, cv2.COLOR_BGR2RGB)
print('Rozmiar obrazka: ', image_from_file.shape)
width = 80
height = 60
image = numpy.zeros((height, width, 3), dtype=numpy.uint8)

sign = lambda x: 0 if x == 0 else math.copysign(1, x)






def find_closest_palette_grey(old_pixel):
    return round(old_pixel/ 255 ) * 255

def find_closest_palette_color(old_pixel, k):
    to_return = [0, 0, 0]
    to_return[0] = round(   (k - 1)*(old_pixel[0]/ 255) ) * 255 / (k-1) 
    to_return[1] = round(   (k - 1)*(old_pixel[1]/ 255) ) * 255 / (k-1)
    to_return[2] = round(   (k - 1)*(old_pixel[2]/ 255) ) * 255 / (k-1)
    return to_return

def draw_point(image, x, y, color=(255, 255, 255)):
    image[image.shape[0] - 1 - y, x] = color

# funkcja do wyznaczania pola
def get_area(a, b, c):
    return (c[0] - a[0] ) * (b[1] - a[1]) - (c[1] - a[1]) *  (b[0] - a[0]) 


#
# Funkcja rysująca linię
#
def draw_line(image, x1, y1, x2, y2, col1 = (0, 0, 0), col2 = (255, 255, 255)):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    col1 = numpy.asarray(col2)
    col2 = numpy.asarray(col2)
    Xi = 0 if dx == 0 else (x2-x1)/dx 
    Yi = 0 if dy == 0 else (y2-y1)/dy 
    d = 0 
    if dx > dy:
        d = 2 * dy - dx
        diff = (col2 - col1) // dx
    else:
        d = 2 * dx - dy
        diff = (col2 - col1) // dy
    x0, y0 = x1, y1
    C = col1
    draw_point(image, x0, y0, col1)

    while x0 != x2 or y0 != y2:
        C = numpy.clip(C+diff, 0, 255)
        if dx > dy:
            x0+=Xi
            d+= 2* dy
            if d >= 0:
                y0+=Yi
                d -= 2 * dx
        else :
            y0+=Yi
            d+= 2*dx
            if d>=0:
                x0+=Xi
                d-= 2* dy        
        draw_point(image, int(x0), int(y0), C )


#
# Funkcja rysująca trójkąt
#
def draw_triangle(image, a, b, c, col1 = [255, 0, 0], col2 = [0, 255, 0], col3 = [0, 0, 255]):

    col1= numpy.asarray(col1)
    col2= numpy.asarray(col1)
    col3= numpy.asarray(col1)

    x_min = min(a[0], b[0], c[0])
    y_min = min(a[1], b[1], c[1])

    x_max = max(a[0], b[0], c[0])
    y_max = max(a[1], b[1], c[1])
    

    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            point = (x, y)

            p1 = get_area(a, b, point)
            p2 = get_area(b, c, point)
            p3 = get_area(c, a, point)
            l = get_area(a, b, c)
            Cp = p1/l * col1 + p2/l * col2 + p3/l * col3

            if sign(p1) == sign(p2) and sign(p2) == sign(p3) and sign(p1) == sign(p3) :
                draw_point(image, x, y, Cp)

def draw_linezad4(image, x1, y1, x2, y2, col1 = (0, 0, 0), col2 = (255, 255, 255)):
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    col1 = numpy.asarray(col1)
    col2 = numpy.asarray(col2)
    Xi = 0 if dx == 0 else (x2-x1)/dx 
    Yi = 0 if dy == 0 else (y2-y1)/dy 
    d = 0 
    if dx > dy:
        d = 2 * dy - dx
        diff = (col2 - col1) // dx
    else:
        d = 2 * dx - dy
        diff = (col2 - col1) // dy
    x0, y0 = x1, y1
    C = col1
    draw_point(image, x0, y0, col1)

    while x0 != x2 or y0 != y2:
        C = numpy.clip(C+diff, 0, 255)
        if dx > dy:
            x0+=Xi
            d+= 2* dy
            if d >= 0:
                y0+=Yi
                d -= 2 * dx
        else :
            y0+=Yi
            d+= 2*dx
            if d>=0:
                x0+=Xi
                d-= 2* dy        
        draw_point(image, int(x0), int(y0), C )

def draw_trianglezad4(image, a, b, c, col1 = [255, 0, 0], col2 = [0, 255, 0], col3 = [0, 0, 255]):

    col1= numpy.asarray(col1)
    col2= numpy.asarray(col2)
    col3= numpy.asarray(col3)

    x_min = min(a[0], b[0], c[0])
    y_min = min(a[1], b[1], c[1])

    x_max = max(a[0], b[0], c[0])
    y_max = max(a[1], b[1], c[1])
    

    for x in range(x_min, x_max):
        for y in range(y_min, y_max):
            point = (x, y)

            p1 = get_area(a, b, point)
            p2 = get_area(b, c, point)
            p3 = get_area(c, a, point)
            l = get_area(a, b, c)
            Cp = p1/l * col1 + p2/l * col2 + p3/l * col3

            if sign(p1) == sign(p2) and sign(p2) == sign(p3) and sign(p1) == sign(p3) :
                draw_point(image, x, y, Cp)

def zadanie1():
    output = numpy.copy(image_gray)
    output = output.astype(int)


    print(output.shape[0]   , output.shape[1])

    #
    # Algorytm
    #
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            old_pixel = output[y][x]
            new_pixel = find_closest_palette_grey(old_pixel)
            output[y][x] = new_pixel
            quant_error = old_pixel - new_pixel
            if x+ 1 < output.shape[1]:
                output[y    ][x + 1] = output[y    ][x + 1] + quant_error * 7/16
            if y+ 1 < output.shape[0] and x-1 > 0 :
                output[y + 1][x - 1] = output[y + 1][x - 1] + quant_error * 3/16
            if y+ 1 < output.shape[0]:
                output[y + 1][x    ] = output[y + 1 ][x ] + quant_error * 5/16
            if x+ 1 < output.shape[1] and  y + 1 < output.shape[0]:
                output[y + 1][x + 1] = output[y + 1][x + 1] + quant_error * 1/16
            #zgodnie z prezka przeksztlacmy liczby do zasiegu 0 - 255
    numpy.clip(output, 0, 255)
            #formatujemy do uint8
    output = output.astype(dtype=numpy.uint8)
    #
    # Wyświetlenie
    #
    pyplot.imshow(output, cmap='gray')
    pyplot.show()


def zadanie2():
    output = numpy.copy(image_color)
    output = output.astype(dtype=int)
    reduced = numpy.copy(image_color)
    K = 2
    tones = 2

    #
    # Algorytm
    #
    for y in range(output.shape[0]-1):
        for x in range(output.shape[1]-1):
            # old_pixel = output[y][x]
            # new_pixel = find_closest_palette_color(old_pixel, K)
            # output[y][x] = new_pixel
            # quant_error = old_pixel - new_pixel           
            # output[y    ][x + 1] = output[y    ][x + 1] + quant_error * 7/16
            # output[y + 1][x - 1] = output[y + 1][x - 1] + quant_error * 3/16
            # output[y + 1][x    ] = output[y + 1 ][x   ] + quant_error * 5/16
            # output[y + 1][x + 1] = output[y + 1][x + 1] + quant_error * 1/16


            oldpixel = [output[y][x][0], output[y][x][1], output[y][x][2]]
            newpixel = [0, 0, 0]
            newpixel = find_closest_palette_color(oldpixel, 10) 
            reduced[y][x] = find_closest_palette_color(reduced[y][x], 10)
            output[y][x] = newpixel
            for i in range(3):
                quant_error = oldpixel[i] - newpixel[i]
                output[y][x + 1][i] = output[y][x + 1][i] + quant_error * 7 / 16
                output[y + 1][x - 1][i] = output[y + 1][x - 1][i] + quant_error * 3 / 16
                output[y + 1][x][i] = output[y + 1][x][i] + quant_error * 5 / 16
                output[y + 1][x + 1][i] = output[y + 1][x + 1][i] + quant_error * 1 / 16
    #zgodnie z prezka przeksztlacmy liczby do zasiegu 0 - 255
    numpy.clip(output, 0, 255)
            #formatujemy do uint8
    output = output.astype(dtype=numpy.uint8)
    #
    # Wyświetlenie
    #
    fig, ax = pyplot.subplots(1, 2)
    ax[0].imshow(output)
    ax[1].imshow(reduced)
    pyplot.show()

    #
    # Histogram
    #
    color = ('r', 'g', 'b')

    for i, col in enumerate(color):
        histr = cv2.calcHist([output], [i], None, [256], [0, 256])
        pyplot.plot(histr, color=col)
        pyplot.xlim([-1, 256])
        pyplot.xlabel('Wartośc składowej koloru []')
        pyplot.ylabel('Liczba pikseli obrazu []')
        pyplot.show()


def zadanie3():

    #
    # Przygotowanie płótna
    #
    width = 80
    height = 60
    image = numpy.zeros((height, width, 3), dtype=numpy.uint8)

    sign = lambda x: 0 if x == 0 else math.copysign(1, x)
    #
    # Funkcja rysująca punkt
    #
    # NOTE(sdatko): punkt 0,0 to lewy dolny róg obrazu
    #

    #
    # Rysowanie
    #
    draw_point(image, 2, 4)
    draw_line(image, 3, 40, 10, 40)
    draw_triangle(image, (30, 10), (20, 30), (40, 20) )
    #
    # Wyświetlenie
    #
    pyplot.imshow(image)
    pyplot.show()

def zadanie4():

    #
    # Przygotowanie płótna
    #
    width = 80
    height = 60
    image = numpy.zeros((height, width, 3), dtype=numpy.uint8)

    sign = lambda x: 0 if x == 0 else math.copysign(1, x)
    #
    # Funkcja rysująca punkt
    #
    # NOTE(sdatko): punkt 0,0 to lewy dolny róg obrazu
    #

    #
    # Rysowanie
    #
    draw_point(image, 2, 4)
    draw_linezad4(image, 3, 40, 10, 40)
    draw_trianglezad4(image, (30, 10), (20, 30), (40, 20) )
    #
    # Wyświetlenie
    #
    pyplot.imshow(image)
    pyplot.show()

def zadanie5():
    #
    # Rysowanie
    #
    width, height = 80, 60
    scale = 2
    super_width, super_height = width * scale, height * scale

    # Supersampled image
    image_scaled = numpy.zeros((super_height, super_width, 3), dtype=numpy.uint8)

    draw_point(image_scaled, 2 * scale, 4 * scale)
    draw_linezad4(image_scaled, 3 * scale, 40 * scale, 10 * scale, 40 * scale)
    draw_trianglezad4(
        image_scaled,
        (30 * scale, 10 * scale),
        (20 * scale, 30 * scale),
        (40 * scale, 20 * scale),
    )

    # --- Downsample to original resolution using box filter ---
    image_final = numpy.zeros((height, width, 3), dtype=numpy.uint8)
    for y in range(height):
        for x in range(width):
            block = image_scaled[y*scale:y*scale+scale, x*scale:x*scale+scale].astype(numpy.uint16)
            avg_color = block.mean(axis=(0, 1))
            image_final[y, x] = numpy.clip(avg_color, 0, 255).astype(numpy.uint8)

    # --- Display the result ---
    pyplot.imshow(image_final)
    pyplot.axis('off')
    pyplot.show()

def menu():
    options = ["zadanie 1", "zadanie 2", "zadanie 3", "zadanie 4", "zadanie 5"]
    
    # Print the options
    print("Wybierz zadanie:")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")

    while True:
        try:
            # Ask the user to input a selection
            selection = int(input("Enter the number of your choice: "))
            
            # Store the input in a variable
            selected_option = selection
            
            # Check if the selection is valid
            if 1 <= selected_option <= 5:
                # Call the corresponding function based on the selection
                if selected_option == 1:
                    zadanie1()
                elif selected_option == 2:
                    zadanie2()
                elif selected_option == 3:
                    zadanie3()
                elif selected_option == 4:
                    zadanie4()
                elif selected_option == 5:
                    zadanie5()
                break  # Exit the loop if a valid option is selected
            else:
                print("Invalid selection. Please choose a number between 1 and 5.")
        except ValueError:
            print("Invalid input. Please enter a number.")

menu()