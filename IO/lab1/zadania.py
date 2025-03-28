import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["figure.figsize"] = (18, 10) # wielkosc obrazu
image = cv.imread("images/example.jpg")
def start():
    image = cv.imread("images/example.jpg")
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.show() # wyswitlenie

    print(type(image), image.shape)
    print(type(image[0, 0, 0])) #info w konsoli


def zadanie1():
    kernel = [
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1],
    ]
    kernel = np.asarray(kernel)
    filtered_image = cv.filter2D(image, -1, kernel=kernel) 

    plt.imshow(filtered_image)
    plt.show()

def zadanie2():
    image = cv.imread("images/example.jpg")
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image_rgb_float = image_rgb / 255.0
    transformation_matrix = np.array([
    [0.393, 0.769, 0.189],
    [0.349, 0.689, 0.168], 
    [0.272, 0.534, 0.131]
    ])
    pixels = image_rgb_float.reshape(-1, 3)
    transformed_pixels = pixels.dot(transformation_matrix.T)
    transformed_pixels = np.clip(transformed_pixels, 0, 1) #limit na 1
    transformed_image = transformed_pixels.reshape(image_rgb.shape)
    plt.imshow(transformed_image)
    plt.show()

def zadanie3():
    image = cv.imread("images/example.jpg")
    image_ycrcb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)
    image_ycrcb_float = image_ycrcb / 255.0

    transformation_matrix = np.array([
        [0.229, 0.587, 0.114],  # Y (luminance channel)
        [0.5, -0.418, -0.082],  # Cr (chrominance channel)
        [-0.168, -0.331, 0.5]   # Cb (chrominance channel)
    ])

    add_vector = np.array([0, 128/255, 128/255]) 

    pixels = image_ycrcb_float.reshape(-1, 3)

    transformed_pixels = pixels.dot(transformation_matrix.T)
    transformed_pixels += add_vector
    transformed_pixels = np.clip(transformed_pixels, 0, 1)
    transformed_image_ycrcb = transformed_pixels.reshape(image_ycrcb.shape)
    transformed_image_rgb = cv.cvtColor((transformed_image_ycrcb * 255).astype(np.uint8), cv.COLOR_YCrCb2RGB)
    Y_channel = transformed_image_ycrcb[:, :, 0]
    Cr_channel = transformed_image_ycrcb[:, :, 1]
    Cb_channel = transformed_image_ycrcb[:, :, 2]
    plt.figure(figsize=(18, 12))
    plt.subplot(2, 3, 1)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title("Originalny obraz")
    plt.axis('off')

    # Y channel (in grayscale)
    plt.subplot(2, 3, 2)
    plt.imshow(Y_channel, cmap='gray')
    plt.title("Y Channel (Luminance)")
    plt.axis('off')

    # Cr channel (in grayscale)
    plt.subplot(2, 3, 3)
    plt.imshow(Cr_channel, cmap='gray')
    plt.title("Cr Channel")
    plt.axis('off')

    # Cb channel (in grayscale)
    plt.subplot(2, 3, 4)
    plt.imshow(Cb_channel, cmap='gray')
    plt.title("Cb Channel")
    plt.axis('off')

    # Image after reversed conversion (back to RGB)
    plt.subplot(2, 3, 6)
    plt.imshow(transformed_image_rgb)
    plt.title("Obraz po konwersji odwrotnej")
    plt.axis('off')

    # Show the plot with all images
    plt.tight_layout()
    plt.show()
def menu():
    options = ["zadanie 1", "zadanie 2", "zadanie 3"]
    
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
            if 1 <= selected_option <= 3:
                # Call the corresponding function based on the selection
                if selected_option == 1:
                    zadanie1()
                elif selected_option == 2:
                    zadanie2()
                elif selected_option == 3:
                    zadanie3()
                break  # Exit the loop if a valid option is selected
            else:
                print("Invalid selection. Please choose a number between 1 and 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Call the function to test it
menu()


#zadanie1()
#zadanie2()
#zadanie3()