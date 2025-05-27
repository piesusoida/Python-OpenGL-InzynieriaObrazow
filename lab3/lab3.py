"""Function definitions that are used in LSB steganography."""
from matplotlib import pyplot as plt
import numpy as np
import binascii
import cv2 as cv
import math
import lorem
plt.rcParams["figure.figsize"] = (18,10)


def encode_as_binary_array(msg):
    """Encode a message as a binary string."""
    msg = msg.encode("utf-8")
    msg = msg.hex()
    msg = [msg[i:i + 2] for i in range(0, len(msg), 2)]
    msg = [ "{:08b}".format(int(el, base=16)) for el in msg]
    return "".join(msg)


def decode_from_binary_array(array):
    """Decode a binary string to utf8."""
    array = [array[i:i+8] for i in range(0, len(array), 8)]
    if len(array[-1]) != 8:
        array[-1] = array[-1] + "0" * (8 - len(array[-1]))
    array = [ "{:02x}".format(int(el, 2)) for el in array]
    array = "".join(array)
    result = binascii.unhexlify(array)
    return result.decode("utf-8", errors="replace")


def load_image(path, pad=False):
    """Load an image.
    
    If pad is set then pad an image to multiple of 8 pixels.
    """
    image = cv.imread(path)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    if pad:
        y_pad = 8 - (image.shape[0] % 8)
        x_pad = 8 - (image.shape[1] % 8)
        image = np.pad(
            image, ((0, y_pad), (0, x_pad) ,(0, 0)), mode='constant')
    return image


def save_image(path, image):
    """Save an image."""
    plt.imsave(path, image) 


def clamp(n, minn, maxn):
    """Clamp the n value to be in range (minn, maxn)."""
    return max(min(maxn, n), minn)

def hide_image(image, secret_image_path, nbits=1):
    with open(secret_image_path, "rb") as file:
        secret_img = file.read()
        
    secret_img = secret_img.hex()
    secret_img = [secret_img[i:i + 2] for i in range(0, len(secret_img), 2)]
    secret_img = ["{:08b}".format(int(el, base=16)) for el in secret_img]
    secret_img = "".join(secret_img)
    return hide_message(image, secret_img, nbits), len(secret_img)

def hide_message(image, message, nbits=1, spos=0):
    """Hide a message in an image (LSB).
    
    nbits: number of least significant bits
    """
 
        
    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()
    # do pos niech bedzie bez zmian
    image_start = image[:spos] 
    # dodajemy napis po pos
    image = image[spos:]
    
    if len(message) > len(image) * nbits + spos:
        raise ValueError("Message is to long :(")
    
    chunks = [message[i:i + nbits] for i in range(0, len(message), nbits)]
    for i, chunk in enumerate(chunks):
        byte = "{:08b}".format(image[i])
        new_byte = byte[:-nbits] + chunk
        image[i] = int(new_byte, 2)
    #laczenie poczatku z zmienionym fragmentem
    image = np.concatenate((image_start, image ), axis=None)
    return image.reshape(shape)



def reveal_message(image, nbits=1, length=0, spos=0):
    """Reveal the hidden message.
    
    nbits: number of least significant bits
    length: length of the message in bits.
    """
    
    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()[spos:]
    #length_in_pixels = math.ceil(length-spos*8/nbits)
    length_in_pixels = math.ceil(length/nbits)
    if len(image) < length_in_pixels or length_in_pixels <= 0:
        length_in_pixels = len(image)
    
    message = ""
    i = 0
    while i < length_in_pixels:
        byte = "{:08b}".format(image[i])
        message += byte[-nbits:]
        i += 1
        
    mod = length % -nbits
    if mod != 0:
        message = message[:mod]
    return message

def hide_image(image, secret_image_path, nbits=1):
    with open(secret_image_path, "rb") as file:
        secret_img = file.read()
        
    secret_img = secret_img.hex()
    secret_img = [secret_img[i:i + 2] for i in range(0, len(secret_img), 2)]
    secret_img = ["{:08b}".format(int(el, base=16)) for el in secret_img]
    secret_img = "".join(secret_img)
    return hide_message(image, secret_img, nbits), len(secret_img)


def reveal_image(image, lenght, nbits=1):
    # pobranie obrazu w postaci napizu z obrazu
    message = reveal_message(image, length=lenght, nbits=nbits)
    # przedstawienie danych w postaci tablicy bajtow w postaci bitow
    message = [message[i:i + 8] for i in range(0, len(message), 8)]
    #konwersja na liczbe 
    message = [int(element, 2) for element in message]
    #Stworzenie bajtow z liczb
    message = bytes(message)

    with open("images/zad4.jpg", "wb") as file:
        file.write(message)
    return load_image("images/zad4.jpg")

def reveal_message5(image, nbits=1):
    # pobranie obrazu w postaci napisu z obrazu
    message = reveal_message(image,  nbits=nbits)
    # przedstawienie danych w postaci tablicy bajtow w postaci bitow
    message = [message[i:i + 8] for i in range(0, len(message), 8)]
    #konwersja na liczbe 
    message = [int(element, 2) for element in message]
    #Stworzenie bajtow z liczb
    message = bytes(message)

    with open("images/zad5.jpg", "wb") as file:
        file.write(message)
    return load_image("images/zad5.jpg")

def reveal_image5(image, nbits=1, ):
    """Reveal the hidden message.
    
    nbits: number of least significant bits
    length: length of the message in bits.
    """
    
    nbits = clamp(nbits, 1, 8)
    shape = image.shape
    image = np.copy(image).flatten()
    
    # kodowanie eoi
    jpg_eoi = bin(255)[2:].zfill(8)+bin(217)[2:].zfill(0)
    print(jpg_eoi)
    message = ""
    i = 0
    while True:
        byte = "{:08b}".format(image[i])
        message += byte[-nbits:]
        # rozpatrujemy czy znalezlismy stopke jpg'a jak tak to konczymy
        if message.endswith(jpg_eoi):
            break
        i += 1
        
        
    mod = i % -nbits
    if mod != 0:
        message = message[:mod]
    return message

def testy():

    image = load_image("images/rembrandt.png")
    image_with_secret, length_of_secret = hide_image(image, "images/spanish.jpg", 1)
    plt.imshow(image_with_secret)
    plt.show()

    original_image = load_image("images/rembrandt.png")  # Wczytanie obrazka
    # Mnożenie stringów działa jak zwielokratnianie
    message = "Czasie największej rozkoszy w życiu mego narodu - od otwarcia sklepów do ich zamknięcia!" * 1
    n = 1  # liczba najmłodszych bitów używanych do ukrycia wiadomości

    message = encode_as_binary_array(message)  # Zakodowanie wiadomości jako ciąg 0 i 1
    image_with_message = hide_message(original_image, message, n)  # Ukrycie wiadomości w obrazku

    save_image("images/image_with_message.png", image_with_message)  # Zapisanie obrazka w formacie PNG
    #save_image("images/image_with_message.jpg", image_with_message)  # Zapisanie obrazka w formacie JPG

    image_with_message_png = load_image("images/image_with_message.png")  # Wczytanie obrazka PNG
    #image_with_message_jpg = load_image("images/image_with_message.jpg")  # Wczytanie obrazka JPG

    secret_message_png = decode_from_binary_array(
        reveal_message(image_with_message_png, nbits=n, length=len(message)))  # Odczytanie ukrytej wiadomości z PNG
    #secret_message_jpg = decode_from_binary_array(
    #    reveal_message(image_with_message_jpg, nbits=n, length=len(message)))  # Odczytanie ukrytej wiadomości z JPG

    print(secret_message_png)
    #print(secret_message_jpg)

    # Wyświetlenie obrazków
    f, ar = plt.subplots(2,2)
    ar[0,0].imshow(original_image)
    ar[0,0].set_title("Original image")
    ar[0,1].imshow(image_with_message)
    ar[0,1].set_title("Image with message")
    ar[1,0].imshow(image_with_message_png)
    ar[1,0].set_title("PNG image")
    #ar[1,1].imshow(image_with_message_jpg)
    #ar[1,1].set_title("JPG image")

    # message = "Moja tajna wiadomość"
    # binary = encode_as_binary_array(message)
    # print("Binary:", binary)
    # message = decode_from_binary_array(binary)
    # print("Retrieved message:", message)
def zadanie1():
    original_image = load_image("images/rembrandt.png")  # Wczytanie obrazka
    # Mnożenie stringów działa jak zwielokratnianie
    message = "Totalnie ukryta wiadomosc, nie do wykrycia" * 1
    n = 1  # liczba najmłodszych bitów używanych do ukrycia wiadomości

    message = encode_as_binary_array(message)  # Zakodowanie wiadomości jako ciąg 0 i 1
    image_with_message = hide_message(original_image, message, n)  # Ukrycie wiadomości w obrazku

    save_image("images/image_with_message.png", image_with_message)  # Zapisanie obrazka w formacie PNG

    image_with_message_png = load_image("images/image_with_message.png")  # Wczytanie obrazka PNG

    secret_message_png = decode_from_binary_array(
        reveal_message(image_with_message_png, nbits=n, length=len(message)))  # Odczytanie ukrytej wiadomości z PNG

    print(secret_message_png)
    # Wyświetlenie obrazków
    f, ar = plt.subplots(2)
    ar[0].imshow(original_image)
    ar[0].set_title("Original image")
    ar[1].imshow(image_with_message)
    ar[1].set_title("Image with message")
    plt.show()

def zadanie2():
    inquisition_org = load_image("images/rembrandt.png") 
    message ="Jas jadl fasole"*int(0.75*inquisition_org.shape[0]*inquisition_org.shape[1]/30)

    message_bin = encode_as_binary_array(message)
    #message_bin = message_bin*int(len(inquisition_org)*0.75/len(message_bin))
    bins = 8
    # dodanie wiadomosci do kazdego z obrazkow zwiekszajac blad
    message_hidden_images = [ hide_message(inquisition_org, message_bin, nbits=i) for i in range(1,bins+1) ]

    fig = plt.figure(figsize=(40, 60))
    for i in range(bins):
        fig.add_subplot(4, 2, i+1)
        plt.imshow(message_hidden_images[i])
    # wyliczanie bledow MSE dla obrazka
    errors = [ ((inquisition_org - message_hidden_images[i]  )**2 ).mean(axis=None)  for i in range(bins)  ]
    x_axis = [i for i in range(1, bins+1)]


    figurete, ax = plt.subplots(1,1)
    ax.plot(x_axis, errors)
    ax.title.set_text('wykres MSE od wartosci nbits')
    ax.set_xlabel('nbits')
    ax.set_ylabel('MSE')
    plt.show()
    print(errors)

def zadanie3():
    # lets see if it runs 
    original_image = load_image("images/spanish.png")  # Wczytanie obrazka
    # Mnożenie stringów działa jak zwielokratnianie
    message = "Nikt nic nie wie" * 1
    n = 1  # liczba najmłodszych bitów używanych do ukrycia wiadomości
    position = 2
    message = encode_as_binary_array(message)  # Zakodowanie wiadomości jako ciąg 0 i 1
    image_with_message = hide_message(original_image, message, nbits=n, spos=position)  # Ukrycie wiadomości w obrazku

    secret_message_png = decode_from_binary_array(
        reveal_message(image_with_message, nbits=n, length=len(message),spos=position))  # Odczytanie ukrytej wiadomości z PNG

    print(secret_message_png)

    plt.imshow(image_with_message)
    plt.show()

def zadanie4():
    original_image = load_image("images/rembrandt.png")
    image_secret, lenght_secret = hide_image(original_image, 'images/spanish.jpg', 5)
    plt.imshow(image_secret)
    plt.title("obraz z sekretnym obrazem")
    plt.show()
    image = reveal_image(image_secret, lenght_secret, 5)
    plt.imshow(image)
    plt.title("schowany obraz")
    plt.show()

def zadanie5():
    original_image = load_image("images/rembrandt.png")
    image_secret, lenght_secret = hide_image(original_image, 'images/spanish.jpg', 1)
    image = reveal_message5(image_secret,  1)
    plt.imshow(image)
    plt.show()


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