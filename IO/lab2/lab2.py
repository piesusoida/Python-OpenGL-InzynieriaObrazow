import cv2
from matplotlib import pyplot
import numpy
import struct
import zlib
from scipy.fftpack import dct
from scipy.fftpack import idct

#
# 2d Discrete Cosinus Transform
#
def dct2(array):
    return dct(dct(array, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(array):
    return idct(idct(array, axis=0, norm='ortho'), axis=1, norm='ortho')


#
# Calculate quantisation matrices
#
# Based on: https://www.hdm-stuttgart.de/~maucher/Python/MMCodecs/html/jpegUpToQuant.html
#           #step-3-and-4-discrete-cosinus-transform-and-quantisation
#
_QY = numpy.array([[16, 11, 10, 16, 24, 40, 51, 61],
                   [12, 12, 14, 19, 26, 48, 60, 55],
                   [14, 13, 16, 24, 40, 57, 69, 56],
                   [14, 17, 22, 29, 51, 87, 80, 62],
                   [18, 22, 37, 56, 68, 109, 103, 77],
                   [24, 35, 55, 64, 81, 104, 113, 92],
                   [49, 64, 78, 87, 103, 121, 120, 101],
                   [72, 92, 95, 98, 112, 100, 103, 99]])

_QC = numpy.array([[17, 18, 24, 47, 99, 99, 99, 99],
                   [18, 21, 26, 66, 99, 99, 99, 99],
                   [24, 26, 56, 99, 99, 99, 99, 99],
                   [47, 66, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99],
                   [99, 99, 99, 99, 99, 99, 99, 99]])


def _scale(QF):
    if QF < 50 and QF >= 1:
        scale = numpy.floor(5000 / QF)
    elif QF < 100:
        scale = 200 - 2 * QF
    else:
        raise ValueError('Quality Factor must be in the range [1..99]')

    scale = scale / 100.0
    return scale


def QY(QF=85):
    return _QY * _scale(QF)


def QC(QF=85):
    return _QC * _scale(QF)

def zadanie1(): #PPM
    
    # PPM file header
    #
    ppm_ascii_header = 'P3 2 3 255 '  # TODO: implement
    ppm_binary_header = 'P6 2 3 255 '  # TODO: implement

    #
    # Image data
    #
    image = numpy.array([[0, 255, 100,   0, 255, 200,   100, 255, 0,   200, 255, 0,   50, 255, 50,   100, 255, 100]], dtype=numpy.uint8)  # TODO: implement

    #
    # Save the PPM image as an ASCII file
    #
    with open('lab4-ascii.ppm', 'w') as fh:
        fh.write(ppm_ascii_header)
        image.tofile(fh, sep=' ')
        fh.write('\n')

    #
    # Save the PPM image as a binary file
    #
    with open('lab4-binary.ppm', 'wb') as fh:
        fh.write(bytearray(ppm_binary_header, 'ascii'))
        image.tofile(fh)
    
    #
    # Display image
    #
    image_from_file = cv2.imread('lab4-ascii.ppm')
    pyplot.imshow(cv2.cvtColor(image_from_file, cv2.COLOR_BGR2RGB))
    pyplot.show() 

    # Display image
    image_from_file = cv2.imread('lab4-binary.ppm')
    pyplot.imshow(cv2.cvtColor(image_from_file, cv2.COLOR_BGR2RGB)) 
    pyplot.show() 


def zadanie2():    
    # PPM file header
    ppm_ascii_header = 'P3 224 32 255 '
    #
    # Image data
    #
    
    

    image = numpy.linspace( [0, 0, 0], [0, 0, 255], 32, dtype=numpy.uint8, endpoint=False)
    image = numpy.concatenate(
        (image, numpy.linspace( [0, 0, 255], [0, 255, 255], 32, dtype=numpy.uint8, endpoint=False)))
    image = numpy.concatenate(
        (image, numpy.linspace( [0, 255, 255], [0, 255, 0], 32, dtype=numpy.uint8, endpoint=False)))
    image = numpy.concatenate(
        (image, numpy.linspace( [0, 255, 0], [255, 255, 0], 32, dtype=numpy.uint8, endpoint=False)))
    image = numpy.concatenate(
        (image, numpy.linspace( [255, 255, 0], [255, 0, 0], 32, dtype=numpy.uint8, endpoint=False)))
    image = numpy.concatenate(
        (image, numpy.linspace( [255, 0, 0], [255, 0, 255], 32, dtype=numpy.uint8, endpoint=False)))
    image = numpy.concatenate(
        (image, numpy.linspace( [255, 0, 255], [255, 255, 255], 32, dtype=numpy.uint8)))

    #
    # Save the PPM image as an ASCII file
    #
    with open('lab4-ascii.ppm', 'w') as fh:
     fh.write(ppm_ascii_header)
     for i in range(32):
         image.tofile(fh, sep=' ')
         fh.write('\n')

    #
    # Save the PPM image as a binary file
    #
    
    #
    # Display image
    #
    image_from_file = cv2.imread('lab4-ascii.ppm')
    pyplot.imshow(cv2.cvtColor(image_from_file, cv2.COLOR_BGR2RGB))
    pyplot.show() 

def zadanie3():
    #
    # Image data
    #
    image_from_file = cv2.imread('lab4-ascii.ppm')
    image =numpy.copy(image_from_file)
    height, width, _ = image.shape

    #
    # Construct signature
    #
    png_file_signature = struct.pack('BBBBBBBB', 137, 80, 78, 71, 13, 10, 26, 10)  # TODO: implement

    #
    # Construct header
    #
    header_id = b'IHDR' # TODO: implement
    header_content = struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0) # TODO: implement
    header_size = struct.pack('BBBB', 0, 0, 0, 13)  # TODO: implement
    header_crc = struct.pack('>I', zlib.crc32(header_id + header_content))  # TODO: implement
    png_file_header = header_size + header_id + header_content + header_crc

    #
    # Construct data
    #
    compress_image = zlib.compress(b''.join([b'\x00' + bytes(row) for row in image])) # zmienna pomocnicza
    



    data_id = b'IDAT'  # TODO: implement
    data_content =  compress_image # TODO: implement   zlib.compress(b''.join([b'\x00' + bytes(row) for row in image])
    data_size = struct.pack('>I', (len(compress_image)))  # TODO: implement
    data_crc = struct.pack('>I', (zlib.crc32(data_id + data_content) ))  # TODO: implement
    png_file_data = data_size + data_id + data_content + data_crc

    #
    # Consruct end
    #
    end_id = b'IEND'
    end_content = b''
    end_size = struct.pack('!I', len(end_content))
    end_crc = struct.pack('!I', zlib.crc32(end_id + end_content))
    png_file_end = end_size + end_id + end_content + end_crc

    #
    # Save the PNG image as a binary file
    #
    with open('lab4.png', 'wb') as fh:
        fh.write(png_file_signature)
        fh.write(png_file_header)
        fh.write(png_file_data)
        fh.write(png_file_end)
    # Display image
    #
    image_from_file = cv2.imread('lab4.png')
    pyplot.imshow(image_from_file)
    pyplot.show()


def convert_8x8_to_channel(blocks, width):
    step = int(width / 8)
    rows = []
    for i in range(0, len(blocks), step):
        rows.append(numpy.concatenate( blocks[i:i+step], axis=1))
    channel = numpy.concatenate(rows, axis=0)
    return channel
def zig_zag(block_of_image) -> list:
    # setup of helper variables
    n = x = y = 0
    zig_zag_vector = numpy.zeros(64, dtype=numpy.uint8)
    # main loop 
    while n < 64 :
        #moving left
        while x > -1 and y <8:
            zig_zag_vector[n] = block_of_image[x , y]
            x-=1
            y+=1
            n+=1
        x+=1
        if y == 8:
            y-=1
            x+=1
        #moving right
        while y > -1 and x <8:
            zig_zag_vector[n] = block_of_image[x, y]
            y-=1
            x+=1
            n+=1
        y+=1
        if x == 8:
            x-=1 
            y+=1
    return zig_zag_vector
def downsample(image_data, sample_rate=2):
    CR = image_data[0::sample_rate,0::sample_rate,1]
    CB = image_data[0::sample_rate,0::sample_rate,2]
    Y  = image_data[0::,0::,0]
    return CR, CB, Y
def produce_blocks( channel_CR, channel_CB):
    CR_blocks = []
    CB_blocks = []
    for i in range(0, channel_CR.shape[0], 8):
        for j in range(0, channel_CR.shape[1], 8):
            CR_blocks.append(channel_CR[i:i + 8, j:j + 8])
            CB_blocks.append(channel_CB[i:i + 8, j:j + 8])
    return CR_blocks, CB_blocks 
def compress(Y, CR, CB):
    concat = numpy.concatenate( (Y.flatten(), CR.flatten(), CB.flatten()), axis=0 )
    compress = zlib.compress(concat)
    return len(str(compress))
def zadanie4():
    #user-defined functions


    #
    # 0. Image data
    #
    # DONE TODO: implement (zad. 4)
    image_from_file = cv2.imread('lab4-ascii.ppm')
    image_data =numpy.copy(image_from_file)


    #
    # 1. Convert RGB to YCbCr
    #
    # DONE TODO: implement (zad. 4)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2YCrCb)

    width = image_data.shape[1]
    height =  image_data.shape[0]
    xxxx = image_data.shape[2]

    blocks = (width * height) // 64
    blocks_float = (width * height) / 64
    if (blocks != blocks_float):
        height_pad = height / 8 - height // 8
        width_pad = width / 8 - width // 8
        if( height_pad >0):
            height_pad = 1 - height_pad
        if( width_pad >0):
            width_pad = 1 - width_pad
        print( height_pad* 8)
        print (width_pad * 8)
        blocks+= 1
        padded_image_array = numpy.pad(image_data, ((int(height_pad * 8), 0), (0, int(width_pad * 8)), (0, 0)), mode='constant', constant_values=(0))
        image_data = numpy.copy(padded_image_array)

    #
    # 2. Downsampling on Cb and Cr channels
    #
    CR_0, CB_0, Y = downsample(image_data, sample_rate=1)
    CR_2, CB_2, Y = downsample(image_data, sample_rate=2)
    CR_4, CB_4, Y = downsample(image_data, sample_rate=4)


    height_0, width_0 = CR_0.shape
    height_2, width_2 = CR_2.shape
    height_4, width_4 = CR_4.shape
    #
    # 3. Produce 8x8 blocks
    #
    # ??? TODO: implement (zad. 4)
    y_blocks= []
    for i in range(0, Y.shape[0], 8):
        for j in range(0, Y.shape[1], 8):
            y_blocks.append(  Y[i:i + 8, j:j + 8])
    
    CR_0_blocks, CB_0_blocks = produce_blocks(CR_0, CB_0)

    CR_2_blocks, CB_2_blocks= produce_blocks(CR_2, CB_2)

    CR_4_blocks, CB_4_blocks= produce_blocks(CR_4, CB_4)



    #
    # 7. Zig-zag
    #

    Y_zig_zag_blocks =  numpy.asarray([ zig_zag(y_blocks[x]) for x in range(len(y_blocks)) ])

    CR_0_zig_zag_blocks = numpy.asarray([ zig_zag(CR_0_blocks[x]) for x in range(len(CR_0_blocks))])
    CB_0_zig_zag_blocks = numpy.asarray([ zig_zag(CB_0_blocks[x]) for x in range(len(CB_0_blocks))])

    CR_2_zig_zag_blocks = numpy.asarray([ zig_zag(CR_2_blocks[x]) for x in range(len(CR_2_blocks))])
    CB_2_zig_zag_blocks = numpy.asarray([ zig_zag(CB_2_blocks[x]) for x in range(len(CB_2_blocks))])

    CR_4_zig_zag_blocks = numpy.asarray([ zig_zag(CR_4_blocks[x]) for x in range(len(CR_4_blocks))])
    CB_4_zig_zag_blocks = numpy.asarray([ zig_zag(CB_4_blocks[x]) for x in range(len(CB_4_blocks))])




#
# 8. Flatten, concatenate, compress and calculate the size -- how many bytes?
#
# TODO: implement (zad. 4)

# flattened :D



    compress_0 =  compress(Y_zig_zag_blocks, CR_0_zig_zag_blocks, CB_0_zig_zag_blocks)
    compress_2 =  compress(Y_zig_zag_blocks, CR_2_zig_zag_blocks, CB_2_zig_zag_blocks)
    compress_4 =  compress(Y_zig_zag_blocks, CR_4_zig_zag_blocks, CB_4_zig_zag_blocks)

    print(f"Dlugosc bez probkowania: {compress_0}")
    print(f"Dlugosc z probkowaniem co 2: {compress_2}") 
    print(f"Dlugosc z probkowaniem co 4: {compress_4}")

    #
    # 7'. Undo Zig Zag
    #
    # We can skip it in this exercise! We did Zig Zag only for analysis in step 8.
    # You can continue with result from step 6. instead of implementing undo here.
    #

    #
    # 6'. Nothing to do here   ¯\_(ツ)_/¯
    #
    # No conversion is really needed here, just proceed to the next step.
    #

    #
    # 5'. Reverse division by quantisation matrix -- multiply
    #
    # TODO: implement (zad. 5)

    #
    # 4'. Reverse DCT
    #
    # TODO: implement (zad. 5)

    #
    # 3'. Combine 8x8 blocks to original image
    #
    y_channel_reforged = convert_8x8_to_channel(y_blocks, width)
    CR_2 = convert_8x8_to_channel(CR_2_blocks, width_2)
    CB_2 = convert_8x8_to_channel(CB_2_blocks, width_2)
    CR_4 = convert_8x8_to_channel(CR_4_blocks, width_4)
    CB_4 = convert_8x8_to_channel(CB_4_blocks, width_4)
    #height, width = CR_channel_reforged.shape

    #
    # 2'. Upsampling on Cb and Cr channels
    #
    image_0 = numpy.copy(image_data)
    image_0[:,:,0] = y_channel_reforged
    image_0[:,:,1] = CR_0
    image_0[:,:,2] = CB_0

    image_2 = numpy.copy(image_data)
    CR_2 = numpy.repeat(CR_2, 2, axis=1)
    CR_2 = numpy.repeat(CR_2, 2, axis=0)
    CB_2 = numpy.repeat(CB_2, 2, axis=1)
    CB_2 = numpy.repeat(CB_2, 2, axis=0)
    image_2[:,:,0] = y_channel_reforged
    image_2[:,:,1] = CR_2
    image_2[:,:,2] = CB_2

    image_4 = numpy.copy(image_data)
    CR_4 = numpy.repeat(CR_4, 4, axis=1)
    CR_4 = numpy.repeat(CR_4, 4, axis=0)
    CB_4 = numpy.repeat(CB_4, 4, axis=1)
    CB_4 = numpy.repeat(CB_4, 4, axis=0)
    image_4[:,:,0] = y_channel_reforged
    image_4[:,:,1] = CR_4
    image_4[:,:,2] = CB_4
    
    #
    # 1'. Convert YCbCr to RGB
    #
    image_0 =  cv2.cvtColor(image_0, cv2.COLOR_YCrCb2RGB)
    image_2 =  cv2.cvtColor(image_2, cv2.COLOR_YCrCb2RGB)
    image_4 =  cv2.cvtColor(image_4, cv2.COLOR_YCrCb2RGB)
    #
    # 0'. Save the decoded image -- as PPM or PNG
    #
    ppm_ascii_header = 'P3 224 32 255 '


    with open('zad4-1-1.ppm', 'w') as fh:
        fh.write(ppm_ascii_header)
        image_0.tofile(fh, sep=' ')
        fh.write('\n')
    with open('zad4-1-2.ppm', 'w') as fh:
        fh.write(ppm_ascii_header)
        image_0.tofile(fh, sep=' ')
        fh.write('\n')
    with open('zad4-1-4.ppm', 'w') as fh:
        fh.write(ppm_ascii_header)
        image_0.tofile(fh, sep=' ')
        fh.write('\n')


    image_from_file = cv2.imread('zad4-1-1.ppm')
    pyplot.imshow(cv2.cvtColor(image_from_file, cv2.COLOR_BGR2RGB))
    pyplot.title("bez probkowania")
    pyplot.show()
    image_from_file = cv2.imread('zad4-1-2.ppm')
    pyplot.imshow(cv2.cvtColor(image_from_file, cv2.COLOR_BGR2RGB))
    pyplot.title("probkowanie co drugi")
    pyplot.show()
    image_from_file = cv2.imread('zad4-1-4.ppm')
    pyplot.imshow(cv2.cvtColor(image_from_file, cv2.COLOR_BGR2RGB))
    pyplot.title("probkowanie co 4")
    pyplot.show()


def menu():
    options = ["zadanie 1", "zadanie 2", "zadanie 3", "zadanie 4"]
    
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
            if 1 <= selected_option <= 4:
                # Call the corresponding function based on the selection
                if selected_option == 1:
                    zadanie1()
                elif selected_option == 2:
                    zadanie2()
                elif selected_option == 3:
                    zadanie3()
                elif selected_option == 4:
                    zadanie4()
                break  # Exit the loop if a valid option is selected
            else:
                print("Invalid selection. Please choose a number between 1 and 4.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Call the function to test it
menu()