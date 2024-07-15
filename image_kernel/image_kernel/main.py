# This is a sample Python script.
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# https://medium.com/@sabribarac/implementing-image-processing-kernels-from-scratch-using-convolution-in-python-4e966e9aafaf


def main(image: str):
    img_pil = Image.open(image)
    img_pil = Image.open('/home/dpaulino/Downloads/cat-1.jpg')

    img_pil.show()
    img = np.asarray(img_pil)
    print('Original Image shape {}'.format(img.shape))
    fig,axis = plt.subplots(2,2)


    fig.set_size_inches(10,10)
    axis[0,0].set_title('Original')
    axis[0,0].imshow(img)

    # kernel for edge detection
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    new_image = apply_convolutional(img, kernel)
    print('New Image shape {}'.format(new_image.shape))
    # Create a PIL image from the new image and display it
    sImg = Image.fromarray(new_image)
    axis[0, 1].set_title('Edge Detection')
    axis[0, 1].imshow(sImg)

    # kernel for vertical edge detection
    kernel = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

    new_image = apply_convolutional(img, kernel)
    print('New Image shape {}'.format(new_image.shape))
    # Create a PIL image from the new image and display it
    sImg = Image.fromarray(new_image)
    axis[1, 0].set_title('Vertical Edge Detection')
    axis[1, 0].imshow(sImg)

    # kernel for horizontal edge detection
    kernel = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])

    new_image = apply_convolutional(img, kernel)
    print('New Image shape {}'.format(new_image.shape))
    # Create a PIL image from the new image and display it
    sImg = Image.fromarray(new_image)
    axis[1, 1].set_title('Horizontal Edge Detection')
    axis[1, 1].imshow(sImg)

    # Kernel for box blur
    #kernel = np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])


    plt.show()
    #sImg.show()


def apply_convolutional(img: np.array, kernel: np.array) -> np.array:
    # no padding stride ==1
    i_height, i_width, i_c = (img.shape[0], img.shape[1], img.shape[2])
    k_height, k_width = (kernel.shape[0], kernel.shape[1])
    i_max_line = i_height - k_height // 2
    i_max_col = i_width - k_width // 2
    # every line of matrix (stride of 1)

    stride = 1
    new_height = int(((i_height - k_height) / stride)) + 1
    new_width = int(((i_width - k_height) / stride)) + 1
    filtered_img = np.zeros(shape=(new_height, new_width, 3))

    l: int = 0
    # every line of matrix (stride of 1)
    for line in range(k_height // 2, (i_max_line)):
        # every column of matrix (stride of 1)
        c: int = 0
        for column in range(k_width // 2, (i_max_col)):
            window = img[line - k_height // 2:line + k_height // 2 + 1, column - k_width // 2:column + k_width // 2 + 1,
                     :]
            #print(img[0:3,0:3])
            #print(window[:,:,:])
            filtered_img[l, c, 0] = int((window[:, :, 0] * kernel).sum())
            filtered_img[l, c, 1] = int((window[:, :, 1] * kernel).sum())
            filtered_img[l, c, 2] = int((window[:, :, 2] * kernel).sum())
            c += 1
        l += 1

    filtered_img = np.clip(filtered_img, 0, 255)
    return filtered_img.astype(np.uint8)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #main('/home/dpaulino/Downloads/cat-1.jpg')
    main('/home/dpaulino/Downloads/Lena_RGB.png')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
