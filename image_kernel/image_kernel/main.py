# This is a sample Python script.
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def main(image: str):
    img_pil = Image.open(image)
    mode =img_pil.mode
    if mode == 'P':
        img_pil=img_pil.convert('L')
    #img_pil.show()
    img = np.array(img_pil)
    #imgtmp=Image.fromarray(img,mode='L')
    #imgtmp.show()
    #exit()

    if len(img.shape) == 2:
        img = np.reshape(img,(img.shape[0],img.shape[1],1))

    print('Original Image shape {}'.format(img.shape))
    fig,axis = plt.subplots(2,2)

    fig.set_size_inches(10,10)
    plot_img(axis,(0,0),'Original',img)


    # kernel for edge detection
    kernel_edge = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    new_image = apply_convolutional(img, kernel_edge)
    print('New Image shape {}'.format(new_image.shape))
    # Create a PIL image from the new image and display it
    #sImg = Image.fromarray(new_image)
    plot_img(axis, (0, 1), 'Edge Detection',new_image)

    # kernel for vertical edge detection
    kernel_vert_edg = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])

    new_image = apply_convolutional(img, kernel_vert_edg)
    # Create a PIL image from the new image and display it
    #sImg = Image.fromarray(new_image)
    plot_img(axis, (1, 0), 'Vertical Edge Detection', new_image)

    # kernel for horizontal edge detection
    kernel_horz_edg = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])

    new_image = apply_convolutional(img, kernel_horz_edg)
    print('New Image shape {}'.format(new_image.shape))
    # Create a PIL image from the new image and display it
    #sImg = Image.fromarray(new_image)
    plot_img(axis, (1, 1), 'Horizontal Edge Detection', new_image)

    # Kernel for box blur
    #kernel_blure = np.array([[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]])

    plt.show()


def plot_img(axis,position:tuple,title,s_image):

    color = 'rgb'
    axis[position[0], position[1]].set_title(title)
    if s_image.shape[2] == 1:
        s_image= np.reshape(s_image,(s_image.shape[0],s_image.shape[1]))
        axis[position[0], position[1]].imshow(s_image,cmap='gray')
        sImg = Image.fromarray(s_image,mode='L')
        sImg.show(title=title)
    else:
        axis[position[0],position[1]].imshow(s_image)
        sImg = Image.fromarray(s_image, mode='RGB')
        sImg.show()

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

    filtered_img = np.zeros(shape=(new_height, new_width, i_c))

    l: int = 0
    # every line of matrix (stride of 1)
    for line in range(k_height // 2, (i_max_line)):
        # every column of matrix (stride of 1)
        c: int = 0
        for column in range(k_width // 2, (i_max_col)):
            window = img[line - k_height // 2:line + k_height // 2 + 1, column - k_width // 2:column + k_width // 2 + 1,:]

            filtered_img[l, c, 0] = int((window[:, :, 0] * kernel).sum())
            if img.shape[2] == 3:
                filtered_img[l, c, 1] = int((window[:, :, 1] * kernel).sum())
                filtered_img[l, c, 2] = int((window[:, :, 2] * kernel).sum())
            c += 1
        l += 1

    filtered_img = np.clip(filtered_img, 0, 255)
    return filtered_img.astype(np.uint8)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #main('cat-1.jpg')
    main('Lena_RGB.png')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
