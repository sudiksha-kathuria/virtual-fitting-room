import numpy as np

def conv2d(image,kernel, padding=0, stride=1):
    if padding>0:
        image = np.pad(image , padding)
    img_h , img_w = image.shape
    kernel_h , kernel_w = kernel.shape
    output_h = (img_h - kernel_h) // stride + 1
    output_w = (img_w - kernel_w) // stride + 1
    output = np.zeros((output_h , output_w))
    for row in range(output_h):
        for col in range(output_w):
            start_row = row * stride
            start_col = col * stride
            patch = image[start_row: start_row + kernel_h , start_col: start_col + kernel_w]
            output[row , col] = np.sum(patch * kernel)
    return output

def read_grid(name):
    print(f"Enter {name}, one row at a time with numbers separated by spaces")
    print("Press Enter when one row is done")
    rows=[]
    while True:
        line = input()
        if line == "":
            break
        rows.append([float(x) for x in line.split()])
    return np.array(rows)

if __name__ == "__main__":
    image = read_grid("image")
    kernel = read_grid("kernel")
    padding = int(input("Padding: "))
    stride = int(input("Stride: "))
    result = conv2d(image,kernel,padding, stride)
    print(result)
    print(result.shape)