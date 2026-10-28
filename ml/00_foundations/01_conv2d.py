import numpy as np;

def conv2d(image,kernel):
    img_h , img_w = image.shape
    kernel_h , kernel_w = kernel.shape
    output_h = img_h - kernel_h + 1
    output_w = img_w - kernel_w + 1
    output = np.zeros((output_h , output_w))
    for row in range(output_h):
        for col in range(output_w):
            patch = image[row: row + kernel_h , col: col + kernel_w]
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
    result = conv2d(image,kernel)
    print(result)