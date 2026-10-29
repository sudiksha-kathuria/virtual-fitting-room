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

def conv2d_multi(image,kernels): 
    #for coloured images, the previous is the base cuz its for black and white
    total_filters = kernels.shape[0]
    total_channels = image.shape[0]
    img_h = image.shape[1]
    img_w = image.shape[2]
    kernels_h = kernels.shape[2]
    kernels_w = kernels.shape[3]
    output_h = img_h - kernels_h + 1
    output_w = img_w - kernels_w + 1
    output = np.zeros((total_filters, output_h, output_w))
    for f in range(total_filters):
        for c in range(total_channels):
            output[f] = output[f] + conv2d(image[c], kernels[f,c])
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

def read_image():
    channels = int(input("Enter the number of channels: "))
    sheets = [read_grid(f"channel {i}") for i in range(channels)]
    return np.stack(sheets)

def test_single():
    image = read_grid("image")
    kernel = read_grid("kernel")
    padding = int(input("Padding: "))
    stride = int(input("Stride: "))
    result = conv2d(image,kernel,padding, stride)
    print(result)
    print(result.shape)

def test_multi():
    base = np.array( [
        [0,0,9,9,9],
        [0,0,9,9,9],
        [0,0,9,9,9],
        [0,0,9,9,9],
        [0,0,9,9,9]
    ])
    image = np.stack([base,base,base])
    vertical = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    horizontal = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    kernels = np.stack([
        np.stack([vertical,vertical,vertical]),
        np.stack([horizontal,horizontal,horizontal])
    ])
    result = conv2d_multi(image, kernels)
    print(result.shape)
    print(result)

def relu(x):
    return np.maximum(x,0)

#def test_relu_simple():
#    print(relu(np.array([[-2, 5], [0, -7]])))

if __name__ == "__main__":
    test_multi()