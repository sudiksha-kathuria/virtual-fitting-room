from foundations.conv2d import conv2d
from tests.helper import read_grid

def main():
    image = read_grid("image")
    kernel = read_grid("kernel")
    padding = int(input("Padding: "))
    stride = int(input("Stride: "))
    result = conv2d(image, kernel, padding, stride)
    print(result)
    print(result.shape)

if __name__ == "__main__":
    main()