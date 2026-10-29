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