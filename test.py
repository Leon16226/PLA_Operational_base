import numpy as np

if __name__ == "__main__":
    x = 2.5
    xp = [1, 2, 3]
    fp = [3, 2, 0]
    y = np.interp(x, xp, fp)
    print(y)
