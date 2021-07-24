import torch

dimension = 4
position = 3

def get_pos(pos):
    print("wave",[torch.pow(torch.tensor(10000), 2 * (di // 2) / dimension) for di in range(dimension)])

    return [pos / torch.pow(torch.tensor(10000), 2 * (di // 2) / dimension) for di in range(dimension) ]

if __name__ == '__main2__':
    embedings = torch.randn([4*4, position, dimension])

    sinusoid = torch.tensor([get_pos(pi) for pi in range(position)])
    sinusoid[:, 0::2] = torch.sin(sinusoid[:, 0::2])
    sinusoid[:, 1::2] = torch.cos(sinusoid[:, 1::2])

    print(sinusoid)

if __name__ == '__main__':
    print(torch.tensor([1,2,3]))
