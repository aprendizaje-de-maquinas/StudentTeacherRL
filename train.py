from a3c import A3C
from network import Net


if __name__ == '__main__':

    agent = A3C('CartPole-v0', Net, 40000, 10, 0.9, 0.0001)

    agent.train()
    agent.disply_graph()
