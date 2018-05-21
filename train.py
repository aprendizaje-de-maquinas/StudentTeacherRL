from GA3C import GA3C
#from a3c import A3C
from network import Net
import AngryBirds ### pylint: disable=unused-import


if __name__ == '__main__':

    agent = GA3C.GA3C()
    #agent = A3C('AngryBirds-v1', Net, 40000, 10, 0.9, 0.0001)
    #agent = A3C('CartPole-v0', Net, 40000, 10, 0.9, 0.0001)

    agent.train()
    #agent.disply_graph()
