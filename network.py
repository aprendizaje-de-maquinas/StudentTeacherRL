from torch import nn
import torch
import torch.nn.functional as F
from torchvision.models import resnet

class Net(resnet.ResNet):

    def __init__(self, sate_dim, action_dim):

        super(Net, self).__init__(resnet.BasicBlock, [2,2,2,2], num_classes=action_dim)
        self.cuda()
        self.distribution = torch.distributions.Categorical

        self.apply(self.init)

    def init(self, m):

        try:
            nn.init.normal(m.weight, mean=0.0, std=0.1)
            nn.init.constant(m.bias, 0.1)
        except:
            pass
            #print(type(m))

    def choose_action(self, state):
        self.eval()
        logits, _ = self.forward(state)
        probs = F.softmax(logits, dim=1).data

        print('act')
        return self.distribution(probs).sample ().numpy()[0]

    def loss_func(self, state, action, value):
        self.train()
        logits, values = self.forward(state)
        td = value - values
        value_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)

        action_loss = -self.distribution(probs).log_prob(action) * td.detach()

        total_loss = (value_loss + action_loss).mean()
        return total_loss

'''
class Net(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Net, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy1 = nn.Linear(state_dim, 100)
        self.policy2 = nn.Linear(100, action_dim)
        self.value1 = nn.Linear(state_dim, 100)
        self.value2 = nn.Linear(100, 1)
        self.init([self.policy1, self.policy2, self.value1, self.value2])
        self.distribution = torch.distributions.Categorical

    def init(self, layers):
        for layer in layers:
            nn.init.normal(layer.weight, mean=0.0, std=0.1)
            nn.init.constant(layer.bias, 0.1)

    def forward(self, x):
        p1 = F.relu(self.policy1(x))
        logits = self.policy2(p1)
        v1 = F.relu(self.value1(x))
        values = self.value2(v1)

        return logits, values

    def choose_action(self, state):
        self.eval()
        logits, _ = self.forward(state)
        probs = F.softmax(logits, dim=1).data
        return self.distribution(probs).sample ().numpy()[0]

    def loss_func(self, state, action, value):
        self.train()
        logits, values = self.forward(state)
        td = value - values
        value_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)

        action_loss = -self.distribution(probs).log_prob(action) * td.detach()

        total_loss = (value_loss + action_loss).mean()
        return total_loss
'''
