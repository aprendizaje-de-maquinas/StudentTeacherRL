from ple import PLE
import gym
from pygame import K_f, K_r
import AngryBirds

p = gym.make('AngryBirds-v0')

p.reset()
p.render()


script = [ "add_f_block", "add_f_block", "add_f_block", "add_f_block", "run", "run", "run", "run", "run" ]

actions = {
            "add_f_block": 0,
            "run": 1
        }


for _ in range(100):
    for f in range(len(script)):

        p.step(actions[script[f]])
        if(actions[script[f]] == 'run'):
            p.reset()

        p.render()

    p.reset()


