from ple import PLE
import gym
from pygame import K_f, K_r
import AngryBirds
import pygame

p = gym.make('AngryBirds-v0')

p.reset()
p.render()


#script = [ "forward", "forward", "forward", "forward", "forward", "run", "run", "run", "run", "run" ]
script = [ 'forward' , 'left' , 'right', 'forward', 'run']


actions = {
    "forward": 0,
    "run": 1,
    "right": 2,
    "left": 3
}


for f in range(len(script)):
    
    # state and reward will be none for all actions but run
    # reward will be 1 if task success
    state, reward, _, _ = p.step(actions[script[f]])
    if state is not None:
        #print(state)
        print(reward)
    p.render()


p.reset()
pygame.display.quit()
pygame.quit()


