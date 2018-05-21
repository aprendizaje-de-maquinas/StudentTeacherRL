import gym
import pygame
import AngryBirds


p = gym.make('AngryBirds-v0').unwrapped #v0 has display with render
#p = gym.make('AngryBirds-v1')  #v1 has no display with render



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


for _ in range(2):
    for f in range(len(script)):
        
        # state and reward will be none for all actions but run
        # reward will be 1 if task success
        state, reward, _, _ = p.step(actions[script[f]])
        if state is not None:
            print(reward)
        p.render()


    p.reset_hard()

p.close()
pygame.display.quit()


