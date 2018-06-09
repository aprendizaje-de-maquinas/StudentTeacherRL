import gym
import pygame
import AngryBirds

p = gym.make('AngryBirds-v0').unwrapped #v0 has display with render
#p = gym.make('AngryBirds-v1')  #v1 has no display with render

p.reset()
p.render()


#script = [ "forward", "forward", "forward", "forward", "forward", "run", "run", "run", "run", "run" ]
# script = [ 'forward' , 'left' , 'right', 'forward', 'run']
script = ['while', 'ifClear', 'forward', 'end', 'run']

actions = {
    "noop": 0,
    "forward": 1,
    "run": 2,
    "right": 3,
    "left": 4,
    "while": 5,
    "ifClear": 6,
    "end": 7
}


for _ in range(10):
    for f in range(len(script)):
        
        # state and reward will be none for all actions but run
        # reward will be 1 if task success
        print(script[f])
        state, reward, _, _ = p.step(actions[script[f]])
        if state is not None:
            print('Reward', reward)
        p.render()


    p.reset_hard()

p.close()
pygame.display.quit()


