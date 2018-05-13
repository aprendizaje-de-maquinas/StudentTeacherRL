from ple import PLE
import gym
from pygame import K_f, K_r
import AngryBirds
#from ple.games.flappybird import FlappyBird


p = gym.make('AngryBirds-v0')

p.reset()
p.render()



#game = codegame.AngryBird()
# game = FlappyBird()
#p = PLE(game, fps=30, display_screen=True, force_fps=True)
#p.init()

script = [ "add_f_block", "add_f_block", "add_f_block", "add_f_block", "run", "run", "run", "run", "run" ]

actions = {
            "add_f_block": 0,
            "run": 1
        }

for f in range(len(script)):
    #if p.game_over(): #check if game is over
    #   print("GAHHH")
    #   p.reset()

    line = input("WAITING")

    p.step(actions[script[f]])
    p.render()
    # obs = p.getScreenRGB()
    # action = myAgent.pickAction(reward, obs)
    #p.act(actions[script[f]])


