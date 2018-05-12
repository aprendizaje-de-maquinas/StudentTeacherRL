from ple import PLE
from codegame import *
from ple.games.flappybird import FlappyBird

game = AngryBird()
# game = FlappyBird()
p = PLE(game, fps=30, display_screen=True, force_fps=True)
p.init()

script = [ "add_f_block", "add_f_block", "add_f_block", "add_f_block", "run", "run", "run", "run", "run" ]

actions = {
            "add_f_block": K_f,
            "run": K_r
        }

for f in range(len(script)):
    if p.game_over(): #check if game is over
        print("GAHHH")
        p.reset_game()
    
    # obs = p.getScreenRGB()
    # action = myAgent.pickAction(reward, obs)
    p.act(actions[script[f]])
    line = input("WAITING")
