'''
Angry birds type game for rl program creation
'''

import pygame
from ple.games import base
from blocks.blocks import Position, Direction, ForwardBlock, TurnLeft, TurnRight
from pygame.constants import K_r, K_f, K_a, K_l, KEYDOWN, NOFRAME, QUIT
import os
import sys
import gym
from gym import spaces
from ple import PLE
import numpy as np
import time

from AngryBirds.bird_envs import SimpleEnv

class BirdPlayer(pygame.sprite.Sprite):

    # The bird only needs to know its current plan and position
    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT, init_pos, image, world, render=True):
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        self.plan = []
        self.render = render

        pygame.sprite.Sprite.__init__(self)

        self.image = image

        self.game_tick = 0
        self.world = world

        self.position = Position(init_pos[0], init_pos[1], Direction(Direction.RIGHT))
        self.orientation = Direction.RIGHT

        self.executingProgram = False
        self.finished = False
        self.died = False

    # Adds a block to the current queue of blocks
    def addBlock(self, block):
        if not self.executingProgram:
            self.plan.append(block)

    def executeProgram(self):
        if not self.finished:
            self.executingProgram = True

    def update(self, _):
        self.game_tick += 1
        if self.executingProgram:

            if len(self.plan) == 0:
                return
            
            self.plan.pop(0)(self)
            
            if self.world[self.position.x][self.position.y] == 'W':
                self.died = True
            if len(self.plan) == 0: #self.position.out_of_bounds():
                self.executingProgram = False
                self.plan = []
                self.finished = True
                



    def draw(self, screen):
        screen.blit(pygame.transform.rotate(self.image, 90 * (self.position.orientation.v - 3)), \
                    (self.SCREEN_WIDTH / 8 * self.position.y, self.SCREEN_HEIGHT / 8 * self.position.x))

class Block(pygame.sprite.Sprite):

    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT, init_pos, image):

        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        pygame.sprite.Sprite.__init__(self)

        self.pos_x = init_pos[0]
        self.pos_y = init_pos[1]

        self.image = image
        self.width = self.image.get_width()

    def update(self, dt):
        pass

class Pig(pygame.sprite.Sprite):

    def __init__(self,
                 SCREEN_WIDTH, SCREEN_HEIGHT, init_pos, image):

        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        self.image = image
        self.width = self.image.get_width()
        pygame.sprite.Sprite.__init__(self)

        self.pos_x = init_pos[0]
        self.pos_y = init_pos[1]

    def update(self, dt):
        pass

class Backdrop():
    def __init__(self, SCREEN_WIDTH, SCREEN_HEIGHT,
                 image_assets, map):
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        self.background_image = image_assets["background"]
        self.wood_image = image_assets["wood"]
        self.pig_image = image_assets["pig"]
        self.map = map


    def draw_background(self, screen):
        screen.blit(self.background_image, (0, 0))
    
    def update_draw_base(self, screen, _):
        for i in range(len(self.map)):
            for j in range(len(self.map[i])):
                if self.map[i][j] == "W":
                    screen.blit(self.wood_image, (self.SCREEN_WIDTH / 8 * j, self.SCREEN_HEIGHT / 8 * i))
                elif self.map[i][j] == "P":
                    screen.blit(self.pig_image, (self.SCREEN_WIDTH / 8 * j, self.SCREEN_HEIGHT / 8 * i))


class AngryBird(base.PyGameWrapper):

    def __init__(self, width=256, height=256, num_blocks=8, render=True):
        
        actions = {
            "forward": K_f,
            "run": K_r,
            "right": K_a,
            "left": K_l
        }
        self.actions = actions
        self.num_blocks = num_blocks

        fps = 30
        self.lives = -1

        base.PyGameWrapper.__init__(self, width, height, actions=actions)

        self.scale = 30.0 / fps

        self.allowed_fps = 30  # restrict the fps

        self.images = {}

        # so we can preload images
        pygame.display.set_mode((1, 1), NOFRAME)

        self.backdrop = None
        self.player = None
        self.pig_pos = None

        self._dir_ = os.path.dirname(os.path.abspath(__file__))
        self._asset_dir = os.path.join(self._dir_, "assets")
        self._maps_dir = os.path.join(self._dir_, "maps")
        self._load_map()
        self._load_images()
        self.level = 0
        self.reward = 0
        self.render = render

        self.init()

    def _load_map(self):

        #sample_map_path = os.path.join(self._maps_dir, "sample_map.txt")
        #f = open(sample_map_path,"r")
        env = str(SimpleEnv(self.num_blocks))
        print(env)
        self.curmap = [ x.strip().split(', ') for x in env.split('\n')]
        print(self.curmap)
        #self.curmap = [ x.strip().split(", ") for x in f.readlines()]
        
        # find player and pig positions
        for i in range(len(self.curmap)):
            for j in range(len(self.curmap[i])):
                if self.curmap[i][j] == "P":
                    self.pig_pos = Position(i, j, Direction.DOWN)
                elif self.curmap[i][j] == "B":
                    self.init_player_pos = [i, j]
 
        #f.close()

    def _load_images(self):
        # preload and convert all the images so its faster when we reset
        bird_path = os.path.join(self._asset_dir, "bird.png")
        self.images["player"] = pygame.image.load(bird_path).convert_alpha()

        bg_path = os.path.join(self._asset_dir, "bg.png")
        self.images["background"] = pygame.image.load(bg_path).convert_alpha()

        wood_path = os.path.join(self._asset_dir, "wood.png")
        self.images["wood"] = pygame.image.load(wood_path).convert_alpha()

        pig_path = os.path.join(self._asset_dir, "pig.png")
        self.images["pig"] = pygame.image.load(pig_path).convert_alpha()

    def reset(self):
        self.backdrop = None
        self.player = None

        self.init()

    def init(self):

        if self.backdrop is None:
            self.backdrop = Backdrop(
                self.width,
                self.height,
                self.images,
                self.curmap
            )

        if self.player is None:
            self.player = BirdPlayer(
                self.width,
                self.height,
                self.init_player_pos,
                self.images["player"], self.curmap, render=self.render
            )

        self.score = 0.0
        self.game_tick = 0

    def getActionSpace(self):
        return self.actions
    
    def getGameState(self):

        state = {
            "player_y": self.player.pos_x,
            "player_x": self.player.pos_x,
        }

        return state

    def getScore(self):
        return self.player.position.x == self.pig_pos.x and self.player.position.y == self.pig_pos.y
        #return self.score

    def _handle_player_events(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:
                key = event.key
                if key == self.actions['forward']:
                    self.player.addBlock(ForwardBlock())
                elif key == self.actions['left']:
                    self.player.addBlock(TurnLeft())
                elif key == self.actions['right']:
                    self.player.addBlock(TurnRight())
                elif key == self.actions['run']:
                    self.player.executeProgram()

                    if self.render:
                        time.sleep(1)

    def game_over(self):
        return self.player.finished

    
    def step(self, dt):


        self.game_tick += 1
        dt = dt / 1000.0

        self._handle_player_events()
        self.player.update(dt)
        
        self.backdrop.draw_background(self.screen)
        self.backdrop.update_draw_base(self.screen, dt)
        self.player.draw(self.screen)
        

class AngryBirdEnv(gym.Env):

    
    def __init__(self, display_screen=True):

        self.game_state = PLE(AngryBird(render=display_screen), fps=30, display_screen=False)
        #self.game_state.init()

        self.display_screen = display_screen
        self._action_set = self.game_state.getActionSet()
        
        self.action_space = spaces.Discrete(len(self._action_set))
        self.screen_height, self.screen_width = self.game_state.getScreenDims()

        self.observation_space = spaces.Box(low=0, high=255, \
                                            shape=(self.screen_width, self.screen_height, 3), dtype=np.uint8)
        self.viewer = None

    def step(self, a):
        
        states = []
        if self._action_set[a] == K_r:

            
            while len(self.game_state.game.player.plan) > 1:

                _ = self.game_state.act(list(self._action_set)[a])
                state = self._get_image()
                if self.display_screen:
                    self.render()

                states.append(state)

        reward = self.game_state.act(list(self._action_set)[a])
        if self.display_screen:
            self.render()

        terminal = (self.game_state.game_over() or self.game_state.game.player.died)

        if self._action_set[a] != K_r:
            states = None
            reward = None
            #states.append(self._get_image())
            pass
        else:
            self.reset()

        return states, reward, terminal, {}

    def _get_image(self):
        image_rotated = np.fliplr(np.rot90(self.game_state.getScreenRGB(),3))
        return image_rotated

    @property
    def _n_actions(self):
        return len(self._action_set)

    def reset(self):
        '''
        Performs ther eset of the gym env
        '''
        #if self.display_screen:
        #   time.sleep(1)
        self.observation_space = spaces.Box(low=0, high=255, \
                                            shape=(self.screen_width, self.screen_height, 3), dtype=np.uint8)

        self.game_state.game.reset()

        state = self._get_image()
        self.render()
        return state

    def dummy(self):
        print ('called')

    def render(self, mode='human', close=False):
        '''
        Performs the rendering for the gym env
        '''
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
                return
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def _seed(self, _):
        self.game_state.init()
