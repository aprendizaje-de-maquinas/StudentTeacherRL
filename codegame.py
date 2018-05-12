import pygame
from ple.games import base
from blocks.blocks import *
from pygame.constants import *
import os, sys

class BirdPlayer(pygame.sprite.Sprite):

    # The bird only needs to know its current plan and position
    def __init__(self,
                 SCREEN_WIDTH, SCREEN_HEIGHT, init_pos, image):
        
        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        self.plan = []

        pygame.sprite.Sprite.__init__(self)

        self.image = image

        self.game_tick = 0

        self.position = Position(init_pos[0], init_pos[1], Direction(Direction.RIGHT))
        self.orientation = Direction.RIGHT

        self.executingProgram = False
        self.finished = False

    # Adds a block to the current queue of blocks
    def addBlock(self, block):
        if not self.executingProgram:
            self.plan.append(block)

    def executeProgram(self):
        if not self.finished:
            self.executingProgram = True

    def update(self, dt):
        self.game_tick += 1
        if self.executingProgram:
            block = self.plan[0]
            block(self)
            self.plan = self.plan[1:]
            if not len(self.plan) or self.position.out_of_bounds(): # out of bounds
                self.executingProgram = False
                self.finished = True          

    def draw(self, screen):
        screen.blit(pygame.transform.rotate(self.image, 90 * (self.orientation - 3)), \
                    (self.SCREEN_WIDTH / 8 * self.position.y, self.SCREEN_HEIGHT / 8 * self.position.x))

class Block(pygame.sprite.Sprite):

    def __init__(self,
                 SCREEN_WIDTH, SCREEN_HEIGHT, init_pos, image):

        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        self.width = self.image.get_width()
        pygame.sprite.Sprite.__init__(self)

        self.pos_x = init_pos[0]
        self.pos_y = init_pos[1]

        self.image = image

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
    
    def update_draw_base(self, screen, dt):
        for i in range(len(self.map)):
            for j in range(len(self.map[i])):
                if self.map[i][j] == "W":
                    screen.blit(self.wood_image, (self.SCREEN_WIDTH / 8 * j, self.SCREEN_HEIGHT / 8 * i))
                elif self.map[i][j] == "P":
                    screen.blit(self.pig_image, (self.SCREEN_WIDTH / 8 * j, self.SCREEN_HEIGHT / 8 * i))


class AngryBird(base.PyGameWrapper):
    """
    Used physics values from sourabhv's `clone`_.

    .. _clone: https://github.com/sourabhv/FlapPyBird


    Parameters
    ----------
    width : int (default: 512)
        Screen width. Consistent gameplay is not promised for different widths or heights, therefore the width and height should not be altered.

    height : inti (default: 512)
        Screen height.
    """

    def __init__(self, width=256, height=256):

        actions = {
            "add_f_block": K_f,
            "run": K_r
        }

        fps = 30
        self.lives = -1

        base.PyGameWrapper.__init__(self, width, height, actions=actions)

        self.scale = 30.0 / fps

        self.allowed_fps = 30  # restrict the fps

        self.images = {}

        # so we can preload images
        pygame.display.set_mode((1, 1), pygame.NOFRAME)

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

    def _load_map(self):
        sample_map_path = os.path.join(self._maps_dir, "sample_map.txt")
        f = open(sample_map_path,"r")
        self.curmap = [ x.strip().split(", ") for x in f.readlines()]

        # find player and pig positions
        for i in range(len(self.curmap)):
            for j in range(len(self.curmap[i])):
                if self.curmap[i][j] == "P":
                    self.pig_pos = Position(i, j, Direction.DOWN)
                elif self.curmap[i][j] == "B":
                    self.init_player_pos = [i, j]
        
        f.close()

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
                self.images["player"],
            )

        self.score = 0.0
        self.game_tick = 0

    def getGameState(self):
        """
        Gets a non-visual state representation of the game.

        Returns
        -------

        dict
            * player y position.
            * players velocity.
            * next pipe distance to player
            * next pipe top y position
            * next pipe bottom y position
            * next next pipe distance to player
            * next next pipe top y position
            * next next pipe bottom y position


            See code for structure.

        """

        state = {
            "player_y": self.player.pos_x,
            "player_x": self.player.pos_x,
        }

        return state

    def getScore(self):
        return self.score

    def _handle_player_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key
                if key == self.actions['add_f_block']:
                    self.player.addBlock(ForwardBlock())
                elif key == self.actions['run']:
                    self.player.executeProgram()

    def game_over(self):
        return self.player.finished

    def step(self, dt):
        self.game_tick += 1
        dt = dt / 1000.0

        # handle player movement
        self._handle_player_events()
        
        self.player.update(dt)

        self.backdrop.draw_background(self.screen)
        self.backdrop.update_draw_base(self.screen, dt)
        self.player.draw(self.screen)

        if self.player.position.x == self.pig_pos.x and self.player.position.y == self.pig_pos.y:
            self.reward += 1
            self.player.finished = True        
