'''
   Simple class to hold an enum of the direction
   we arecurrently facing
'''
class Direction(object):
    UP    = 1
    LEFT  = 2
    DOWN  = 3
    RIGHT = 4

    def __init__(self):
        self.v = Direction.UP
        return

    def next(self):
        if self.v + 1 > Direction.RIGHT:
            self.v = Direction.UP
        else :
            self.v += 1

    def prev(self):
        if self.v == 0:
            self.v = Direction.RIGHT
        else:
            self.v -= 1

    def __repr__(self):
        if self.v == Direction.UP:
            return 'UP'
        elif self.v == Direction.LEFT:
            return 'LEFT'
        elif self.v == Direction.DOWN:
            return 'DOWN'
        else:
            return 'RIGHT'

'''
   Simple class to hold the current world location
   complete with x,y coordinates and direction for printing
'''
class Position(object):

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

        self.orientation = Direction()
        self._repr = ['UP', 'LEFT', 'DOWN', 'RIGHT']


    def __repr__(self):
        return '({} , {}) In Direction: {}'.format(self.x, self.y, self.orientation)

'''
   Global location of the agent
'''
location = Position()

'''
   Meta class for blocks that need knowledge of the
   location in the world to simplify their code
'''
class MovementBlock(object):

    def __init__(self):
        self.position = location
        return

'''
   Implements a for loop over a range
   Goes from the lowerbound to upper bound with a given
   step size just like a regular pythonic for loop would

   on calling this object, it will execute the body of the
   loop that is set using the body argument to init

   allows pretty print of the loop in a pythonic manner
'''
class RangeForBlock(object):

    def __init__(self, lowerBound, upperBound, step, body):
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.step = step

        self.body = body
        return

    def __call__(self):
        for _ in range(self.lowerBound, self.upperBound, self.step):
            self.body()
        return

    def __repr__(self, indent=0):

        tabs = ''.join(['\t' for _ in range(indent)])

        return tabs + 'for i in range({}, {}, {}):'.format(self.lowerBound, self.upperBound, self.step) + \
            '\n' + self.body.__repr__(indent=indent+1)

class WhileBlock(object):

    def __init__(self, cond, body):
        self.cond = cond

        self.body = body
        return

    def __call__(self):
        while self.cond():
            self.body()
        return

    def __repr__(self, indent=0):

        tabs = ''.join(['\t' for _ in range(indent)])

        return tabs + 'while {}:'.format(self.cond) + '\n' + self.body.__repr__(indent=indent+1)

'''
   Maybe for the future an iteration over an iterable in the same
   vane as the RangeForBlock
'''
'''
class IterForBlock(object):

    def __init__(self, iterable, body):
        self.iterable = iterable
        self.body = body

    def __call__(self):
        for i in self.iterable:
            self.body(i)

    def __repr__(self):
        return 'for i in {}'.format(self.iterable) + '\n' + self.body.__repr__()
'''

'''
   Allows the agent to turn left in the environment
   also allows for pretty print
'''
class TurnLeft(MovementBlock):

    def __call__(self):
        Direction.next(self.position.orientation)

    def __repr__(self, indent=0):
        return ''.join(['\t' for _ in range(indent)]) + 'TurnLeft()'

'''
   Allows the agent to turn right in the environment
   also allows for pretty print
'''
class TurnRight(MovementBlock):

    def __call__(self):
        Direction.prev(self.position.orientation)

    def __repr__(self, indent=0):

        return ''.join(['\t' for _ in range(indent)]) + 'TurnRight()'

'''
   Allows the agent to move forward in the environment
   irrespective of its orientation

   also allows for pretty print
'''
class ForwardBlock(MovementBlock):

    def __call__(self):

        orientation = self.position.orientation.v
        if orientation == Direction.UP:
            self.position.y += 1
        elif orientation == Direction.LEFT:
            self.position.x -= 1
        elif orientation == Direction.DOWN:
            self.position.y -= 1
        else:
            self.position.x += 1

        return

    def __repr__(self, indent=0):

        return ''.join(['\t' for _ in range(indent)]) + 'Forward()'

'''
   Allows the agent to move backward in the environment
   irrespective of its orientation

   also allows for pretty print
'''
class BackwardBlock(MovementBlock):

    def __call__(self):
        orientation = self.position.orientation
        if orientation == Direction.UP:
            self.position.y -= 1
        elif orientation == Direction.LEFT:
            self.position.x += 1
        elif orientation == Direction.DOWN:
            self.position.y += 1
        else:
            self.position.x -= 1

        return

    def __repr__(self, indent=0):

        return ''.join(['\t' for _ in range(indent)]) + 'Backward()'

'''
   Allows the agent to move left in the environment
   irrespective of its orientation

   also allows for pretty print
'''
class LeftBlock(MovementBlock):

    def __call__(self):
        orientation = self.position.orientation
        if orientation == Direction.UP:
            self.position.x -= 1
        elif orientation == Direction.LEFT:
            self.position.y -= 1
        elif orientation == Direction.DOWN:
            self.position.x += 1
        else:
            self.position.y += 1

        return

    def __repr__(self, indent=0):

        return ''.join(['\t' for _ in range(indent)]) + 'Left()'

'''
   Allows the agent to move right in the environment
   irrespective of its orientation

   also allows for pretty print
'''
class RightBlock(MovementBlock):

    def __call__(self):
        orientation = self.position.orientation
        if orientation == Direction.UP:
            self.position.x += 1
        elif orientation == Direction.LEFT:
            self.position.y += 1
        elif orientation == Direction.DOWN:
            self.position.x -= 1
        else:
            self.position.y -= 1

        return

    def __repr__(self, indent=0):

        return ''.join(['\t' for _ in range(indent)]) + 'Right()'

'''
   Allows the stacking of blocks together
   useful for building the body of a loop

   on a call to this will execute all the
   blocks that are stacked

   nice pretty print as well
'''
class StackedBlock(object):

    def __init__(self, blocks=[]):
        self.blocks = blocks
        return

    def stack(self, block):
        self.blocks.append(block)
        return

    def __call__(self):
        for block in self.blocks:
            block()
        return

    def __repr__(self, indent=0):
        ret = ""
        for block in self.blocks:
            ret = ret + block.__repr__(indent=indent) + '\n'
        return ret


if __name__ == '__main__':

    s_block = StackedBlock()


    for _ in range(7):
        s_block.stack(ForwardBlock())
        s_block.stack(TurnLeft())
        s_block.stack(TurnRight())

        #s_block.stack(BackwardBlock())

    for_block = RangeForBlock(0, 1, 1, s_block)

    for_block_2 = RangeForBlock(0, 1, 1, for_block)

    print(for_block_2)

    for_block()
    print(location)
