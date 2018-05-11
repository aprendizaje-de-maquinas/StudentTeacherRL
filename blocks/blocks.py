'''
TODO:

add conditional blocks.

HitWall()
NotAtTarget()
ForwardClear()
LeftClear()
RightClear()
BackClear()

any else?
'''


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
   Used for building out empty control structures
   does nothing!
'''
class NullBlock(object):

    def __init__(self):
        return

    def __call__(self):
        return

    def __repr__(self, indent=0):
        return ''.join(['\t' for _ in range(indent)]) + 'pass'

'''
   Simple if else class
'''
class IfElseBlock(object):

    def __init__(self, cond, ifbody=None, elsebody=None):

        if ifbody is None: self._if = NullBlock()
        else: self._if = ifbody
        
        if elsebody is None: self._else = NullBlock()
        else: self._else = elsebody
        
        self.cond = cond
        
        return

    def __call__(self):
        if self.cond:
            self._if()
        else:
            self._else()

        return

    def __repr__(self, indent=0):    
        return ''.join(['\t' for _ in range(indent)]) + 'if ' + self.cond + ':\n' + \
            self._if.__repr__(indent + 1) + '\n' + \
            ''.join(['\t' for _ in range(indent)]) + 'else:\n' + self._else.__repr__(indent+1)

'''
   Simple if class
'''
class IfBlock(object):

    def __init__(self, cond, ifbody=None):

        if ifbody is None: self._if = NullBlock()
        else: self._if = ifbody
        
        self.cond = cond
        
        return

    def __call__(self):
        if self.cond:
            self._if()

        return

    def __repr__(self, indent=0):    
        return ''.join(['\t' for _ in range(indent)]) + 'if ' + self.cond + ':\n' + \
            self._if.__repr__(indent + 1)
            
'''
   Implements a for loop over a range
   Goes from the lowerbound to upper bound with a given
   step size just like a regular pythonic for loop would

   on calling this object, it will execute the body of the
   loop that is set using the body argument to init

   allows pretty print of the loop in a pythonic manner
'''
class RangeForBlock(object):

    def __init__(self, lowerBound, upperBound, step, body=None):
        self.lowerBound = lowerBound
        self.upperBound = upperBound
        self.step = step

        if body is None: self.body = NullBlock()
        else: self.body = body
        
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

    def __init__(self, cond, body=None):
        self.cond = cond

        if body is None: self.body = NullBlock()
        else: self.body = body
        
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
        s_block.stack(IfElseBlock('test'))

        #s_block.stack(BackwardBlock())

    for_block = RangeForBlock(0, 1, 1, s_block)

    for_block_2 = RangeForBlock(0, 1, 1, for_block)

    print(for_block_2)

    for_block()
    print(location)
