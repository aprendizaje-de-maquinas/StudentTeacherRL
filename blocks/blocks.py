class Position(object):

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return '({} , {})'.format(self.x, self.y)

location = Position()

class MovementBlock(object):

    def __init__(self):
        self.position = location
        return


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

    def __repr__(self):
        return 'for i in range({}, {}, {})'.format(self.lowerBound, self.upperBound, self.step) + '\n' + self.body.__repr__()

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

class ForwardBlock(MovementBlock):

    def __call__(self):
        self.position.y += 1
        return

    def __repr__(self):
        return '\tForward()'

class BackwardBlock(MovementBlock):

    def __call__(self):
        self.position.y -= 1
        return

    def __repr__(self):
        return '\tBackward()'

class LeftBlock(MovementBlock):

    def __call__(self):
        self.position.x -= 1
        return

    def __repr__(self):
        return '\tLeft()'

class RightBlock(MovementBlock):

    def __call__(self):
        self.position.x += 1
        return

    def __repr__(self):
        return '\tRight()'

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

    def __repr__(self):
        ret = ""
        for block in self.blocks:
            ret = ret + block.__repr__() + '\n'
        return ret


if __name__ == '__main__':

    s_block = StackedBlock()


    for _ in range(10):
        s_block.stack(ForwardBlock())
        #s_block.stack(BackwardBlock())

    for_block = RangeForBlock(0, 10, 2, s_block)

    print(for_block)

    for_block()
    print(location)
