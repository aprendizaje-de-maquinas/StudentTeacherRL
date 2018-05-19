import random

class EnvBase(object):

    def __init__(self, dim):

        assert dim > 1, 'the environment must be large enough to move in!'
        self.arr = [ [0 for _ in range(dim)] for _ in range(dim)]
        self.dim = dim
        self.orienation = 1

        
    def __repr__(self):

        ret = ''
        for row in self.arr:
            for x in range(len(row)):
                if row[x] == 0:
                    ret += 'W'
                elif row[x]==1:
                    ret += 'P'
                elif row[x]==2:
                    ret += 'B'
                else:
                    ret += 'E'

                if x != len(row) -1 :
                    ret += ', '

                

            
            #ret += ' '.join(['W' if x == 0 elif x==1 'P' elif x==2 'B' else 'E' for x in row])
            ret += '\n'

        return ret


class SimpleEnv(EnvBase):

    def __init__(self, dim):
        super(SimpleEnv, self).__init__(dim)

        path_length = random.randint(2, self.dim)
        orientation = random.randint(1, 2)

        start1 = random.randint(0, self.dim - path_length)
        start2 = random.randint(0, self.dim-1)

        #print(start1, start2, path_length, orientation)
        if orientation == 1:
            pig = random.randint(start1, start1+path_length-1)
            bird = pig
            while bird == pig:
                bird = random.randint(start1, start1+path_length-1)

            for i in range(start1, start1+path_length):
                if i == pig:
                    self.arr[i][start2] = 1
                elif i == bird:
                    self.arr[i][start2] = 2
                else:
                    self.arr[i][start2] = 3
        else:
            pig = random.randint(start1, start1+path_length-1)
            bird = pig
            while bird == pig:
                bird = random.randint(start1, start1+path_length-1)

            for i in range(start1, start1+path_length):
                if i == pig:
                    self.arr[start2][i] = 1
                elif i == bird:
                    self.arr[start2][i] = 2
                else:
                    self.arr[start2][i] = 3
            #for i in range(start1, start1+path_length):
            #    self.arr[start2][i] = 1
       

if __name__ == '__main__':

    env = SimpleEnv()

    #print(env)
