class GA3C:
    def __init__(self):
        pass

    def train(self):
        from GA3C.Server import Server

        # Start main program
        Server().main()


if __name__ == '__main__':

    agent = GA3C()
    agent.train()
