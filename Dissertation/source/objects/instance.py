import numpy as np

class Instances:

    def __init__(self, x, y, step=-1, score=-999, is_useful=True):
        self.x = np.array(x)
        self.y = np.array(y)
        self.step = step
        self.score = score
        self.is_useful = is_useful


    def __cmp__(self, other):
        return cmp(self.label, other.label)