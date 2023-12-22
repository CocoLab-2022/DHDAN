import random
import numpy as np
import tqdm
import torch
import pickle

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state= map(np.stack, zip(batch))
        return state, action, reward, next_state

    def __len__(self):
        return len(self.buffer)
def get_class_attr(Cls) -> []:
    """
    get attribute name from Class(type)
    :param Cls:
    :return:
    """
    import re
    return [a for a, v in Cls.__dict__.items()
              if not re.match('<function.*?>', str(v))
              and not (a.startswith('__') and a.endswith('__'))]

def get_class_attr_val(cls):
    """
    get attribute name and their value from class(variable)
    :param cls:
    :return:
    """
    attr = get_class_attr(type(cls))
    attr_dict = {}
    for a in attr:
        attr_dict[a] = getattr(cls, a)
    return attr_dict


def load_obj(path):
    return pickle.load(open(path, 'rb'))