from abc import ABC, abstractmethod

class Base(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        pass
    
    @abstractmethod
    def eval(self):
        pass