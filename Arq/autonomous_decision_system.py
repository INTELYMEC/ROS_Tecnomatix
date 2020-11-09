
                                                                                                    
from abc import ABCMeta, abstractmethod          

                                                                                
class Autonomous_Decision_System(metaclass=ABCMeta):
    def __init__(self):
        self.method = ""       

    def register(self, who):
        self.subscriber = who      
  
    @abstractmethod                                                             
    def process(self):                                                             
        pass      
