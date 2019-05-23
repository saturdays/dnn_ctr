import numpy as np

seed = 131
mask=0x0fffffffffffffff

def BKDR2hash64v2(str):
    hash = 0
    for c in str:
        hash = hash*seed + ord(c)
    return hash&mask
    
class ps_server():
    def __init__(self, length):
        self.ps_data = {}
        self.length = length
    def get(self, key):
        if key not in self.ps_data:
            self.ps_data[key] = np.random.rand(self.length).astype(np.float32)*0.001
        return self.ps_data[key]
    def set_key(self, key, v):        
        self.ps_data[key] = v  