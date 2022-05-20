import numpy as np
from collections import defaultdict

#used disjoint_set and path compression to find conneted compontens that is neraby nodes and grouped them as 1
class disjoint_set:
    def __init__(self, total):
        self._num = total             
        self._arr = np.empty((self._num, 3), dtype=np.int64)
        for i in range(self._num):
            self._arr[i][0] = i              
            self._arr[i][1] = 0              
            self._arr[i][2] = 1              
        self._INT = np.empty((self._num, ), dtype=np.float64)
        
    @property
    def num(self):
        return self._num
    @property
    def arr(self):
        return self._arr    
    @property
    def INT(self):
        return self._INT    
    def update_INT(self, xp, yp, w):        
        new_INT = max(w, self._INT[xp], self._INT[yp])
        self._INT[yp] = new_INT
        self._INT[xp] = new_INT    
    def find(self, x):
        if self._arr[x][0] == x:
            return x
        else:
            self._arr[x][0] = self.find(self._arr[x][0])
            
            return self._arr[x][0]
    
    def union(self, x, y):      
        xp = self.find(x)
        yp = self.find(y)        
        if self._arr[xp][1] < self._arr[yp][1]:
            self._arr[xp][0] = yp
            self._arr[yp][2] += self._arr[xp][2]
    
        elif self._arr[xp][1] > self._arr[yp][1]:
            self._arr[yp][0] = xp
            self._arr[xp][2] += self._arr[yp][2]
    
        else:
            self._arr[xp][0] = yp
            self._arr[yp][1] += 1
            self._arr[yp][2] += self._arr[xp][2]
    
    def is_same_parent(self, x, y):        
        return self.find(x) == self.find(y)
    
    def conclusion(self):    
        con = defaultdict(list) 
        for i in range(self._num):
             con[self.find(self._arr[i][0])].append(i)                
        return con