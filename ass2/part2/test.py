import numpy as np
import pdb
a = np.random.binomial(1, 0.1, [4,5])
"""
a = np.arange(24).reshape([3,8])
print(np.shape(a))
d = np.tile(a,(4,1))
print(np.shape(d))
list_d = [d[:,i] for i in range(np.shape(d)[1])]
b = np.stack(list_d,axis=1)
print(np.shape(b))
c = np.split(b,4,0)
print(np.shape(c))
e = np.stack(c,axis=1)
print(np.shape(e))
f = np.take(e,[0,2], axis=0)
print(np.shape(f))
g = np.reshape(f,[2*8,-1])
print(np.shape(g))
"""
pdb.set_trace()
