#Integer Indexing
import numpy as np
x=np.array([[1,2],
            [3,4],
            [5,6]])
y=x[[0,1,2],[0,1,0]]
print(y)
#the selection includes elements at 
#(0,0),(1,1),(2,0) from the first array