from G5 import G5
from G3 import G3
from G1 import G1
from G0 import G0

fault_name = ['N','IR6','B6','OR6','短路','断路','不平衡','断条']

for i in range(8):
    for j in fault_name:
        a1, b1 = G5(5000, j, i)
    
        a2, b2 = G3(5000, j, i)
    
        a3, b3 = G1(5000, j, i)
    
        a4, b4 = G0(5000, j, i)
