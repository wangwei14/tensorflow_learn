# coding=utf-8

from __future__ import print_function
import numpy as np

# 1. 数组切片  
if False:  
    arr = np.arange(0,10)  
    print(arr)
    print(arr[2])  
    print(arr[3:5])  
    print(arr[:5])  
    print(arr[-1])  
    print(arr[5:-1])  
    print(arr[1:8:2])  
    print(arr[::-1])  
    print(arr[5:1:-1])  
      
# 2. 数组列表下标存取  
if False:  
    arr = np.arange(0,10)  
    print(arr[[9, 7, 5, 3, 1]])  
    arr[[3, 1, 2]] = 99, 100, 101  
         
# 3. 以数组为下标存取  
if False:  
    index_arr = np.arange(0,12).reshape(4,-1)  
    arr = np.arange(100,50,-2)  
    print(arr[index_arr])  
      
    index_index_arr = np.random.permutation(index_arr.shape[0])  
    print(index_arr[index_index_arr])  
      
# 4. 多维数组切片存取  
if False:  
    arr = np.arange(0,36).reshape(6,6)  
    print(arr[1:6:2, [4, 3, 2]])  
    print(arr[[2, 4, 5]][:,[2,3]])

s = [[i] for i in range(5)]
s += [[0.] for i in range(2)]
t = [[i+1] for i in range(5)]
print(s)
print(s+t)
