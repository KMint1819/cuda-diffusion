# Connecting pytorch and C++
1. Build the library
    ```bash
    cd python/
    pip install ./
    ```
    > pip install will have error with g++11. If the this message
    ```bash
    /usr/include/c++/11/bits/std_function.h:435:145: error: parameter packs not expanded with ‘...’:
      435 |         function(_Functor&& __f) 
    ```  
    > shows up, use g++10 like below instead.
    ```bash
    CC=gcc-10 pip install ./
    ```
    
2. How to use the library
```bash
import torch # Must import torch first!
import gten # Our built target name

# Use the library 
print(gten.hello('ECE508 group ten'))
```
3. Example of forward pass of our block
```python
cd python/crossattn
python forward_our.py
```