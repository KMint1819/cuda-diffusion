# Connecting pytorch and C++
1. Build the library
    ```bash
    cd python/
    pip install ./
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