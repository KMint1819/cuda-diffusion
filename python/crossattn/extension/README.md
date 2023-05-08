# Connecting pytorch and C++
1. Build the library
    ```bash
    cd python/extension
    pip install ./
    ```
    - You might have to remove the generated folders if you want to rebuild
2. How to use the library
```bash
import torch # Must import torch first!
import gten # Our built target name

# Use the library 
print(gten.hello('ECE508 group ten'))
```