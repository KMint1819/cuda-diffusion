# Connecting pytorch and C++
1. Build the library
    ```bash
    cd python/extension
    python setup.py install
    ```
    - You might have to remove the generated folders if you want to rebuild
2. How to use the library
```bash
import torch # Must import torch first!
import torda # Our built target name

# Use the library 
print(torda.hello('ECE508 group ten'))
```