# Extend-Pytorch
Source: https://pytorch.org/tutorials/advanced/cpp_extension.html

## Get started
1. Build the extension
    ```bash
    cd python/extension
    python setup.py install
    ```
    - The `setup.py` file simply builds the extension in the current directory.
    - What could be set in `setup.py`:
        - module name
        - Cpp source files
2. Run an example code
    ```bash
    python play_extension.py
    ```
    - In `play_extension.py`, we import our built extension and call it from the python side.