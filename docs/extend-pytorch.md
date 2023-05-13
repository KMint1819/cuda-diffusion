# Extend-Pytorch
Source: https://pytorch.org/tutorials/advanced/cpp_extension.html

## Get started
1. Build the extension
    ```bash
    cd python/
    python -m pip install ./
    ```
    - The `python/setup.py` file simply builds the extension in the current directory.
    - What could be set in `setup.py`:
        - module name
        - Cpp source files
        - our additional python modules
2. Run an example code
    ```bash
    cd crossattn/
    python ensure_our.py
    ```
    - In `ensure_our.py`, we take the result from our c++ extension and compare with the correct answer