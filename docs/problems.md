# Problems

1. Importing cv2
    - Error message: cv2 cannot find libSM.so.6 even after updating the dockerfile and pushed
    - Solution: change the tag of the dockerfile
2. Cannot find tokenizer
    - Error: 
        ```
        OSError: Can't load tokenizer for 'openai/clip-vit-large-patch14'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure 'openai/clip-vit-large-patch14' is the correct path to a directory containing all relevant files for a CLIPTokenizer tokenizer.
        ```
    - Solution: 
        - https://huggingface.co/docs/transformers/installation#offline-mode
