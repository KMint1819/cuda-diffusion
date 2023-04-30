# Fake data for testing
Since the data are all flatten, you will have to reshape them to the correct shape before using them. The shape is specified in `python/generate_data.py`
- Data included:
    - `input.txt`: input tensor
    - `out.txt`: output tensor
    - `norm-weight.txt`: parameters to normalize the weight
    - `norm-bias.txt`: parameters to normalize the bias
    - `qkv-weight.txt`: weight for qkv
    - `qkv-bias.txt`: bias for qkv
    - `proj_out-weight.txt`: weight for proj_out (feed forward)
    - `proj_out-bias.txt`: bias for proj_out (feed forward)