# Fake data for testing
Since the data are all flatten, you will have to reshape them to the correct shape before using them. The shape is specified in `python/crossattn/generate_data.py`
- Data included:
    - `input.txt`: input tensor
    - `out.txt`: output tensor
    - `to_q-weight.txt`: weight for to_q
    - `to_k-weight.txt`: weight for to_k
    - `to_v-weight.txt`: weight for to_v
    - `to_out-0-weight.txt`: weight for to_out
    - `to_out-0-bias.txt`: bias for to_out