from crossattn_copied import CrossAttention as CrossAttentionCopied
from crossattn_truncated import CrossAttention as CrossAttentionTruncated
from crossattn_our import CrossAttention as CrossAttentionOur

import argparse
import torch
import numpy as np
from pathlib import Path

torch.set_printoptions(sci_mode=False)
tolerance = 1e-4
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

KWARGS = {
    'query_dim': 320,
    'context_dim': 768,
    'heads': 8,
    'dim_head': 40,
    'dropout': 0.0,
}
CROSSATTN_TYPE = {
    'copied': CrossAttentionCopied,
    'truncated': CrossAttentionTruncated,
    'our': CrossAttentionOur,
}

def get_data():
    def load_data(p, shape):
        raw = np.loadtxt(p, dtype=np.float32)
        tensor = torch.from_numpy(raw).reshape(shape)
        return tensor


    inner_dim = KWARGS['dim_head'] * KWARGS['heads'] 

    data_dir = Path(__file__).cwd().parent.parent / 'data'
    x = load_data(data_dir / 'input.txt', (1, 4096, 320))
    context = load_data(data_dir / 'context.txt', (1, 77, 768))
    state_dict = {
        'to_q.weight': load_data(data_dir / 'to_q-weight.txt', (inner_dim, KWARGS['query_dim'])),
        'to_k.weight': load_data(data_dir / 'to_k-weight.txt', (inner_dim, KWARGS['context_dim'])),
        'to_v.weight': load_data(data_dir / 'to_v-weight.txt', (inner_dim, KWARGS['context_dim'])),
        'to_out.0.weight': load_data(data_dir / 'to_out-0-weight.txt', (KWARGS['query_dim'], inner_dim)),
        'to_out.0.bias': load_data(data_dir / 'to_out-0-bias.txt', (KWARGS['query_dim'],)),
    }
    return x, context, state_dict

def main(args):
    attn = CROSSATTN_TYPE[args.attn_type](**KWARGS)
    print('=' * 80)
    for k, v in attn.state_dict().items():
        print(k, v.shape)
    print('=' * 80)
    x, context, state_dict = get_data()

    attn.load_state_dict(state_dict)

    attn = attn.eval().to(device)
    x = x.to(device)
    context = context.to(device)
    with torch.no_grad():
        out = x.clone()
        for i in range(1):
            out = attn(out, context)

        print(f'output: ', out)
        print(f'sum: ', out.sum())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('attn_type', type=str, choices=['copied', 'truncated', 'our'], default='our')
    args = parser.parse_args()
    print('Running crossatnn with attention type: ', args.attn_type)
    main(args)