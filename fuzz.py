import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

# https://stackoverflow.com/questions/6800193/what-is-the-most-efficient-way-of-finding-all-the-factors-of-a-number-in-python
from functools import reduce
from math import sqrt
def factors(n):
        step = 2 if n%2 else 1
        return set(reduce(list.__add__,
                    ([i, n//i] for i in range(1, int(sqrt(n))+1, step) if n % i == 0)))

MIN_B = 1
MAX_B = 128
MIN_SEQLEN_Q = 2
MIN_SEQLEN_KV = 2
MAX_SEQLEN_Q = 2**17
MAX_SEQLEN_KV = 2**17
MIN_HEAD = 1
MAX_HEAD = 2048
MIN_DQK = 1
MAX_DQK = 128
MIN_DMOD = 8
MIN_DV = 1
MAX_DV = 128
MAX_ELEM = 2**29

i = 0
num_gpus = torch.cuda.device_count()

while True:
    device = i % num_gpus
    b = torch.randint(low=MIN_B, high=MAX_B+1, size=(1,)).item()
    high_sq = int(min(MAX_SEQLEN_Q, MAX_ELEM/b) + 1)
    high_skv = int(min(MAX_SEQLEN_KV, MAX_ELEM/b) + 1)
    if high_sq <= MIN_SEQLEN_Q or high_skv <= MIN_SEQLEN_KV:
        continue
    s_q = torch.randint(low=MIN_SEQLEN_Q, high=high_sq, size=(1,)).item()
    s_kv = torch.randint(low=MIN_SEQLEN_KV, high=high_skv, size=(1,)).item()
    high_hq = int(min(MAX_HEAD, MAX_ELEM/(b*s_q)) + 1)
    h_q = torch.randint(low=MIN_HEAD, high=high_hq, size=(1,)).item()
    h_kv_choices = list(factors(h_q))
    h_k = h_kv_choices[torch.randint(low=0, high=len(h_kv_choices), size=(1,)).item()]
    h_v = h_kv_choices[torch.randint(low=0, high=len(h_kv_choices), size=(1,)).item()]
    high_dqk = int(min(MAX_DQK, MAX_ELEM/(b*s_q*h_q), MAX_ELEM/(b*s_kv*h_k))//MIN_DMOD) + 1
    high_dv = int(min(MAX_DV, MAX_ELEM/(b*s_kv*h_v))//MIN_DMOD) + 1
    if high_dqk <= MIN_DQK or high_dv <= MIN_DV:
        continue
    d_qk = torch.randint(low=MIN_DQK//MIN_DMOD + 1, high=high_dqk, size=(1,)).item() * MIN_DMOD
    d_v = torch.randint(low=MIN_DV//MIN_DMOD + 1, high=high_dv, size=(1,)).item() * MIN_DMOD
    out_numel = b * s_q * h_q * d_v;
    if out_numel > MAX_ELEM:
        continue
    i += 1
    q_permute = list(torch.randperm(3)) + [3]
    q_reverse = [q_permute.index(i) for i in range(4)]
    k_permute = list(torch.randperm(3)) + [3]
    k_reverse = [k_permute.index(i) for i in range(4)]
    v_permute = list(torch.randperm(3)) + [3]
    v_reverse = [v_permute.index(i) for i in range(4)]

    print(f"GPU: {device} case: {i}\n"
        f"Q {[b, h_q, s_q, d_qk]} numel {b*s_q*h_q*d_qk} layout {q_permute}\n"
        f"K {[b, h_k, s_kv, d_qk]} numel {b*s_kv*h_k*d_qk} layout {k_permute}\n"
        f"V {[b, h_v, s_kv, d_v]} numel {b*s_kv*h_v*d_v} layout {v_permute}\n"
        f"O {[b, h_q, s_q, d_v]} numel {out_numel}\r")
   
    qfillshape = [[b, h_q, s_q, d_qk][idx] for idx in q_permute]
    kfillshape = [[b, h_k, s_kv, d_qk][idx] for idx in k_permute]
    vfillshape = [[b, h_v, s_kv, d_v][idx] for idx in v_permute]
    q = torch.randn(qfillshape, dtype=torch.half, device=f'cuda:{device}', requires_grad=True).permute(q_reverse)
    k = torch.randn(kfillshape, dtype=torch.half, device=f'cuda:{device}', requires_grad=True).permute(k_reverse)
    v = torch.randn(vfillshape, dtype=torch.half, device=f'cuda:{device}', requires_grad=True).permute(v_reverse)
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        out = F.scaled_dot_product_attention(q, k, v).sum().backward()
    with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True).sum().backward()

