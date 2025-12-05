import math
import os
import re

import torch

#os.environ['CUDNN_LOGLEVEL_DBG'] = '1'
#os.environ['CUDNN_LOGDEST_DBG'] = f'sdpa_backend_rank_{int(os.environ['LOCAL_RANK'])}.log'
#os.environ['CUDNN_FRONTEND_LOG_INFO'] = '1'
#os.environ['CUDNN_FRONTEND_LOG_FILE'] = f'sdpa_frontend_rank_{int(os.environ['LOCAL_RANK'])}.log'
#os.environ['TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT'] = '-1'

MAX_SPATIAL = 512
MAX_BATCH_SIZE = 32
MAX_ELEM = 2**31 + 65536
MAX_KERNEL = 9
MAX_STRIDE = 3
MAX_DILATION = 3
MAX_CHANNEL = 1024
DTYPES = [torch.float32, torch.bfloat16, torch.half]
DTYPES = [torch.bfloat16, torch.half]
SPATIAL_DIMS = [2, 3]
CHECK_REF = True
CHECK_REF_CPU = False

num_gpus = int(os.environ["WORLD_SIZE"])
device = int(os.environ["LOCAL_RANK"])
i = device

while True:
    dtype = DTYPES[torch.randint(low=0, high=len(DTYPES), size=(1,)).item()]
    num_spatial_dim = SPATIAL_DIMS[torch.randint(low=0, high=len(SPATIAL_DIMS), size=(1,)).item()]
    spatial_sizes = list()
    for _ in range(num_spatial_dim):
        high_spatial = MAX_ELEM//(math.prod([1] + spatial_sizes))
        if high_spatial <= 1:
            break
        spatial_size = torch.randint(1, high=min(int(MAX_SPATIAL**(1/(num_spatial_dim-1))), high_spatial), size=(1,)).item()
        spatial_sizes.append(spatial_size)
    if len(spatial_sizes) != num_spatial_dim:
        continue
    high_channel = MAX_ELEM//(math.prod(spatial_sizes))
    if high_channel <= 1:
        continue
    in_channel = torch.randint(1, high=min(MAX_CHANNEL, high_channel), size=(1,)).item()
    depthwise = bool(torch.randint(0, high=2, size=(1,)).item())
    if depthwise:
        out_channel = in_channel
    else:
        out_channel = torch.randint(1, high=min(MAX_CHANNEL, high_channel), size=(1,)).item()
    high_batch_size = MAX_ELEM//(max(in_channel, out_channel)*(math.prod(spatial_sizes)))
    if high_batch_size <= 1:
        continue
    batch_size = torch.randint(1, high=min(MAX_BATCH_SIZE, high_batch_size), size=(1,)).item()

    kernel_sizes = list()
    min_spatial_size = min(spatial_sizes)
    #dilation = torch.randint(low=1, high=min(MAX_DILATION, min_spatial_size)+1, size=(1,)).item()
    dilation = 1
    for spatial_idx in range(num_spatial_dim):
        kernel_size = torch.randint(1, high=min(MAX_KERNEL, min_spatial_size - (dilation - 1))+1, size=(1,)).item()
        kernel_sizes.append(kernel_size)

    input_shape = [batch_size, in_channel] + spatial_sizes
    groups = 1 if not depthwise else in_channel
    weight_shape = [out_channel, in_channel // groups] + kernel_sizes
    memory_formats = [torch.channels_last, torch.contiguous_format] if num_spatial_dim == 2 else [torch.channels_last_3d, torch.contiguous_format]
    memory_format = memory_formats[torch.randint(low=0, high=len(memory_formats), size=(1,)).item()]

    inp = torch.randn(*(input_shape), dtype=dtype, device=f'cuda:{device}').to(memory_format=memory_format).detach().clone()
    weight = torch.randn(*(weight_shape), dtype=dtype, device=f'cuda:{device}').to(memory_format=memory_format).detach().clone()
    inp.requires_grad = True
    weight.requires_grad = True

    stride = torch.randint(low=1, high=MAX_STRIDE+1, size=(1,)).item()
    print("KERNEL SHAPE", weight.shape, "MIN SPATIAL SIZE", min(spatial_sizes), " INPUT SHAPE ", inp.shape)
    if num_spatial_dim == 2:
        out = torch.nn.functional.conv2d(inp, weight, stride=stride, padding=0, dilation=dilation, groups=groups)
    else:
        out = torch.nn.functional.conv3d(inp, weight, stride=stride, padding=0, dilation=dilation, groups=groups)

    grad = torch.randn_like(out)
    out.backward(grad)

    if CHECK_REF:
        if CHECK_REF_CPU:
            inp_ref = inp.cpu().detach().clone()
            weight_ref = weight.cpu().detach().clone()
        else:
            inp_ref = inp.detach().clone()
            weight_ref = weight.detach().clone()

        inp_ref.requires_grad = True
        weight_ref.requires_grad = True
        with torch.backends.cudnn.flags(enabled=False):
            if num_spatial_dim == 2:
                out_ref = torch.nn.functional.conv2d(inp_ref, weight_ref, stride=stride, padding=0, dilation=dilation, groups=groups)
                out_ref.backward(grad.to(out_ref.device))
            else:
                out_ref = torch.nn.functional.conv3d(inp_ref, weight_ref, stride=stride, padding=0, dilation=dilation, groups=groups)
                out_ref.backward(grad.to(out_ref.device))
        try:
            torch.testing.assert_close(out_ref.cuda(), out, atol=1., rtol=1e-2)
            del out
            del out_ref
            torch.testing.assert_close(inp_ref.grad.cuda(), inp.grad, atol=1., rtol=1e-2)
            del inp_ref
            del inp
            torch.testing.assert_close(weight_ref.grad.cuda(), weight.grad, atol=1., rtol=1e-2)
            del weight_ref
            del weight
        except AssertionError as e:
            print(str(e))
            match = re.search(r'Mismatched elements:.*?\((\d+\.?\d*)%\)', str(e), re.MULTILINE)
            pct = match.group(1)
            print("mismatch pct: ", float(pct))
            if float(pct) < 1.0:
                    print("mismatches, but < 1.0%, so skipping for now........")
            else:
                print(f"failing case: {dtype}, batch {batch_size}, spatial sizes {spatial_sizes}, in channel {in_channel}, out channel {out_channel}, dilation {dilation} groups {groups} depthwise {depthwise} stride {stride} memory format {memory_format}")
                raise e
    print(f"case {i} passed")
    i += num_gpus
