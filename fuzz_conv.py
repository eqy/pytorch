import math
import os

import torch

#os.environ['CUDNN_LOGLEVEL_DBG'] = '1'
#os.environ['CUDNN_LOGDEST_DBG'] = f'sdpa_backend_rank_{int(os.environ['LOCAL_RANK'])}.log'
#os.environ['CUDNN_FRONTEND_LOG_INFO'] = '1'
#os.environ['CUDNN_FRONTEND_LOG_FILE'] = f'sdpa_frontend_rank_{int(os.environ['LOCAL_RANK'])}.log'
#os.environ['TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT'] = '-1'

MAX_SPATIAL = 32768
MAX_BATCH_SIZE = 512
MAX_ELEM = 2**32
MAX_KERNEL = 13
MAX_STRIDE = 3
MAX_DILATION = 3
MAX_CHANNEL = 2048
DTYPES = [torch.float32, torch.bfloat16, torch.half]
SPATIAL_DIMS = [2, 3]
CHECK_REF = True

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
    for _ in range(num_spatial_dim):
        kernel_size = torch.randint(1, high=MAX_KERNEL+1, size=(1,)).item()
        kernel_sizes.append(kernel_size)

    input_shape = [batch_size, in_channel] + spatial_sizes
    weight_shape = [out_channel, in_channel] + kernel_sizes
    memory_formats = [torch.channels_last, torch.contiguous_format] if num_spatial_dim == 2 else [torch.channels_last_3d, torch.contiguous_format]
    memory_format = memory_formats[torch.randint(low=0, high=len(memory_formats), size=(1,)).item()]

    inp = torch.randn(*(input_shape), dtype=dtype, device='cuda').to(memory_format=memory_format).detach().clone()
    weight = torch.randn(*(weight_shape), dtype=dtype, device='cuda').to(memory_format=memory_format).detach().clone()
    inp.requires_grad = True
    weight.requires_grad = True

    stride = torch.randint(low=1, high=MAX_STRIDE+1, size=(1,)).item()
    dilation = torch.randint(low=1, high=MAX_DILATION+1, size=(1,)).item()
    groups = 1 if not depthwise else in_channels

    if num_spatial_dim == 2:
        out = torch.nn.functional.conv2d(inp, weight, stride=stride, padding=0, dilation=dilation, groups=groups)
    else:
        out = torch.nn.functional.conv3d(inp, weight, stride=stride, padding=0, dilation=dilation, groups=groups)

    if CHECK_REF:
        inp_ref = inp.cpu().detach().clone()
        weight_ref = weight.cpu().detach().clone()
        if num_spatial_dim == 2:
            out = torch.nn.functional.conv2d(inp_ref, weight_ref, stride=stride, padding=0, dilation=dilation, groups=groups)
        else:
            out = torch.nn.functional.conv3d(inp_ref, weight_ref, stride=stride, padding=0, dilation=dilation, groups=groups)

    torch.testing.assert_close(out_ref, out, atol=5e-3, rtol=1e-3)

    out.backward(torch.randn_like(out))

    print(dtype, batch_size, spatial_sizes, in_channel, out_channel, depthwise)
