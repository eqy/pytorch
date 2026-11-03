"""Compare cuDNN LSTM forward algorithms for issue #203213.

See cudnn_small_hidden_notes.txt for measured results, build instructions and
remaining experiments. Direct mode forces only the forward algorithm; backward
uses the installed PyTorch heuristic. Native modes exercise ordinary nn.LSTM.
Run as a module: python -m fastrnns.cudnn_small_hidden (see the notes for PYTHONPATH).
"""

import argparse
import ctypes as C
import json
import statistics
import time

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import parametrize, run_tests, TestCase


P, I, Z = C.c_void_p, C.c_int, C.c_size_t
lib = C.CDLL("libcudnn.so.9")
lib.cudnnGetErrorString.restype = C.c_char_p


def bind(name, types):
    fn = getattr(lib, name)
    fn.argtypes = types
    fn.restype = I

    def checked(*args):
        status = fn(*args)
        if status:
            raise RuntimeError(f"{name}: {lib.cudnnGetErrorString(status).decode()}")

    return checked


create_handle = bind("cudnnCreate", [C.POINTER(P)])
set_stream = bind("cudnnSetStream", [P, P])
create_dropout = bind("cudnnCreateDropoutDescriptor", [C.POINTER(P)])
set_dropout = bind("cudnnSetDropoutDescriptor", [P, P, C.c_float, P, Z, C.c_ulonglong])
create_rnn = bind("cudnnCreateRNNDescriptor", [C.POINTER(P)])
set_rnn = bind("cudnnSetRNNDescriptor_v8", [P] + [I] * 12 + [P, C.c_uint])
create_data = bind("cudnnCreateRNNDataDescriptor", [C.POINTER(P)])
set_data = bind("cudnnSetRNNDataDescriptor", [P, I, I, I, I, I, C.POINTER(I), P])
create_tensor = bind("cudnnCreateTensorDescriptor", [C.POINTER(P)])
set_tensor = bind("cudnnSetTensorNdDescriptor", [P, I, I, C.POINTER(I), C.POINTER(I)])
weight_size = bind("cudnnGetRNNWeightSpaceSize", [P, P, C.POINTER(Z)])
temp_sizes = bind("cudnnGetRNNTempSpaceSizes", [P, P, I, P, C.POINTER(Z), C.POINTER(Z)])
forward = bind("cudnnRNNForward", [P, P, I] + [P] * 11 + [Z, P, Z, P, Z, P])


class DirectRNN:
    @torch.no_grad()
    def __init__(self, model, x, algo, training=True):
        self.model = model
        self.x = x.contiguous()
        seq, batch, inputs = x.shape
        hidden, layers = model.hidden_size, model.num_layers
        dtype = {torch.float32: 0, torch.float16: 2, torch.bfloat16: 9}[x.dtype]
        math_type = 1 if dtype == 2 else (0 if torch.backends.cudnn.allow_tf32 else 3)
        self.weights = torch._cudnn_rnn_flatten_weight(
            model._flat_weights, 4, inputs, 2, hidden, 0, layers, False, False
        )
        self.weight_bytes = self.weights.numel() * self.weights.element_size()
        self.hx = torch.zeros(layers, batch, hidden, device=x.device, dtype=x.dtype)
        self.cx = torch.zeros_like(self.hx)
        self.y = torch.empty(seq, batch, hidden, device=x.device, dtype=x.dtype)
        self.hy, self.cy = torch.empty_like(self.hx), torch.empty_like(self.cx)
        self.handle, self.dropout, self.rnn = P(), P(), P()
        self.xdesc, self.ydesc, self.hdesc = P(), P(), P()
        self.training = training
        try:
            create_handle(C.byref(self.handle))
            set_stream(self.handle, torch.cuda.current_stream().cuda_stream)
            create_dropout(C.byref(self.dropout))
            set_dropout(self.dropout, self.handle, 0.0, None, 0, 0)
            create_rnn(C.byref(self.rnn))
            set_rnn(
                self.rnn,
                algo,
                2,
                2,
                0,
                0,
                dtype,
                0,
                math_type,
                inputs,
                hidden,
                hidden,
                layers,
                self.dropout,
                1,
            )
            lengths = (I * batch)(*([seq] * batch))
            create_data(C.byref(self.xdesc))
            create_data(C.byref(self.ydesc))
            set_data(self.xdesc, dtype, 0, seq, batch, inputs, lengths, None)
            set_data(self.ydesc, dtype, 0, seq, batch, hidden, lengths, None)
            create_tensor(C.byref(self.hdesc))
            set_tensor(
                self.hdesc,
                dtype,
                3,
                (I * 3)(layers, batch, hidden),
                (I * 3)(batch * hidden, hidden, 1),
            )
            required = Z()
            weight_size(self.handle, self.rnn, C.byref(required))
            if required.value != self.weight_bytes:
                raise RuntimeError(
                    f"weight buffer mismatch: {required.value} != {self.weight_bytes}"
                )
            workspace, reserve = Z(), Z()
            temp_sizes(
                self.handle,
                self.rnn,
                int(training),
                self.xdesc,
                C.byref(workspace),
                C.byref(reserve),
            )
            self.workspace = torch.empty(
                workspace.value, device=x.device, dtype=torch.uint8
            )
            self.reserve = torch.empty(
                reserve.value, device=x.device, dtype=torch.uint8
            )
            self.dy = torch.randn_like(self.y)
        except Exception:
            self.close()
            raise

    def forward(self):
        forward(
            self.handle,
            self.rnn,
            int(self.training),
            None,
            self.xdesc,
            self.x.data_ptr(),
            self.ydesc,
            self.y.data_ptr(),
            self.hdesc,
            self.hx.data_ptr(),
            self.hy.data_ptr(),
            self.hdesc,
            self.cx.data_ptr(),
            self.cy.data_ptr(),
            self.weight_bytes,
            self.weights.data_ptr(),
            self.workspace.numel(),
            self.workspace.data_ptr(),
            self.reserve.numel(),
            self.reserve.data_ptr(),
        )
        return self.y, (self.hy, self.cy)

    @torch.no_grad()
    def backward(self):
        return torch.ops.aten._cudnn_rnn_backward.default(
            self.x,
            self.model._flat_weights,
            4,
            self.weights,
            self.hx,
            self.cx,
            self.y,
            self.dy,
            None,
            None,
            2,
            self.model.hidden_size,
            0,
            self.model.num_layers,
            False,
            0.0,
            True,
            False,
            [],
            None,
            self.reserve,
            [True, True, True, True],
        )

    def step(self):
        self.forward()
        return self.backward()

    def close(self):
        for suffix, desc in (
            ("RNNDataDescriptor", self.xdesc),
            ("RNNDataDescriptor", self.ydesc),
            ("TensorDescriptor", self.hdesc),
            ("RNNDescriptor", self.rnn),
            ("DropoutDescriptor", self.dropout),
            ("", self.handle),
        ):
            if desc:
                bind("cudnnDestroy" + suffix, [P])(desc)


def timed(fn, steps=10, repeats=3):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(steps):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000 / steps)
    return round(statistics.median(samples), 3)


class TestAlgorithms(TestCase):
    @parametrize("hidden", [128, 129, 136, 160, 192, 193, 200])
    @parametrize("layers", [1, 2])
    @parametrize("batch", [32, 1024])
    @torch.backends.cudnn.flags(enabled=True, allow_tf32=False)
    def test_standard(self, device, hidden, layers, batch):
        torch.manual_seed(0)
        model = torch.nn.LSTM(14, hidden, layers).to(device)
        x = torch.randn(56, batch, 14, device=device, requires_grad=True)
        direct = DirectRNN(model, x.detach(), 0)
        try:
            actual, actual_states = direct.forward()
            grads = direct.backward()
            with torch.backends.cudnn.flags(enabled=False):
                expected, states = model(x)
                expected.backward(direct.dy)
            self.assertEqual(actual, expected, atol=2e-5, rtol=2e-4)
            self.assertEqual(actual_states, states, atol=2e-5, rtol=2e-4)
            self.assertEqual(grads[0], x.grad, atol=2e-5, rtol=2e-4)
            for grad, parameter in zip(grads[3], model.parameters()):
                self.assertEqual(grad, parameter.grad, atol=2e-3, rtol=2e-3)
        finally:
            direct.close()


instantiate_device_type_tests(TestAlgorithms, globals(), only_for="cuda")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    parser.add_argument(
        "--backend", choices=["direct", "cudnn", "aten"], default="direct"
    )
    parser.add_argument(
        "--hidden",
        nargs="+",
        type=int,
        default=[120, 128, 129, 136, 144, 160, 184, 192, 193, 200, 224],
    )
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--seq", type=int, default=56)
    parser.add_argument("--inputs", type=int, default=14)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument(
        "--dtype", choices=["float32", "float16", "bfloat16"], default="float32"
    )
    args, rest = parser.parse_known_args()
    if not args.bench:
        run_tests(argv=[__file__] + rest)
    else:
        if rest:
            parser.error(f"unrecognized arguments: {' '.join(rest)}")
        if (
            min(
                args.batch,
                args.seq,
                args.inputs,
                args.layers,
                args.steps,
                args.repeats,
                *args.hidden,
            )
            <= 0
        ):
            parser.error("dimensions, steps, and repeats must be positive")
        torch.manual_seed(0)
        print(
            json.dumps(
                dict(
                    torch=torch.__version__,
                    cuda=torch.version.cuda,
                    cudnn=torch.backends.cudnn.version(),
                    device=torch.cuda.get_device_name(),
                    capability=torch.cuda.get_device_capability(),
                    torch_path=torch.__file__,
                    cudnn_tf32=torch.backends.cudnn.allow_tf32,
                    config=vars(args),
                )
            ),
            flush=True,
        )
        dtype = getattr(torch, args.dtype)
        x = torch.randn(args.seq, args.batch, args.inputs, device="cuda", dtype=dtype)
        for hidden in args.hidden:
            model = torch.nn.LSTM(args.inputs, hidden, args.layers).cuda().to(dtype)
            algos = (0, 3) if args.reverse else (3, 0)
            if args.backend != "direct":
                algos = (None,)
            optimizer = torch.optim.Adam(model.parameters())
            for algo in algos:
                for training in (True, False):
                    direct = None
                    try:
                        row = dict(
                            hidden=hidden,
                            batch=args.batch,
                            seq=args.seq,
                            inputs=args.inputs,
                            layers=args.layers,
                            dtype=args.dtype,
                            backend=args.backend,
                            algo=algo,
                            training=training,
                        )
                        if args.backend == "direct":
                            direct = DirectRNN(model, x, algo, training)
                            row["forward_ms"] = timed(
                                direct.forward, args.steps, args.repeats
                            )
                            if training and not args.forward_only:
                                row["forward_backward_ms"] = timed(
                                    direct.step, args.steps, args.repeats
                                )
                        else:
                            model.train(training)

                            @torch.no_grad()
                            def native_forward():
                                return model(x)

                            def native_step():
                                optimizer.zero_grad()
                                model(x)[0][-1].square().mean().backward()
                                optimizer.step()

                            with torch.backends.cudnn.flags(
                                enabled=args.backend == "cudnn"
                            ):
                                row["forward_ms"] = timed(
                                    native_forward, args.steps, args.repeats
                                )
                                if training and not args.forward_only:
                                    row["train_step_ms"] = timed(
                                        native_step, args.steps, args.repeats
                                    )
                        print(json.dumps(row), flush=True)
                    except RuntimeError as error:
                        if "CUDNN_STATUS_NOT_SUPPORTED" not in str(error):
                            raise
                        row["error"] = str(error)
                        print(json.dumps(row), flush=True)
                    finally:
                        if direct is not None:
                            direct.close()
