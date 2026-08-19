"""Builds a TensorRT engine with FP16 everywhere except EfficientViT's
`context_module` (ReLU linear-attention) layers, which overflow FP16's
~65504 range because linear attention sums are unbounded (no softmax
normalization before the dot product) -- see bisection via polygraphy
that isolated the NaN to stages.3/.../context_module/main/MatMul.

Usage:
    python3 build_mixed_precision_engine.py <onnx_path> <engine_path> \
        --imgsz 1008 --min-batch 1 --opt-batch 4 --max-batch 16
"""
import argparse
import time

import tensorrt as trt


def build(onnx_path, engine_path, imgsz, min_batch, opt_batch, max_batch,
          input_name='pixel_values', workspace_gb=8, force_fp32_match='context_module'):
    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, '')
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    print(f'[Parse] {onnx_path} ...')
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f'  {parser.get_error(i)}')
            raise RuntimeError('ONNX parse failed')

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    n_forced = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        if force_fp32_match not in layer.name:
            continue
        # Skip shape/index/bool-producing layers -- only float/half tensors can be
        # reassigned to FP32; forcing int64/bool outputs to float breaks the graph.
        out_dtypes = [layer.get_output(k).dtype for k in range(layer.num_outputs)]
        if not all(dt in (trt.float32, trt.float16) for dt in out_dtypes):
            continue
        layer.precision = trt.float32
        for k in range(layer.num_outputs):
            layer.set_output_type(k, trt.float32)
        n_forced += 1
    print(f'[Precision] Forced {n_forced}/{network.num_layers} layers to FP32 '
          f'(matched "{force_fp32_match}")')

    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_name,
        (min_batch, 3, imgsz, imgsz),
        (opt_batch, 3, imgsz, imgsz),
        (max_batch, 3, imgsz, imgsz),
    )
    config.add_optimization_profile(profile)

    print(f'[Build] shapes min={min_batch} opt={opt_batch} max={max_batch} ...')
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError('TensorRT engine build failed')

    with open(engine_path, 'wb') as f:
        f.write(serialized)
    print(f'[Build] Saved: {engine_path}  ({(time.perf_counter() - t0) / 60:.1f} min)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('onnx_path')
    p.add_argument('engine_path')
    p.add_argument('--imgsz', type=int, default=1008)
    p.add_argument('--min-batch', type=int, default=1)
    p.add_argument('--opt-batch', type=int, default=4)
    p.add_argument('--max-batch', type=int, default=16)
    args = p.parse_args()
    build(args.onnx_path, args.engine_path, args.imgsz,
          args.min_batch, args.opt_batch, args.max_batch)
