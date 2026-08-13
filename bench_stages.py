#!/usr/bin/env python3
"""
Measure per-stage throughput:
  1. decode fps   : NVDEC via nvurisrcbin, counting buffers/sec before nvinfer (real pipeline)
  2. text encode  : weight/sam3_1008_open/text_encoder.engine, dynamic-prompt architecture (not baked)
  3. vision decode: weight/sam3_1008_open/vision_decoder.engine, dynamic-prompt architecture
  4. baked engine : SAM3Fixed/sam3_{imgsz}_det_*/sam3.engine (vision+decoder combined, used by ds9_rtsp.py)

(2)+(3) do NOT apply to SAM3Fixed/ds9_rtsp.py (SAM3Fixed) -- that pipeline
uses the pre-baked engine (4); the text encoder only runs once at build time
and is not in the runtime data path. (2)+(3) ARE both in the runtime data
path for SAM3Open/ds9_rtsp.py (SAM3Open): (3) runs inside nvinfer
every inferred frame, (2) runs in Python only when a prompt changes.

Usage: python3 bench_stages.py [--imgsz 420] [--batch 1 2 4]
"""
import argparse
import time

import tensorrt as trt
import torch


def bench_engine(path, batch, iters=30, warmup=10):
    lg = trt.Logger(trt.Logger.ERROR)
    eng = trt.Runtime(lg).deserialize_cuda_engine(open(path, "rb").read())
    ctx = eng.create_execution_context()
    bufs = {}
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        shape = list(eng.get_tensor_shape(n))
        if shape[0] == -1:
            shape[0] = batch
        if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT:
            ctx.set_input_shape(n, shape)
    for i in range(eng.num_io_tensors):
        n = eng.get_tensor_name(i)
        shape = tuple(ctx.get_tensor_shape(n))
        dt = {trt.float32: torch.float32, trt.float16: torch.float16,
              trt.bool: torch.bool, trt.int32: torch.int32, trt.int64: torch.int64
              }[eng.get_tensor_dtype(n)]
        t = torch.zeros(shape, dtype=dt, device="cuda") if dt != torch.bool \
            else torch.ones(shape, dtype=dt, device="cuda")
        bufs[n] = t
        ctx.set_tensor_address(n, t.data_ptr())

    stream = torch.cuda.Stream()
    for _ in range(warmup):
        ctx.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        ctx.execute_async_v3(stream_handle=stream.cuda_stream)
    stream.synchronize()
    dt_ms = (time.perf_counter() - t0) / iters * 1000
    return dt_ms, dt_ms / batch, batch / (dt_ms / 1000)


def bench_decode_fps(rtsp_url, seconds=8):
    """Counts buffers/sec through nvurisrcbin -> nvstreammux, with NO nvinfer,
    on the real pipeline (uses actual NVDEC decode, not an estimate)."""
    import threading
    from pyservicemaker import Pipeline, BatchMetadataOperator, Probe

    class Counter(BatchMetadataOperator):
        def __init__(self):
            super().__init__()
            self.n = 0

        def handle_metadata(self, batch_meta):
            self.n += len(list(batch_meta.frame_items))

    p = Pipeline("decode_fps_probe")
    p.add("nvurisrcbin", "src", {"uri": rtsp_url, "latency": 500, "drop-on-latency": True})
    p.add("nvstreammux", "mux", {"batch-size": 1, "width": 1280, "height": 720,
                                  "live-source": True, "batched-push-timeout": 40000})
    p.link(("src", "mux"), ("vsrc_%u", ""))
    p.add("fakesink", "sink", {})
    p.link("mux", "sink")
    counter = Counter()
    p.attach("mux", Probe("counter", counter))

    done = threading.Event()

    def _run():
        p.start().wait()
        done.set()

    threading.Thread(target=_run, daemon=True).start()
    time.sleep(2.0)  # warmup / connect RTSP
    n0 = counter.n
    t0 = time.perf_counter()
    time.sleep(seconds)
    n1 = counter.n
    elapsed = time.perf_counter() - t0
    p.stop()
    return (n1 - n0) / elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=420)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--rtsp", default="rtsp://localhost:8554/cam0")
    ap.add_argument("--skip-decode", action="store_true")
    args = ap.parse_args()

    print(f"{'='*70}\n1) DECODE FPS (NVDEC, real pipeline, no nvinfer)\n{'='*70}")
    if args.skip_decode:
        print("  (skipped -- needs a fake RTSP source running at", args.rtsp, ")")
    else:
        try:
            fps = bench_decode_fps(args.rtsp)
            print(f"  {args.rtsp}: {fps:.1f} fps")
        except Exception as e:
            print(f"  ERROR: {e}  (needs a running RTSP source -- fake-stream one with mediamtx + ffmpeg)")

    print(f"\n{'='*70}\n2) TEXT ENCODER (weight/sam3_1008_open/text_encoder.engine) -- dynamic-prompt architecture\n{'='*70}")
    for b in args.batch:
        try:
            ms, ms_per, fps = bench_engine("weight/sam3_1008_open/text_encoder.engine", b)
            print(f"  batch={b:<3d} {ms:8.3f} ms/batch  {ms_per:7.3f} ms/prompt  {fps:8.1f} prompt/s")
        except Exception as e:
            print(f"  batch={b}: ERROR {e}")

    print(f"\n{'='*70}\n3) VISION DECODER (weight/sam3_1008_open/vision_decoder.engine) -- dynamic-prompt architecture\n{'='*70}")
    for b in args.batch:
        try:
            ms, ms_per, fps = bench_engine("weight/sam3_1008_open/vision_decoder.engine", b)
            print(f"  batch={b:<3d} {ms:8.3f} ms/batch  {ms_per:7.3f} ms/img  {fps:8.1f} img/s")
        except Exception as e:
            print(f"  batch={b}: ERROR {e}")

    baked = f"weight/sam3_{args.imgsz}_det_car_truck/sam3.engine"
    print(f"\n{'='*70}\n4) BAKED ENGINE ({baked}) -- SAM3Fixed/ds9_rtsp.py architecture\n{'='*70}")
    for b in args.batch:
        try:
            ms, ms_per, fps = bench_engine(baked, b)
            print(f"  batch={b:<3d} {ms:8.3f} ms/batch  {ms_per:7.3f} ms/img  {fps:8.1f} img/s")
        except Exception as e:
            print(f"  batch={b}: ERROR {e}")


if __name__ == "__main__":
    main()
