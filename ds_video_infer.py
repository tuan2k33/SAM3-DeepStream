#!/usr/bin/env python3
"""
Run SAM3 inference on a single video file via DeepStream (GStreamer native API).

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 ds_video_infer.py <video.mp4> [output.mp4]
                                                     [--classes adult child phone]
                                                     [--width 1920] [--height 1080]
"""

import argparse
import os
import signal
import threading
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

try:
    import pyds
    HAS_PYDS = True
except ImportError:
    HAS_PYDS = False

_HERE = Path(__file__).parent   
_ROOT = _HERE.parent            


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("video",            help="Input video file")
    p.add_argument("output", nargs="?", help="Output mp4 (default: <video>_ds.mp4)")
    p.add_argument("--classes", nargs="+", default=["adult", "child", "phone"])
    p.add_argument("--width",  type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--bitrate", type=int, default=8000, help="Encoder bitrate kbps")
    p.add_argument("--interval", type=int, default=0,   help="nvinfer interval (0=every frame)")
    p.add_argument("--full", action="store_true", help="Use full model (det+mask, config_infer_full.txt)")
    return p.parse_args()


def make_element(factory, name, **props):
    el = Gst.ElementFactory.make(factory, name)
    if el is None:
        raise RuntimeError(f"Cannot create GStreamer element: {factory}")
    for k, v in props.items():
        el.set_property(k.replace("_", "-"), v)
    return el


def osd_probe(pad, info, classes):
    if not HAS_PYDS:
        return Gst.PadProbeReturn.OK
    buf = info.get_buffer()
    if not buf:
        return Gst.PadProbeReturn.OK
    batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
    if not batch:
        return Gst.PadProbeReturn.OK
    l_frame = batch.frame_meta_list
    while l_frame:
        fm = pyds.NvDsFrameMeta.cast(l_frame.data)
        if fm.frame_num % 30 == 0:
            counts = {}
            l_obj = fm.obj_meta_list
            while l_obj:
                om = pyds.NvDsObjectMeta.cast(l_obj.data)
                cls = int(om.class_id)
                counts[cls] = counts.get(cls, 0) + 1
                l_obj = l_obj.next
            if counts:
                parts = [f"{classes[k] if k < len(classes) else k}={v}"
                         for k, v in sorted(counts.items())]
                print(f"  frame {fm.frame_num}: {' '.join(parts)}", flush=True)
        l_frame = l_frame.next
    return Gst.PadProbeReturn.OK


def build_pipeline(video_file: str, output_file: str,
                   infer_config: str, classes: list,
                   input_w: int, input_h: int,
                   bitrate: int, interval: int) -> Gst.Pipeline:
    pipeline = Gst.Pipeline.new("video-infer")

    # Decode: filesrc → qtdemux → h264parse → nvv4l2decoder
    filesrc = make_element("filesrc",       "filesrc",  location=video_file)
    qtdemux = make_element("qtdemux",       "qtdemux")
    h264p   = make_element("h264parse",     "h264parse")
    decoder = make_element("nvv4l2decoder", "decoder")

    mux = make_element("nvstreammux", "mux",
                       batch_size=1,
                       width=input_w,
                       height=input_h,
                       live_source=False,
                       batched_push_timeout=40000)

    infer = make_element("nvinfer", "infer",
                         config_file_path=infer_config,
                         interval=interval)

    conv1 = make_element("nvvideoconvert", "conv1")
    osd   = make_element("nvdsosd", "osd", process_mode=0, display_text=True)
    conv2 = make_element("nvvideoconvert", "conv2")
    enc   = make_element("nvh264enc", "enc", bitrate=bitrate, gop_size=30)
    parse = make_element("h264parse",  "parse_out")
    muxer = make_element("mp4mux",     "mp4mux")
    sink  = make_element("filesink",   "sink", location=output_file, sync=False)

    for el in [filesrc, qtdemux, h264p, decoder,
               mux, infer, conv1, osd, conv2, enc, parse, muxer, sink]:
        pipeline.add(el)

    assert filesrc.link(qtdemux)

    # Pre-request mux sink pad before PLAYING (must happen before pad-added callback)
    mux_sink = mux.request_pad_simple("sink_0")
    assert mux_sink is not None, "failed to get nvstreammux sink_0"

    def on_qtdemux_pad(element, pad):
        caps = pad.get_current_caps() or pad.query_caps(None)
        if not caps:
            return
        if "video" not in caps.get_structure(0).get_name():
            return
        print(f"[ds] qtdemux video pad: {caps.to_string()[:80]}", flush=True)
        h264p_sink = h264p.get_static_pad("sink")
        if not h264p_sink.is_linked():
            pad.link(h264p_sink)

    qtdemux.connect("pad-added", on_qtdemux_pad)

    assert h264p.link(decoder)
    assert decoder.get_static_pad("src").link(mux_sink) == Gst.PadLinkReturn.OK

    assert mux.link(infer)
    assert infer.link(conv1)
    assert conv1.link(osd)
    assert osd.link(conv2)
    assert conv2.link(enc)
    assert enc.link(parse)
    assert parse.link(muxer)
    assert muxer.link(sink)

    infer.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER, osd_probe, classes
    )

    return pipeline


def main():
    args = parse_args()

    video_file  = args.video
    output_file = args.output or str(
        Path(video_file).with_suffix("") ) + "_ds.mp4"

    suffix       = "_".join(args.classes)
    config_name  = "config_infer_full.txt" if args.full else "config_infer.txt"
    infer_config = str(_ROOT / f"sam3ds_{suffix}" / config_name)

    if not os.path.exists(video_file):
        print(f"ERROR: video not found: {video_file}"); return
    if not os.path.exists(infer_config):
        print(f"ERROR: infer config not found: {infer_config}"); return

    print(f"[ds] video  : {video_file}")
    print(f"[ds] output : {output_file}")
    print(f"[ds] config : {infer_config}")
    print(f"[ds] classes: {args.classes}  {args.width}×{args.height}")

    Gst.init(None)
    pipeline = build_pipeline(
        video_file, output_file, infer_config,
        args.classes, args.width, args.height,
        args.bitrate, args.interval,
    )

    loop = GLib.MainLoop()
    done = threading.Event()

    def on_message(bus, msg):
        if msg.type == Gst.MessageType.EOS:
            print("[ds] EOS")
            loop.quit(); done.set()
        elif msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            print(f"[ds] ERROR: {err}  {dbg}")
            loop.quit(); done.set()
        elif msg.type == Gst.MessageType.WARNING:
            w, _ = msg.parse_warning()
            print(f"[ds] WARN: {w}", flush=True)

    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", on_message)

    def _shutdown(sig, frame):
        print("\n[ds] Stopping...")
        pipeline.send_event(Gst.Event.new_eos())

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    pipeline.set_state(Gst.State.PLAYING)
    loop.run()
    pipeline.set_state(Gst.State.NULL)

    size_mb = os.path.getsize(output_file) / 1e6 if os.path.exists(output_file) else 0
    print(f"[ds] Done → {output_file}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
