import json
import logging
import sys
import time
from pathlib import Path

import mlflow
import onnx
import tensorrt as trt
import torch
import torch.nn as nn


LOGGER = logging.getLogger("onnx2trt")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s %(message)s")

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def export_to_onnx(cfg_unit, model, pred_dl, onnx_name):
    # — упрощённый ONNX-экспорт через PL.to_onnx() —
    batch = next(iter(pred_dl))  # dict: input_ids, attention_mask
    # переносим на устройство
    batch = {k: v.to(model.device) for k, v in batch.items()}
    input_ids = batch["input_ids"]  # shape e.g. (8, 512)
    attention_mask = batch["attention_mask"]  # shape e.g. (8, 512)

    # сохраняем ONNX
    onnx_path = Path(cfg_unit.path) / f"{onnx_name}.onnx"

    class EssayONNXWrapper(nn.Module):
        def __init__(self, pl_model):
            super().__init__()
            self.pl_model = pl_model

        def forward(self, input_ids, attention_mask):
            return self.pl_model(
                {"input_ids": input_ids, "attention_mask": attention_mask}
            )

    wrapper = EssayONNXWrapper(model).eval()

    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask),
        f=onnx_path.as_posix(),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "logits": {0: "batch"},
        },
        opset_version=17,
    )

    # логировать файл в MLflow
    mlflow.log_artifact(str(onnx_path), artifact_path="onnx_models")


def convert_to_tensorrt(
    onnx: str,
    batch_max: int = 32,
    seq_len: int | None = None,
    feature_dim: int | None = None,
    out_path: str | None = None,
    fp16: bool = True,
    int8: bool = False,
    workspace_mb: int = 4096,
    calibration_cache: str | None = None,
) -> str:
    """
    Build a TensorRT engine from `onnx` and save `*.plan` + meta JSON.

    Args
    ----
    onnx : str
        Path to ONNX graph.
    batch_max : int
        Max batch size accepted at inference.
    seq_len, feature_dim : int | None
        If the first input is shaped [B, S, F], pass both numbers to
        generate a tighter optimisation profile.  Otherwise they are ignored
        and the profile assumes fully dynamic [1..batch_max, …].
    out_path : str | None
        Where to write the .plan file (defaults to same stem as ONNX).
    fp16 / int8 : bool
        Allow half- or INT8-precision kernels.
    workspace_mb : int
        Scratch memory cap for algorithm search (MiB).
    calibration_cache : str | None
        Optional cache path when int8=True and graph lacks Q/DQ.
    """
    onnx_path = Path(onnx).expanduser().resolve()
    plan_path = Path(out_path or onnx_path.with_suffix(".plan")).resolve()
    json_path = plan_path.with_suffix(".json")

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    LOGGER.info("Parsing  %s", onnx_path)
    with onnx_path.open("rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                LOGGER.error(parser.get_error(i))
            sys.exit("ONNX parse failed – aborting.")

    # ── Builder config ───────────────────────────────────────────────────────────
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mb * 2**20)

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        LOGGER.info("FP16 kernels ON")
    else:
        LOGGER.info("FP16 kernels OFF")

    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
        if calibration_cache:
            calib = trt.Int8EntropyCalibrator2(
                calibration_stream=[],
                cache_file=calibration_cache,
                read_cache=Path(calibration_cache).exists(),
            )
            config.int8_calibrator = calib
        LOGGER.info("INT8 kernels ON (calib cache: %s)", calibration_cache)

    # ── Optimisation profile (dynamic batch) ─────────────────────────────────────
    # Collect all input tensors
    dynamic_inputs = [network.get_input(i) for i in range(network.num_inputs)]
    if len(dynamic_inputs) == 0:
        raise RuntimeError("No ONNX inputs found—cannot build a dynamic profile.")

    # We'll derive the shape from the first input;
    # assume all dynamic inputs share the same rank/pattern.
    base_shape = list(dynamic_inputs[0].shape)  # e.g. [-1, -1] or [-1, -1, 5]

    # Replace -1/batch_dim with `batch_max`
    base_shape[0] = batch_max
    # If this is a [B, S] graph, len(base_shape)==2, so only seq_len matters
    if seq_len and len(base_shape) >= 2:
        base_shape[1] = seq_len
    # If it were [B, S, F], you’d also set base_shape[2]=feature_dim, etc.
    if feature_dim and len(base_shape) >= 3:
        base_shape[2] = feature_dim

    # Now build the three vectors: min_shape, opt_shape, max_shape
    min_shape = [1 if d == batch_max else d for d in base_shape]
    opt_shape = [(batch_max // 2) if d == batch_max else d for d in base_shape]
    max_shape = base_shape  # e.g. [ batch_max, seq_len, (feature_dim) ]

    # Create exactly one optimization profile and attach it to every dynamic input
    profile = builder.create_optimization_profile()
    for inp in dynamic_inputs:
        name = inp.name  # “input_ids” first, then “attention_mask”
        profile.set_shape(name, tuple(min_shape), tuple(opt_shape), tuple(max_shape))
    config.add_optimization_profile(profile)

    # ── Build & serialise ────────────────────────────────────────────────────────
    LOGGER.info(
        "Building TensorRT engine  (batch≤%d, workspace=%d MiB)…", batch_max, workspace_mb
    )
    tic = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        sys.exit("Engine build failed.")

    plan_path.write_bytes(serialized_engine)
    dur = time.time() - tic
    LOGGER.info(
        "✓ Engine ready: %s  (%.1f s, %.2f MiB)",
        plan_path.name,
        dur,
        serialized_engine.nbytes / 2**20,
    )

    # meta-file for traceability
    meta = {
        "onnx": str(onnx_path),
        "engine": str(plan_path),
        "tensorrt_ver": trt.__version__,
        "batch_max": batch_max,
        "seq_len": seq_len,
        "feature_dim": feature_dim,
        "fp16": fp16 and builder.platform_has_fast_fp16,
        "int8": int8,
        "workspace_mb": workspace_mb,
        "build_time_s": round(dur, 2),
    }
    json_path.write_text(json.dumps(meta, indent=2))
    LOGGER.info("Meta-info  →  %s", json_path.name)
    return str(plan_path)


def get_onnx_input_dimensions(path_to_onnx):
    # 1) Load the ONNX graph
    model = onnx.load(path_to_onnx)

    # 2) Iterate over graph inputs and print each name & shape
    for inp in model.graph.input:
        name = inp.name
        shape_dims = []
        tensor_type = inp.type.tensor_type
        for dim in tensor_type.shape.dim:
            # dim.dim_value is zero if it’s symbolic/dynamic (e.g. -1), otherwise it’s the concrete integer.
            if dim.dim_value > 0:
                shape_dims.append(dim.dim_value)
            else:
                # dim.dim_param is the symbolic name, e.g. "batch_size" or "sequence"
                shape_dims.append(dim.dim_param if dim.dim_param else "dynamic")
        LOGGER.info(f"Input {name!r} has shape: {shape_dims}")
