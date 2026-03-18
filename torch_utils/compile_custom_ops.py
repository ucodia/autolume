import argparse
import os
import sys

import torch

from torch_utils import custom_ops

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ["TORCH_EXTENSIONS_DIR"] = os.path.join(PROJECT_ROOT, "torch_extensions")

OPS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ops")

CUSTOM_OPS = [
    dict(
        module_name="upfirdn2d_plugin",
        sources=["upfirdn2d.cpp", "upfirdn2d.cu"],
        headers=["upfirdn2d.h"],
    ),
    dict(
        module_name="bias_act_plugin",
        sources=["bias_act.cpp", "bias_act.cu"],
        headers=["bias_act.h"],
    ),
    dict(
        module_name="filtered_lrelu_plugin",
        sources=["filtered_lrelu.cpp", "filtered_lrelu_wr.cu", "filtered_lrelu_rd.cu", "filtered_lrelu_ns.cu"],
        headers=["filtered_lrelu.h", "filtered_lrelu.cu"],
    ),
]


def get_local_arch():
    if not torch.cuda.is_available():
        print("Error: No CUDA GPU detected. Pass --arch explicitly.", file=sys.stderr)
        sys.exit(1)
    major, minor = torch.cuda.get_device_capability(torch.cuda.current_device())
    return f"{major}.{minor}"


def main():
    parser = argparse.ArgumentParser(description="Pre-compile custom CUDA ops")
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Semicolon-separated CUDA arch list (e.g. '7.5;8.6;8.9;12.0'). "
             "Defaults to the local GPU's compute capability.",
    )
    args = parser.parse_args()

    arch_list = args.arch if args.arch else get_local_arch()
    os.environ["TORCH_CUDA_ARCH_LIST"] = arch_list

    print(f"TORCH_EXTENSIONS_DIR = {os.environ['TORCH_EXTENSIONS_DIR']}")
    print(f"TORCH_CUDA_ARCH_LIST = {arch_list}")
    print()

    custom_ops.verbosity = "full"

    for op in CUSTOM_OPS:
        plugin = custom_ops.get_plugin(
            module_name=op["module_name"],
            sources=op["sources"],
            headers=op["headers"],
            source_dir=OPS_DIR,
            extra_cuda_cflags=["--use_fast_math", "--allow-unsupported-compiler"],
        )
        if plugin is None:
            print(f"Failed to compile {op['module_name']}", file=sys.stderr)
            sys.exit(1)

    print()
    print("All custom ops compiled successfully.")


if __name__ == "__main__":
    main()
