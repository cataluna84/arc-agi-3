"""Build qwen_rtx6000_probe.ipynb for exp004 Phase 0.

Run:
    uv run python experiments/exp004_qwen_agent/rtx6000_probe_kernel/build_notebook.py

The generated notebook is a dev/probe kernel only. It verifies the
Kaggle RTX 6000 runtime envelope with internet disabled before we spend
time adapting Qwen direct policy to the new accelerator pool.
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent


def source_lines(source: str) -> list[str]:
    return textwrap.dedent(source).strip("\n").splitlines(keepends=True)


def make_cell(cell_type: str, source: str) -> dict[str, Any]:
    cell: dict[str, Any] = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source_lines(source),
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    return cell


def main() -> None:
    cells = [
        make_cell(
            "markdown",
            """
            # exp004 Phase 0: RTX 6000 probe

            Dev/probe notebook for ARC-AGI-3's upgraded RTX 6000
            (`g4-standard-48`) accelerator pool.

            Rules this probe is designed around:
            - notebook runtime <= 9h;
            - internet disabled;
            - external data / pretrained models must be freely and
              publicly available for prize-relevant submissions;
            - submission parquet is generated only by the competition
              harness, so this probe is **not** intended for submission.
            """,
        ),
        make_cell(
            "code",
            """
            from __future__ import annotations

            import importlib
            import importlib.metadata as md
            import json
            import os
            import platform
            import shutil
            import subprocess
            import sys
            import time
            from pathlib import Path

            OUT = Path("/kaggle/working/rtx6000_probe_summary.json")
            summary = {
                "probe": "exp004_qwen_rtx6000",
                "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "env": {},
                "commands": {},
                "packages": {},
                "gpu": {},
                "qwen": {},
            }


            def run(cmd: list[str], timeout: int = 120) -> dict:
                try:
                    proc = subprocess.run(
                        cmd,
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=timeout,
                    )
                    return {
                        "returncode": proc.returncode,
                        "stdout": proc.stdout[-8000:],
                        "stderr": proc.stderr[-8000:],
                    }
                except Exception as exc:
                    return {"error": f"{type(exc).__name__}: {exc}"}


            def pkg_version(import_name: str, dist_name: str | None = None) -> str:
                dist_name = dist_name or import_name
                try:
                    return md.version(dist_name)
                except Exception:
                    try:
                        mod = importlib.import_module(import_name)
                        return str(getattr(mod, "__version__", "importable_unknown"))
                    except Exception as exc:
                        return f"missing: {type(exc).__name__}: {exc}"


            def save() -> None:
                summary["finished_utc"] = time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ",
                    time.gmtime(),
                )
                OUT.write_text(json.dumps(summary, indent=2, sort_keys=True))
                print(json.dumps(summary, indent=2, sort_keys=True))
                print(f"wrote {OUT}")
            """,
        ),
        make_cell(
            "code",
            """
            summary["env"].update(
                {
                    "python": sys.version,
                    "platform": platform.platform(),
                    "executable": sys.executable,
                    "cwd": os.getcwd(),
                    "kaggle_is_competition_rerun": os.getenv(
                        "KAGGLE_IS_COMPETITION_RERUN",
                    ),
                    "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
                    "qwen_model_path": os.getenv("QWEN_MODEL_PATH"),
                    "probe_qwen_config": os.getenv("PROBE_QWEN_CONFIG", "1"),
                    "probe_qwen_load": os.getenv("PROBE_QWEN_LOAD", "1"),
                }
            )

            input_root = Path("/kaggle/input")
            if input_root.exists():
                files = [str(p) for p in input_root.rglob("*") if p.is_file()]
                summary["env"]["input_file_count"] = len(files)
                summary["env"]["input_file_sample"] = files[:200]
            else:
                summary["env"]["input_file_count"] = 0

            summary["commands"]["df_h"] = run(["df", "-h"], timeout=60)
            summary["commands"]["free_h"] = run(["free", "-h"], timeout=60)
            summary["commands"]["uname"] = run(["uname", "-a"], timeout=60)
            print("environment probe complete")
            """,
        ),
        make_cell(
            "code",
            """
            def find_dir_with(pattern: str, roots: list[Path]) -> Path | None:
                for root in roots:
                    if not root.exists():
                        continue
                    if any(root.glob(pattern)):
                        return root
                    for hit in root.rglob(pattern):
                        return hit.parent
                return None


            summary["overlay_install"] = {"enabled": os.getenv("PROBE_INSTALL_OVERLAYS", "1")}

            if os.getenv("PROBE_INSTALL_OVERLAYS", "1") == "1":
                wheel_roots = [
                    Path("/kaggle/input/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"),
                    Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels"),
                    Path("/kaggle/input"),
                ]
                comp_wheels = find_dir_with("pillow-*.whl", wheel_roots)
                if comp_wheels is not None:
                    pillow_target = Path("/tmp/_pillow_pkg")
                    pillow_target.mkdir(parents=True, exist_ok=True)
                    res = run(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "--no-index",
                            f"--find-links={comp_wheels}",
                            "--upgrade",
                            "--target",
                            str(pillow_target),
                            "pillow",
                        ],
                        timeout=180,
                    )
                    summary["overlay_install"]["pillow"] = {
                        "wheel_dir": str(comp_wheels),
                        "target": str(pillow_target),
                        "result": res,
                    }
                    sys.path.insert(0, str(pillow_target))
                    for mod in [m for m in list(sys.modules) if m.startswith("PIL")]:
                        del sys.modules[mod]
                else:
                    summary["overlay_install"]["pillow"] = {"error": "wheels not found"}

                tx_roots = [
                    Path("/kaggle/input/arc-agi-3-transformers-wheels"),
                    Path("/kaggle/input/datasets/cataluna84/arc-agi-3-transformers-wheels"),
                    Path("/kaggle/input"),
                ]
                tx_wheels = find_dir_with("transformers-*.whl", tx_roots)
                if tx_wheels is not None:
                    tx_target = Path("/tmp/_transformers_pkg")
                    tx_target.mkdir(parents=True, exist_ok=True)
                    res = run(
                        [
                            sys.executable,
                            "-m",
                            "pip",
                            "install",
                            "--no-index",
                            f"--find-links={tx_wheels}",
                            "--upgrade",
                            "--target",
                            str(tx_target),
                            "--no-deps",
                            "transformers",
                            "tokenizers",
                            "accelerate",
                            "huggingface_hub",
                            "safetensors",
                            "regex",
                            "filelock",
                            "fsspec",
                            "pyyaml",
                            "tqdm",
                        ],
                        timeout=300,
                    )
                    summary["overlay_install"]["transformers"] = {
                        "wheel_dir": str(tx_wheels),
                        "target": str(tx_target),
                        "result": res,
                    }
                    sys.path.insert(0, str(tx_target))
                    purge_roots = {
                        "transformers",
                        "tokenizers",
                        "accelerate",
                        "huggingface_hub",
                        "safetensors",
                    }
                    for mod in [
                        m for m in list(sys.modules) if m.split(".")[0] in purge_roots
                    ]:
                        del sys.modules[mod]
                else:
                    summary["overlay_install"]["transformers"] = {
                        "error": "wheels not found"
                    }

            print("offline overlay install probe complete")
            """,
        ),
        make_cell(
            "code",
            """
            summary["commands"]["nvidia_smi_L"] = run(["nvidia-smi", "-L"], timeout=60)
            summary["commands"]["nvidia_smi"] = run(["nvidia-smi"], timeout=60)
            summary["commands"]["gpu_query"] = run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader",
                ],
                timeout=60,
            )

            try:
                import torch

                devices = []
                for idx in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(idx)
                    devices.append(
                        {
                            "index": idx,
                            "name": torch.cuda.get_device_name(idx),
                            "total_memory_gb": round(
                                props.total_memory / (1024**3),
                                2,
                            ),
                            "major": props.major,
                            "minor": props.minor,
                            "multi_processor_count": props.multi_processor_count,
                        }
                    )
                summary["gpu"].update(
                    {
                        "torch_version": torch.__version__,
                        "cuda_available": torch.cuda.is_available(),
                        "cuda_version": torch.version.cuda,
                        "device_count": torch.cuda.device_count(),
                        "devices": devices,
                    }
                )
            except Exception as exc:
                summary["gpu"]["torch_error"] = f"{type(exc).__name__}: {exc}"

            print("gpu probe complete")
            """,
        ),
        make_cell(
            "code",
            """
            packages = {
                "torch": "torch",
                "transformers": "transformers",
                "accelerate": "accelerate",
                "tokenizers": "tokenizers",
                "safetensors": "safetensors",
                "vllm": "vllm",
                "PIL": "Pillow",
                "flash_attn": "flash-attn",
                "flashinfer": "flashinfer-python",
                "xformers": "xformers",
                "triton": "triton",
            }

            for import_name, dist_name in packages.items():
                summary["packages"][import_name] = pkg_version(import_name, dist_name)

            print("package probe complete")
            """,
        ),
        make_cell(
            "code",
            """
            def find_qwen_model_path() -> str | None:
                env_path = os.getenv("QWEN_MODEL_PATH")
                if env_path and Path(env_path).exists():
                    return env_path

                roots = [
                    Path("/kaggle/input/qwen3-6-35b-a3b-bf16"),
                    Path("/kaggle/input/datasets/cataluna84/qwen3-6-35b-a3b-bf16"),
                    Path("/kaggle/input/models"),
                    Path("/kaggle/input"),
                ]
                for root in roots:
                    if not root.exists():
                        continue
                    for cfg in root.rglob("config.json"):
                        text = cfg.read_text(errors="ignore")[:4000].lower()
                        path_text = str(cfg.parent).lower()
                        if "qwen" in text or "qwen" in path_text:
                            return str(cfg.parent)
                return None


            model_path = find_qwen_model_path()
            summary["qwen"]["model_path"] = model_path

            if model_path and os.getenv("PROBE_QWEN_CONFIG", "1") == "1":
                try:
                    from transformers import AutoConfig

                    t0 = time.time()
                    cfg = AutoConfig.from_pretrained(
                        model_path,
                        local_files_only=True,
                        trust_remote_code=True,
                    )
                    summary["qwen"]["config_load_s"] = round(time.time() - t0, 3)
                    summary["qwen"]["model_type"] = getattr(cfg, "model_type", None)
                    summary["qwen"]["architectures"] = getattr(
                        cfg,
                        "architectures",
                        None,
                    )
                except Exception as exc:
                    summary["qwen"]["config_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

            if model_path and os.getenv("PROBE_QWEN_LOAD", "1") == "1":
                try:
                    import torch
                    from transformers import AutoModelForImageTextToText, AutoProcessor

                    t0 = time.time()
                    processor = AutoProcessor.from_pretrained(
                        model_path,
                        local_files_only=True,
                        trust_remote_code=True,
                    )
                    summary["qwen"]["processor_load_s"] = round(
                        time.time() - t0,
                        3,
                    )

                    t0 = time.time()
                    model = AutoModelForImageTextToText.from_pretrained(
                        model_path,
                        dtype=torch.bfloat16,
                        device_map="auto",
                        local_files_only=True,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                    )
                    summary["qwen"]["model_load_s"] = round(time.time() - t0, 3)
                    summary["qwen"]["model_class"] = type(model).__name__
                    summary["qwen"]["device_map"] = getattr(
                        model,
                        "hf_device_map",
                        None,
                    )
                    if torch.cuda.is_available():
                        summary["qwen"]["cuda_allocated_gb_after_load"] = round(
                            torch.cuda.memory_allocated() / 1e9,
                            3,
                        )

                    def apply_template(messages, *, enable_thinking: bool) -> str:
                        try:
                            return processor.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                                enable_thinking=enable_thinking,
                            )
                        except TypeError:
                            return processor.apply_chat_template(
                                messages,
                                tokenize=False,
                                add_generation_prompt=True,
                            )


                    def generate_reply(
                        *,
                        label: str,
                        messages: list[dict],
                        max_new_tokens: int,
                        enable_thinking: bool,
                    ) -> dict:
                        text = apply_template(
                            messages,
                            enable_thinking=enable_thinking,
                        )
                        inputs = processor(
                            text=text,
                            return_tensors="pt",
                        ).to(model.device)
                        t0 = time.time()
                        with torch.inference_mode():
                            out = model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=False,
                                use_cache=True,
                            )
                        elapsed = time.time() - t0
                        gen_tokens = out[0, inputs["input_ids"].shape[1] :]
                        reply = processor.batch_decode(
                            [gen_tokens],
                            skip_special_tokens=True,
                        )[0]
                        return {
                            "label": label,
                            "enable_thinking": enable_thinking,
                            "max_new_tokens": max_new_tokens,
                            "generate_s": round(elapsed, 3),
                            "tokens_per_s": round(max_new_tokens / elapsed, 3)
                            if elapsed > 0
                            else None,
                            "reply": reply,
                        }


                    action_messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are an ARC-AGI-3 controller. Do not reason. "
                                "Reply with one valid action token only."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Available actions: ACTION1, ACTION2, ACTION3. "
                                "Choose ACTION1. Output exactly ACTION1."
                            ),
                        },
                    ]
                    json_messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are an ARC-AGI-3 controller. Do not reason. "
                                "Reply with compact JSON only."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                "Available actions: ACTION1, ACTION2, ACTION3. "
                                "Choose ACTION1. Output {'action':'ACTION1'}."
                            ),
                        },
                    ]
                    summary["qwen"]["generation_probes"] = [
                        generate_reply(
                            label="action_no_thinking_8",
                            messages=action_messages,
                            max_new_tokens=8,
                            enable_thinking=False,
                        ),
                        generate_reply(
                            label="json_no_thinking_16",
                            messages=json_messages,
                            max_new_tokens=16,
                            enable_thinking=False,
                        ),
                        generate_reply(
                            label="action_default_8",
                            messages=action_messages,
                            max_new_tokens=8,
                            enable_thinking=True,
                        ),
                    ]
                    del model
                    del processor
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as exc:
                    summary["qwen"]["model_load_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )

            print("qwen probe complete")
            """,
        ),
        make_cell(
            "code",
            """
            save()
            """,
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }

    out_path = ROOT / "qwen_rtx6000_probe.ipynb"
    out_path.write_text(json.dumps(notebook, indent=1))
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
