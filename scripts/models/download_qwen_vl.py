"""Download the optional local vision model used by the GPU profile."""

from modelscope import snapshot_download


if __name__ == "__main__":
    snapshot_download(
        "Qwen/Qwen3-VL-2B-Instruct",
        local_dir="./models/Qwen3-VL-2B-Instruct",
    )
