# 验证说明

## 本机离线检查

```bash
python -m project.assistant_cli --help
python -m compileall -q project scripts/models
python -m unittest discover -s project/tests -v
```

网络与模型使用Fake/mock；CPU执行这些测试不等于在CPU上运行模型。基础依赖不包含torch。

## 服务器接收后

1. 确认接收commit、git status和当前工作目录；不重置原41项脏工作区。
2. 复用已验证的GPU虚拟环境，核对CUDA、torch/torchvision与模型目录，不自动升级。
3. 可运行 `python scripts/models/record_l20_env.py`，只记录环境，不加载权重或下载。
4. 设置LOCAL_MODEL_DEVICE、VISION_MODEL_PATH与DEEPSEEK_API_KEY；先CLI --help。
5. 在明确批准单次真实运行后，以一份匿名图片及目标启动CLI。确认感知使用指定CUDA、本地权重，最终规划来自DeepSeek；不存在CPU回退。
6. 记录成功/失败、模型及版本、延迟和必要输出。没有真实运行就记“未验证”。

普通PDF解析只提取文本，不代表扫描件/图表已由视觉模型理解。反馈记录不代表重新生成。
不再执行旧架构比较、pilot或盲评包生成命令。
