# 面向大学生职业发展规划的多模态智能体

本项目只按原版立项申报书完成职业规划系统，不再开展大小模型、单体/群体或单轮/多轮对照研究。

服务器已接管，后续代码与进度以服务器仓库及 GitHub main 为准，本机不再作为开发端。

## 运行方式

唯一入口为服务器Ubuntu终端CLI。最终规划推理由DeepSeek API完成；其他需要模型的模块使用服务器本地GPU模型，禁止CPU模型加载、自动CPU回退和运行时自动下载权重。文件解析、规则与关键词检索不属于模型加载。

```bash
python -m venv .venv
source .venv/bin/activate
# 先按服务器现有CUDA环境安装匹配的torch和torchvision；已有验证环境不要重装。
pip install -r requirements-gpu.txt
export DEEPSEEK_API_KEY='YOUR_KEY'
export LOCAL_MODEL_DEVICE=cuda:1
export VISION_MODEL_PATH=/path/to/existing/Qwen3-VL-2B-Instruct
export GUIDANCE_MODEL_PATH=/path/to/existing/Qwen3-4B-Instruct-2507
export GUIDANCE_MAX_ROUNDS=4
python -m project.assistant_cli --goal "获得后端实习" --text "本科大三，会Python和SQL"
```

上面的模型路径是占位符，必须改为服务器既有权重目录；密钥仅存环境变量或本地.env，不提交Git。非模型开发/离线测试只需 `pip install -r requirements.txt`，这不是CPU模型部署方案。GPU依赖包含基础依赖，不再需要API/Web依赖文件。

文档：`python -m project.assistant_cli --goal "职业规划" --docs examples/sample_profile.txt`。
图片：`python -m project.assistant_cli --goal "分析岗位要求" --images /path/to/poster.png`。
支持txt、md、csv、tsv、pdf、docx、xlsx；扫描PDF暂未实现自动视觉解析。音频为可选本地GPU功能，需已有Whisper权重及对应运行依赖。

## 当前能力与缺口

已保留：文本/文档提取、图片代理、八维度画像、65条职业知识、DeepSeek规划、三档反馈、独立展示版本、SQLite与脱敏JSONL日志。
本地需求引导已实现：先感知材料，针对缺失/矛盾单题追问，区分未知/拒答/明确没有，回答带来源进入规划。2026-10-02 已完成 Qwen3-4B/L20 三例真实引导验证；反馈输出适配已通过 100 项离线回归和三档本地 GPU 小样本验收。

2026-10-05 按授权完成真实规划接本地反馈演示：18 项检查通过、752→2204 字、一次 API 和一次本地生成，原规划及 CLI/SQLite/JSONL 版本一致。此前单次图片→GPU→DeepSeek→CLI 展示及持久化验证也已通过。上述结果不代表广泛模型质量、长期稳定性或人工满意度验收。

引导最多默认 4 轮（`GUIDANCE_MAX_ROUNDS` 可设 0–16）；输入“结束引导”可停止追问并继续规划。`--answers-json` 保留八维度键名，接受字符串答案，部分答案只补必要缺口；明确跳过引导用 `--no-follow-up`（仍会正常感知材料及请求最终规划，不是离线模式）。同轮补充信息中仅部分缺少原文依据时，会提示并忽略这些内容，继续使用其余已确认信息。模型未配置、加载失败、响应结构或问题引用无效，或本轮全部补充信息缺少原文依据时，须输入“继续”才会带未解决项进入规划，否则退出。现有服务器私有配置未自动写入新模型路径；真实运行前应按授权核对设备占用。

反馈为“过短”时展开已有内容，为“过于详细”时压缩表达；成功后显示第 2 版。“合适”沿用原文且不调用模型。模型只选择原字段引用，原规划不覆盖；模型输出无效、加载失败或存储失败会提示并返回非零。保留必要内容后无法再缩短时明确继续使用原文。`FEEDBACK_MODEL_PATH` 可指定已有权重，留空时复用显式配置的 `GUIDANCE_MODEL_PATH`。

DeepSeek失败时CLI返回非零状态，不把内部诊断模板当作最终规划。现有模型名配置保留为 `deepseek-v4-flash`；此别名不保证固定服务端版本。

服务器已有模型：Qwen3-VL-2B-Instruct（2B）、Qwen3-4B-Instruct-2507（4B）、Qwen3-VL-8B-Instruct（8B）、Llama-3-8B-Instruct（8B）。目录存在不等于均已验证；当前图片链路使用 2B。权重不上传 GitHub。

## 验证与文档

```bash
python -m project.assistant_cli --help
python -m compileall -q project scripts/models
python -m unittest discover -s project/tests -v
```

离线测试使用 Fake 或 mock，不加载真实模型。执行范围见 `dataset/completion_protocol.json`。

`docs/` 为服务器本地计划、交接与验收文档目录，已加入 `.gitignore`，不提交 Git。运行证据位于同样忽略的 `data/`；本地文档和证据继续保留。
