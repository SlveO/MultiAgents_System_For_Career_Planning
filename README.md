# 面向大学生职业发展规划的多模态智能体

本项目只按原版立项申报书完成职业规划系统，不再开展大小模型、单体/群体或单轮/多轮对照研究。

## 运行方式

唯一入口为服务器Ubuntu终端CLI。最终规划推理由DeepSeek API完成；其他需要模型的模块使用服务器本地GPU模型，禁止CPU模型加载、自动CPU回退和运行时自动下载权重。文件解析、规则与关键词检索不属于模型加载。

```bash
python -m venv .venv
source .venv/bin/activate
# 先按服务器现有CUDA环境安装匹配的torch和torchvision；已有验证环境不要重装。
pip install -r requirements-gpu.txt
export DEEPSEEK_API_KEY='YOUR_KEY'
export LOCAL_MODEL_DEVICE=cuda:0
export VISION_MODEL_PATH=/path/to/existing/Qwen3-VL-2B-Instruct
python -m project.assistant_cli --goal "获得后端实习" --text "本科大三，会Python和SQL"
```

上面的模型路径是占位符，必须改为服务器既有权重目录；密钥仅存环境变量或本地.env，不提交Git。非模型开发/离线测试只需 `pip install -r requirements.txt`，这不是CPU模型部署方案。GPU依赖包含基础依赖，不再需要API/Web依赖文件。

文档：`python -m project.assistant_cli --goal "职业规划" --docs examples/sample_profile.txt`。
图片：`python -m project.assistant_cli --goal "分析岗位要求" --images /path/to/poster.png`。
支持txt、md、csv、tsv、pdf、docx、xlsx；扫描PDF暂未实现自动视觉解析。音频为可选本地GPU功能，需已有Whisper权重及对应运行依赖。

## 当前能力与缺口

已保留：文本/文档提取、图片代理、固定八题画像、65条职业知识、DeepSeek规划、三档反馈记录、SQLite与脱敏JSONL日志。
尚待服务器完成：按缺失信息引导学生、通过本地模型执行引导/输出适配、反馈后修改并显示规划、真实GPU与API联合验收。不能把固定问卷或反馈记录称为已完成这两个智能体。

DeepSeek失败时CLI返回非零状态，不把内部诊断模板当作最终规划。现有模型名配置保留为 `deepseek-v4-flash`，实际可用性需要服务器用真实API核验；本机不代跑API。

## 验证与文档

```bash
python -m project.assistant_cli --help
python -m compileall -q project scripts/models
python -m unittest discover -s project/tests -v
```

离线测试使用Fake或mock，不加载真实模型。详见[当前计划](docs/completion-plan.md)、[验证说明](docs/verification.md)、[接管说明](docs/server-handoff.md)和[整理报告](docs/cleanup-report-20260929.zh-CN.md)。
