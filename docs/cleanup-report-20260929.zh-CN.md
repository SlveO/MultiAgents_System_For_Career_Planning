# 两端对比与清理报告

日期：2026-09-29。分支：codex/original-proposal-cleanup-20260929。尚未commit、push、合并或修改远端。

## 执行依据

负责人批准直接删除无用材料，取消另存归档；仅接收本机b96d14d与服务器7995c628595300bd67b122b19df7571ec57e138b。其他未合并分支完全不采用。
本次在本机新清理分支上修改，沿用当前工作目录；没有创建整合worktree副本。已获批准的旧文档直接删除或重写，无关log.txt、output/及既有运行数据保留。

## 实际处置

- 删除旧架构实验运行器、Fake四组对照入口、研究Schema/协议及专用测试。不再保留旧实验执行链。
- 删除网页、FastAPI、JWT、重复多模态聊天管线、Docker、API依赖及专用测试，CLI为唯一产品入口。
- 删除旧论文、阶段审阅及双端临时交流文档；重写活动文档、README及AGENTS。
- 删除旧下载单模型脚本及闲置显存管理器；不接收服务器全套比较模型下载/调度工具。
- 删除本次检查产生的、已被放弃的成员分支临时副本，不保留该分支的代码或测试产物；不触及其他既有运行记录。
- 保留65条职业知识、来源线索、已有文本/文档处理、日志和原始样例素材。旧素材目录名仍保留，但不再作为pilot任务。
- 保留可选音频/视频与关键词默认检索；没有为其额外安装模型或依赖。

## 真正采用的服务器成果

| 来源 | 本次采用 | 未采用 |
|---|---|---|
| 7995c62: project/experiments/local_models.py | AutoProcessor/Qwen3VL、本地文件限定、指定CUDA、BF16、chat template输入及输出裁切方式，适配到project/agents/image.py | 三组比较适配器、4B/8B调度、实验Schema、LMFE约束和轮次控制 |
| 7995c62: scripts/models/record_l20_env.py | 环境记录工具；将旧model_registry依赖改为当前配置 | 旧实验目录/权重布局的固定假设 |
| 7995c62: requirements-gpu.txt | Transformers兼容范围及已有torch/torchvision环境说明 | API、LMFE等已无活动消费者的依赖 |

不是全量merge。服务器原始真实结果仍留原处；本机不修改、删除或重新标注它们。

## 行为调整与小范围修正

- 本地模型设备必须是CUDA；图片、音频、可选向量加载不回退CPU。
- 图片及音频使用已有本地目录；图像加载local_files_only，禁运行时自动下载。
- 配置通过LOCAL_MODEL_DEVICE、VISION_MODEL_PATH、AUDIO_MODEL_PATH提供，不携带私人目录。
- CLI只把DeepSeek返回视为成功规划；失败或缺失最终输出返回1，不以模板冒充成果。
- SQLite连接使用显式关闭的上下文管理器，修正Windows临时文件占用风险。
- 持久化脱敏删除证据source字段，修正文档绝对路径进入会话记录的问题。
- 新增原版执行合同及CUDA-only/CLI失败状态测试；更新已过时的CPU回退和API依赖测试。

## 本轮验证

解释器：本机已有Python 3.10.19环境；未安装/升级依赖。

| 检查 | 结果 |
|---|---|
| python -B -m unittest discover -s project/tests -v | 当前保留套件55/55通过，无跳过；其中6项为新增服务器CLI边界测试 |
| python -m compileall -q project scripts/models | 通过 |
| python -X utf8 -m project.assistant_cli --help | 通过，中文帮助正常；Windows旧输出编码下需UTF-8模式 |
| git diff --check | 通过；LF/CRLF提示不属于错误 |
| 真实GPU、本地权重加载、真实DeepSeek调用 | 未运行，必须服务器验收 |
| 前端/API测试 | 随明确退出范围的产品代码删除，不作为仍需通过的套件 |

55项不等于旧研究全套测试，也不等于GPU验收。已有文本+文档Fake端到端通过，原路径持久化测试仍保留并通过，没有删除该失败用例掩盖问题。

## 尚未完成

固定八题仍不是本地模型需求引导；三档反馈仍只记录、没有修改输出。后续按completion-plan.md在服务器实现，不复用被放弃的成员分支。
本轮图片实现虽取自已用过的加载思路，但重组后尚无真实GPU证据。扫描PDF自动视觉解析、音视频完整验收也未完成。

最终推理只用DeepSeek的分工已落实为CLI边界；其余智能体使用本地模型是后续实现约束，不能把现有规则解析描述为已经完成本地模型智能体。

## 下一步

负责人查看最终差异与报告，确认是否批准提交/推送及目标共享分支。发布后服务器在新工作区接收同一commit，不覆盖原41项未提交工作。
服务器提交接收回执，并在授权范围内进行新CLI首次真实GPU/API检查，再按原版目标补齐功能。此前本机尚未结束交接职责。
