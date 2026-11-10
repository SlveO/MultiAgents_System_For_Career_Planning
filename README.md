# 多模态职业规划桌面 Demo

本项目当前用于申报功能的桌面内部演示。主流程覆盖材料理解、追问、职业规划、反馈、图谱与来源、导出和会话恢复。HTTP 接口与 Sites 对接合同见 [WEB_CONTRACT.md](WEB_CONTRACT.md)。

## GitHub → 本机 ChatGPT Sites

服务器上的 GitHub 源码交付与本机 Sites 部署步骤见 [SITES_HANDOFF.md](SITES_HANDOFF.md)。从完整仓库创建 Sites 时，应用目录是 `web/`，不是仓库根目录；GitHub 推送本身不会创建或发布 Site。

## 本地启动

在仓库根目录运行：

```bash
bash scripts/web/start-local-demo.sh --check
bash scripts/web/start-local-demo.sh
```

检查命令只确认已有工具、依赖、权重路径和密钥配置，不启动服务、不加载模型或调用 API。启动成功后打开 `http://127.0.0.1:5180`。网页只监听本机；API 为 `127.0.0.1:8788`，无需单独转发。

从自己的电脑访问服务器时，在 VS Code“端口”面板转发 **5180**，或建立 SSH 转发：

```bash
ssh -N -L 15180:127.0.0.1:5180 <服务器用户>@<服务器地址>
```

然后打开 `http://127.0.0.1:15180`。启动终端和转发通道都需保持运行。停止时在启动终端按 `Ctrl+C`；重启时再次运行启动命令。不要用全局进程终止命令停止其他服务。

CLI 调试帮助：`.venv-l20/bin/python -m project.assistant_cli --help`。

## 演示流程

1. 填写职业目标，可补充背景和阅读偏好。
2. 上传材料并声明材料主体；每个会话最多 3 份、每份 25 MiB，暂不支持音频。
3. 开始理解材料，按页面逐题回答追问，或用已有信息继续。
4. 生成职业规划；规划使用 DeepSeek，材料理解和追问使用已有本地模型。
5. 查看图谱和来源，进行多轮反馈、切换版本并导出报告。
6. 在“我的记录”恢复已有会话。简要摘选、恢复完整阅读和“这样刚好”不调用模型；完整阅读状态下的“更详细”会调用本地模型扩展。

## 环境与数据

运行服务器本地 Demo 时复用现有 `.venv-l20`、Node 22.13+、配置文件和权重。模型配置由服务器本地 `.env` 提供；本地模型固定使用 `cuda:1`，不回退 CPU、不下载模型权重或替换已验证环境。干净克隆网页用于 Sites 时，可按 [SITES_HANDOFF.md](SITES_HANDOFF.md) 使用锁文件安装前端依赖；该构建不需要 Python、CUDA 或权重。DeepSeek 规划需要网络访问。

本 Demo 是单用户本机入口；能访问该入口的人共用同一演示身份。`dataset/demo/anonymous-export.json` 是匿名静态导出参考；默认数据位于 `data/local-demo/`，应用重启后保留，供“我的记录”恢复已保存的会话。会话、规划、评价和事件最多保留 30 天；材料成功提取后删除，未提取材料在清理器运行时于 24 小时内删除。主动删除会话后不可恢复。

## Sites 与服务器 API

Sites 运行前端 Worker；Python/GPU、SQLite 和模型留在服务器。服务器 API 需要由 Sites 服务端通过 HTTPS 到达，并在 Sites 环境中配置服务令牌。目录、构建产物和运行时变量详见 [SITES_HANDOFF.md](SITES_HANDOFF.md)。
