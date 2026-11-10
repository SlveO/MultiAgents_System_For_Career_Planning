# 从 GitHub 到本机 ChatGPT Sites

本文件说明如何把服务器准备的 Demo 源码交给用户本机的 ChatGPT Sites 工作流。完整 Sites 仓库以 `web/` 为应用目录；GitHub 推送只是源码交付，不会自动创建或发布 Site。

## 当前交付边界

- 本次服务器工作是同步已验证的工作区，并提交、推送当前分支 `codex/proposal-expansion-20261105` 到现有 `origin`：<https://github.com/SlveO/MultiAgents_System_For_Career_Planning.git>。
- 不合并或推送 `main`，不强推，也不更改仓库可见性。分支与提交状态以 Git 的实际结果为准；不要预先记录虚构的 SHA。
- 后续由用户本机 ChatGPT 桌面 Sites 工作流创建私有 Site、同步并发布。服务器不直接发布 Sites。
- Demo 功能验证已由用户确认完成；这次交接不重复功能试跑，也不调用 GPU、DeepSeek 或其他项目 API。
- 前序清理移除的旧运行材料和兼容别名不影响当前源码交接；本次只同步已验证工作区，不重复清理。

## 架构与 Sites 入口

- `web/` 是 React、Vite/Vinext 和 Cloudflare Worker 前端。现有 Vinext Worker 入口支持 `/api` 代理，部署时需要保留 Worker。
- 构建输出为 `web/dist/server/index.js`（默认 `fetch` Worker handler）和 `web/dist/client`。不要只把 `client` 当静态站点发布，否则会丢失 Worker 的 API 代理。
- Python/GPU API、SQLite 数据和模型继续运行在 GPU 服务器；不部署到 Sites，也不新增 D1/R2。
- Sites 运行时支持 HTTP、HTTPS 和 WebSockets，不支持原生 TCP；部分框架、私有网络和后台服务也不受支持。当前网页以 Worker 提供服务，后端仍需由 Sites 服务端通过 HTTPS 访问。
- 当前 `web/.openai/hosting.json` 尚未登记 `project_id`，其中 `d1` 和 `r2` 保持 `null`。桌面端注册 Sites 项目后，再填入实际生成的项目 ID，不要猜测或手工编造。

官方参考：[ChatGPT Sites 文档](https://learn.chatgpt.com/docs/sites)。

## 克隆与构建

干净克隆需要 Node.js 22.13+ 和 npm；网页构建不需要 Python、CUDA、权重或真实 API 密钥：

```bash
git clone --branch codex/proposal-expansion-20261105 --single-branch https://github.com/SlveO/MultiAgents_System_For_Career_Planning.git
cd MultiAgents_System_For_Career_Planning/web
npm run install:ci
npm run build
```

`install:ci` 使用已提交的锁文件执行 `npm ci`，安装构建所需的开发和可选依赖，并隔离 sharp 对全局 libvips 的探测。构建使用 Vinext，不是 `next build`。

干净克隆不含服务器的 `.sites-runtime`；`readExecutionProfile` 遇到缺失配置会回退到 portable profile，因此不需要 `bash` 或 `timeout`。不要把服务器的 execution profile 或 `node_modules` 搬到本机。本轮没有实际执行跨平台安装，不声称 Windows 或 macOS 已验证。

## Sites 服务端运行变量

在 Sites 服务端环境中设置以下变量；不要把秘密写进前端、GitHub 或仓库 `.env`：

| 变量 | 设置值 |
|---|---|
| `CAREER_API_BASE_URL` | GPU API 可达的 HTTPS 根地址，不带 `/v1`，不能是 `127.0.0.1` 或 `localhost` |
| `CAREER_SERVICE_TOKEN` | 与 GPU API 配置相同的服务令牌，仅在两个服务端保存 |
| `CAREER_ALLOW_INSECURE_LOOPBACK` | `false` |

服务器 API 默认绑定 `127.0.0.1:8787`；本地单用户 Demo 的 `5180/8788` 不能作为云端 API 地址。服务器需另行提供 Sites 可访问的 HTTPS API 入口，并在服务器 `.env` 中保留 DeepSeek 凭证和本地权重路径。临时隧道停止后其地址会失效；本交接不承诺 API 服务或 HTTPS 入口已启动。

## 本机 Sites 部署步骤

1. 在 ChatGPT 桌面打开克隆项目中的 `web/` 应用。
2. 通过 Sites 工作流创建私有项目，并保存 Sites 实际生成的项目 ID。
3. 在 Sites 服务端配置上述三个运行变量。不要复制服务器 `.env`、DeepSeek key、权重或 GPU 环境。
4. 按 Sites 助手的流程在 `web/` 应用根目录构建、同步并发布。若助手要求应用源码位于仓库根目录，直接把 `web/` 子目录作为独立应用源导入并保留；不要迁移 Python 后端或改动页面。

可复制给桌面 Sites 工作流的交接提示：

> 打开这个 GitHub 克隆项目中的 `web/` 应用，复用现有源码和 Vinext Worker 构建配置，通过 Sites 工作流创建私有项目并保存实际生成的项目 ID，然后配置 Sites 服务端变量，按助手流程在 `web/` 应用根目录构建、同步和发布。若助手要求源码位于仓库根目录，请将 `web/` 作为独立应用源导入并保留；不要迁移 Python 后端或改动页面。不要只发布 `dist/client`，也不要复制服务器 `.env`。

本次尚未执行本机 Sites 项目创建、同步或发布；GitHub 提交推送也以服务器实际执行结果为准。仓库不会预置项目 ID 或发布网址。
