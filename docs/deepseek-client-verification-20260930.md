# DeepSeek 客户端修复与单次验证

日期：2026-09-30。基线：a092c045dd02c835e991796ab3140d034c2e7ca7。
验证时执行分支：codex/deepseek-proxy-fix-20260930，当时为基线上的未提交客户端/测试修改；2026-10-01 随服务器归一纳入版本管理。以下按发生顺序保留阶段结论，最终覆盖范围以后文“授权后的单次 API 验证”为准。

## 官方资料与范围

- https://api-docs.deepseek.com/：OpenAI 兼容端点为 /chat/completions，Bearer 鉴权；当前推荐 deepseek-flash，deepseek-v4-flash 仍接受但由当前 Flash 服务处理。本次保留用户已有模型配置，不把旧别名当作固定服务端模型版本。
- https://www.python-httpx.org/advanced/proxies/：可通过 proxy 参数显式指定代理；SOCKS5 需要可选依赖。
- https://www.python-httpx.org/environment_variables/：HTTP(S)_PROXY、ALL_PROXY、NO_PROXY 及 SSL_CERT_FILE/SSL_CERT_DIR 的作用。

先前失败是客户端构造时解析不适用的 ALL_PROXY=socks://，不代表 API 密钥错误。本次不改接口地址、提示词或生成参数，仅修复代理选择与配置异常类型。

## 代码变化

project/core/brain_client.py 的同步/流式请求共用 _http_client：先按目标协议选择环境代理，遵循 urllib 的大小写优先级和 NO_PROXY 主机匹配，只有无专用代理时才选择 ALL_PROXY。显式传给 HTTPX，避免扫描无关代理配置。没有改动 os.environ、全局代理或 .env。

不支持的有效代理协议报不可重试的 BrainConfigError，不自动直连。CA 文件/目录继续生效，TLS 校验未关闭。错误信息不携带代理凭据。NO_PROXY 使用标准库的主机/域名规则，本轮没有声称覆盖所有 HTTPX 扩展匹配语法。

新增六项离线测试：HTTPS 优先于无效 ALL_PROXY、NO_PROXY、无效代理脱敏错误、大小写优先及无代理、CA 配置、流式共用工厂。全套61项通过，CLI代码未变；compileall 和 git diff --check 通过。

## 单次真实结果

新目录：data/acceptance/cli-proxy-fix-20260930/；原失败目录 cli-once-20260930 保留。

- 仅一个匿名图片案例，进程级 BRAIN_RETRY_TIMES=0；无权重下载，无 CPU 模型回退。
- Qwen3-VL-2B 在 cuda:1 完成四项岗位事实提取，实际参数设备均为 cuda:1；约11.94秒。
- 客户端初始化通过；进入一次 HTTP 请求后约0.24秒抛 BrainHTTPError，code=BRAIN_HTTP，status_code=null。
- 未获得 HTTP 响应状态或规划文本；不能判断是代理、TLS、网络还是远端链路哪一环失败，也不能断言密钥无效。API密钥、模型ID可用性仍未通过真实验证。
- 总用时12.28秒，CLI退出码1，最终规划未生成；未重跑。反馈占位未被消费，不计人工质量评价。
- vision.json、brain-input.json、brain-error.json、result.json、cli-output.json、tests.log 保存证据。brain-input 是拟发送输入，不证明服务器已收到。

## 下一步

先做不含密钥的代理/TLS连通性诊断，区分 HTTP(S) 代理链路和服务器直连可达性，再由负责人确定网络路径及下一次真实调用范围。当前不改全局代理、不安装依赖、不自动追加 API 请求。需求引导和反馈适配实现尚未在本轮启动。

## 后续无密钥网络诊断

中断恢复后确认没有遗留诊断进程或结果，再执行一次检查；结果保存在 data/acceptance/network-check-20260930/results.json。

- DNS解析成功；配置的HTTP代理TCP连接成功。
- 服务器直连TLS握手成功，TLSv1.3且证书校验通过。
- 配置代理及直连两条路径访问根路径和 /models 均收到HTTP 401，TLS校验开启。请求不含Authorization，这是无密钥探测的正常鉴权拒绝，不是密钥无效证据。
- 当前两条网络路径均可达，之前的BRAIN_HTTP未在此次探测重现，不能据此确定原先的连接故障原因或保证生成请求成功。
- 未运行GPU、未访问生成接口、未改全局代理；密钥有效性、模型可用性及最终规划生成仍未验证。下一步需经授权执行一次真实规划请求；没有依据要求立即切换直连。

## 授权后的单次 API 验证

复用 cli-proxy-fix-20260930/brain-input.json 的原始规划输入，单独调用一次 DeepSeek，未重跑视觉模型，零重试。产物目录 data/acceptance/deepseek-only-20260930/，started.json 记录输入来源与 SHA-256，response.json 保存原文，parsed-plan.json 保存解析结果。

结果：6.67秒返回真实规划文本；原编排器 JSON 提取与30/90/180天路线校验通过。当前密钥与配置模型名在此次请求可用。规划引用了图片中的 SQL、仪表盘和周指标报告要求。

边界：这是复用上轮真实感知结果的单独 API 检查，不是重新完成了一次 CLI 端到端运行，也没有执行 CLI 展示、反馈与持久化闭环。未进行人工质量评分。人工快速检查发现，将“未提供 SQL 能力信息”写为“缺少 SQL 能力”过于确定，且 offer/面试率等指标受外部条件影响；这些是后续需求引导和规划质量需处理的事项，不因结构校验通过而忽略。
