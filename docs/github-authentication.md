# 服务器 GitHub 持久化认证

本仓库原来使用 `credential.helper=cache --timeout=28800`，凭据仅保存在内存，不能跨缓存失效或重启长期使用。2026-10-01 改为准备仓库专用 SSH Deploy Key：取代码仍走原 HTTPS 地址，推送走 `ssh://git@ssh.github.com:443/SlveO/MultiAgents_System_For_Career_Planning.git`。是否完成 GitHub 端登记、实际推送及远端哈希核验，以服务器 `data/maintenance/github-auth-20261001/` 和交付消息为准。

## 一次配置与持续使用

1. 在服务器为本仓库生成独立 Ed25519 密钥。当前私钥保存在 `.git/server-auth/career_planning_ed25519`，目录权限 0700、私钥权限 0600；公钥是同名 `.pub` 文件。私钥不提交 Git、不发送聊天、不复制到参考资料或日志。
2. 仓库管理员在 GitHub 仓库 Settings → Deploy keys → Add deploy key 中添加公钥，勾选 **Allow write access**。无需把账户密码或 Token 交给 Agent。
3. 本地 `.git/config` 的 `core.sshCommand` 指定上述私钥、`IdentitiesOnly=yes`、`BatchMode=yes` 和专用 `known_hosts`；仅作用于本仓库，不改变用户全局 SSH 或其他仓库。
4. GitHub SSH 443 的 Ed25519 主机密钥按官方指纹核对，启用 `StrictHostKeyChecking=yes`；不关闭主机身份校验。
5. 添加成功后，先验证 SSH 身份，再刷新 main；远端没有意外修改时正常推送，最后用远端 `refs/heads/main` 完整哈希与本地核对。遵守分支保护，不强推。

该密钥为了非交互推送未设口令，不依赖 8 小时内存缓存或 ssh-agent；文件和 GitHub 授权保留时，重启后仍可使用。能读取私钥的进程具有此仓库相应权限。Deploy Key 不自动过期；停用服务器、怀疑泄露或更换密钥时，先在 GitHub 的 Deploy keys 页面撤销旧公钥，再重新登记新公钥。不要将此密钥复用于其他仓库。

认证不等于发布授权：后续提交、合入及推送仍按具体任务授权执行。认证也不能绕过沙箱网络限制或 GitHub 分支保护。迁移或重新克隆仓库时，`.git/server-auth/` 不会随 Git 内容复制，应重新配置认证。

回退前的本地配置保存在 `data/maintenance/github-auth-20261001/config-before.json`。若更换方案，按记录恢复 `core.sshCommand` 和 `remote.origin.pushurl`；不要清除其他项目的凭据或全局配置。

官方说明：[Deploy Keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/managing-deploy-keys)、[SSH 443](https://docs.github.com/en/authentication/troubleshooting-ssh/using-ssh-over-the-https-port)、[GitHub 主机公钥指纹](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/githubs-ssh-key-fingerprints)。
