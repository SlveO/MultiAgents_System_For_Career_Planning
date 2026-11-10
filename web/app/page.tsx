import { getChatGPTUser, chatGPTSignInPath, chatGPTSignOutPath } from "./chatgpt-auth";
import { CareerWorkspace } from "./workspace";
import { ArrowUpRightIcon } from "lucide-react";
import { Button } from "@/components/ui/button";

export const dynamic = "force-dynamic";

export default async function Home() {
  const user = await getChatGPTUser();
  if (!user) {
    return <div className="app-shell">
      <a className="skip-link" href="#main-content">跳至主要内容</a>
      <header className="topbar">
        <a className="brand" href="/" aria-label="行向首页"><span className="brand-mark" aria-hidden="true"><ArrowUpRightIcon /></span><span>行向</span></a>
        <a className="login-data-link" href="#sign-in-privacy">数据说明</a>
      </header>
      <main className="sign-in-page" id="main-content" tabIndex={-1}>
        <div className="login-copy"><h1>从你的目标开始规划</h1><p>整理你的经历和材料，用几轮问答明确方向，再把规划拆成可以执行的下一步。</p></div>
        <ol className="login-journey" aria-label="职业规划流程"><li>准备材料</li><li>补充信息</li><li>查看规划</li></ol>
        <Button asChild size="lg"><a href={chatGPTSignInPath("/")} target="_top">使用 ChatGPT 登录</a></Button>
        <p className="login-access">仅限受邀用户。登录后查看你自己的材料与规划。</p>
        <section className="login-privacy" id="sign-in-privacy" aria-labelledby="privacy-title"><h2 id="privacy-title">关于你的材料与规划</h2><p>材料用于梳理背景和生成建议，生成规划时会调用 DeepSeek。你可以在工作台查看、导出或删除自己的记录。</p></section>
      </main>
      <footer className="page-footer"><span>行向 · 职业规划</span><span>规划为决策提供参考，选择由你作出。</span></footer>
    </div>;
  }
  return <CareerWorkspace displayName={user.displayName} signOutPath={chatGPTSignOutPath("/")} localDemo={user.userId === "career_local_demo"} />;
}
