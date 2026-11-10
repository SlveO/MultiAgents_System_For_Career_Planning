import { env } from "cloudflare:workers";
import { getChatGPTUser } from "../../chatgpt-auth";
import { proxyCareerRequest } from "../../../lib/career-proxy.mts";

export const dynamic = "force-dynamic";
type Context = { params: Promise<{ path: string[] }> };
async function handle(request: Request, context: Context) {
  const user = await getChatGPTUser();
  const { path } = await context.params;
  return proxyCareerRequest(request, path, user, { baseUrl: env.CAREER_API_BASE_URL, token: env.CAREER_SERVICE_TOKEN, allowInsecureLoopback: env.CAREER_ALLOW_INSECURE_LOOPBACK === "true" });
}
export const GET = handle;
export const POST = handle;
export const DELETE = handle;
