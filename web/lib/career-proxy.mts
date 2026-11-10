/** Server-only gateway. Identity is supplied by the Sites server auth helper. */
export type ProxyUser = { userId: string } | null;
export type ProxyConfig = { baseUrl?: string; token?: string; allowInsecureLoopback?: boolean };
const WRITE_METHODS = new Set(["POST", "DELETE"]);
const MAX_BODY = 25 * 1024 * 1024 + 128 * 1024;
const ID = "[a-zA-Z0-9_-]{1,128}";
const ALLOWED = [
  new RegExp(`^sessions(?:/${ID})?$`),
  new RegExp(`^sessions/${ID}/(?:uploads|perceive|answers|plans|feedback|evaluations|events|versions|export)$`),
  new RegExp(`^tasks/${ID}$`),
];
export function proxyError(code: string, message: string, status: number) {
  return Response.json({ error: { code, message } }, { status, headers: { "Cache-Control": "no-store" } });
}
export async function proxyCareerRequest(request: Request, path: string[], user: ProxyUser, config: ProxyConfig, fetcher: typeof fetch = fetch): Promise<Response> {
  if (!user?.userId) return proxyError("unauthorized", "请先登录后继续。", 401);
  const route = path.join("/");
  if (!ALLOWED.some(pattern => pattern.test(route)) || !["GET", "POST", "DELETE"].includes(request.method)) return proxyError("not_found", "接口不存在。", 404);
  const write = WRITE_METHODS.has(request.method);
  if (write) {
    if (request.headers.get("origin") !== new URL(request.url).origin || request.headers.get("sec-fetch-site") === "cross-site") return proxyError("origin_denied", "请从本站页面提交。", 403);
    const key = request.headers.get("idempotency-key");
    if (!key || !/^[a-zA-Z0-9_.:-]{8,128}$/.test(key)) return proxyError("idempotency_required", "提交标识缺失，请刷新后重试。", 400);
  }
  let base: URL;
  try {
    base = new URL(config.baseUrl || "");
    if (base.username || base.password || base.search || base.hash || (base.pathname !== "/" && base.pathname !== "")) throw new Error();
    const localAllowed = config.allowInsecureLoopback && base.protocol === "http:" && ["127.0.0.1", "localhost", "[::1]"].includes(base.hostname);
    if (base.protocol !== "https:" && !localAllowed) throw new Error();
    if (!config.token || /[\r\n]/.test(config.token)) throw new Error();
  } catch { return proxyError("service_unconfigured", "服务尚未连接，请联系管理员。", 503); }
  const incoming = new URL(request.url);
  const target = new URL(`/v1/${route}`, base);
  if (route.endsWith("/export")) {
    const format = incoming.searchParams.get("format") || "json";
    if (!["json", "markdown"].includes(format)) return proxyError("invalid_format", "不支持的导出格式。", 400);
    target.searchParams.set("format", format);
  }
  const headers = new Headers({ Authorization: `Bearer ${config.token}`, "X-Career-User": user.userId, Accept: request.headers.get("accept") || "application/json" });
  for (const name of ["content-type", "idempotency-key"]) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }
  const declaredLength = Number(request.headers.get("content-length") || 0);
  if (declaredLength > MAX_BODY) return proxyError("file_too_large", "单个文件不能超过 25 MiB。", 413);
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 45000);
  let exceeded = false;
  let body: ReadableStream<Uint8Array> | undefined;
  if (write && request.body) {
    let bytes = 0;
    body = request.body.pipeThrough(new TransformStream<Uint8Array, Uint8Array>({ transform(chunk, sink) {
      bytes += chunk.byteLength;
      if (bytes > MAX_BODY) { exceeded = true; controller.abort(); sink.error(new Error("body limit")); return; }
      sink.enqueue(chunk);
    } }));
  }
  try {
    const init: RequestInit & { duplex?: "half" } = { method: request.method, headers, body, redirect: "error", signal: controller.signal, ...(body ? { duplex: "half" as const } : {}) };
    const upstream = await fetcher(target, init);
    const responseHeaders = new Headers({ "Cache-Control": "no-store, private", Vary: "Cookie", "X-Content-Type-Options": "nosniff" });
    for (const name of ["content-type", "content-disposition"]) {
      const value = upstream.headers.get(name);
      if (value) responseHeaders.set(name, value);
    }
    return new Response(upstream.body, { status: upstream.status, headers: responseHeaders });
  } catch {
    if (exceeded) return proxyError("file_too_large", "单个文件不能超过 25 MiB。", 413);
    return proxyError(controller.signal.aborted ? "service_timeout" : "service_unavailable", "暂时无法连接规划服务。提交可能已被接收；请使用原提交标识重试或稍后恢复会话。", controller.signal.aborted ? 504 : 502);
  } finally { clearTimeout(timer); }
}
