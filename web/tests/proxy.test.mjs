import test from "node:test";
import assert from "node:assert/strict";
import { proxyCareerRequest } from "../lib/career-proxy.mts";

const config = { baseUrl: "https://example-tunnel.trycloudflare.com", token: "test-only-service-key" };
const user = { userId: "platform-user-a" };
const request = (method = "GET", headers = {}, path = "/api/sessions") => new Request(`https://private-site.example${path}`, { method, headers: { ...(method === "POST" ? { origin: "https://private-site.example", "idempotency-key": "test-request-0001" } : {}), ...headers }, ...(method === "POST" ? { body: JSON.stringify({ user_goal: "匿名测试目标" }) } : {}) });

test("anonymous requests cannot reach the GPU even with spoofed browser identity", async () => {
  let calls = 0;
  const result = await proxyCareerRequest(request("GET", { "X-Career-User": "victim", "oai-authenticated-user-id": "victim" }), ["sessions"], null, config, async () => { calls++; return Response.json({}); });
  assert.equal(result.status, 401); assert.equal(calls, 0);
});

test("two users forward only their trusted Sites identity, with no browser credentials", async () => {
  const identities = [];
  const fetcher = async (url, init) => {
    assert.equal(String(url), "https://example-tunnel.trycloudflare.com/v1/sessions");
    const headers = new Headers(init.headers);
    identities.push(headers.get("X-Career-User"));
    assert.equal(headers.get("authorization"), "Bearer test-only-service-key");
    assert.equal(headers.get("cookie"), null);
    assert.equal(headers.get("oai-authenticated-user-id"), null);
    assert.equal(init.redirect, "error");
    return Response.json({ sessions: [] });
  };
  for (const userId of ["platform-user-a", "platform-user-b"]) {
    const result = await proxyCareerRequest(request("GET", { "X-Career-User": "victim", authorization: "Bearer browser-key", cookie: "private=cookie", "oai-authenticated-user-id": "victim" }), ["sessions"], { userId }, config, fetcher);
    assert.equal(result.status, 200);
    assert.match(result.headers.get("cache-control"), /no-store/);
    assert.doesNotMatch(await result.text(), /test-only-service-key/);
  }
  assert.deepEqual(identities, ["platform-user-a", "platform-user-b"]);
});

test("writes require the same origin and a durable idempotency key", async () => {
  let calls = 0;
  const fetcher = async () => { calls++; return Response.json({}); };
  assert.equal((await proxyCareerRequest(request("POST", { origin: "https://attacker.example" }), ["sessions"], user, config, fetcher)).status, 403);
  assert.equal((await proxyCareerRequest(request("POST", { origin: "null" }), ["sessions"], user, config, fetcher)).status, 403);
  assert.equal((await proxyCareerRequest(request("POST", { "sec-fetch-site": "cross-site" }), ["sessions"], user, config, fetcher)).status, 403);
  assert.equal((await proxyCareerRequest(request("POST", { "idempotency-key": "" }), ["sessions"], user, config, fetcher)).status, 400);
  assert.equal(calls, 0);
});

test("the identical explicit retry preserves body and idempotency key", async () => {
  const received = [];
  const fetcher = async (_, init) => { received.push({ key: new Headers(init.headers).get("idempotency-key"), body: await new Response(init.body).text() }); return Response.json({ id: "session1" }); };
  await proxyCareerRequest(request("POST"), ["sessions"], user, config, fetcher);
  await proxyCareerRequest(request("POST"), ["sessions"], user, config, fetcher);
  assert.equal(received.length, 2); assert.deepEqual(received[0], received[1]);
});

test("arbitrary paths, encoded traversal, and unsupported methods are rejected", async () => {
  let calls = 0;
  const fetcher = async () => { calls++; return Response.json({}); };
  for (const path of [["admin"], ["sessions", "..", "secret"], ["sessions", "%2e%2e"], ["sessions", "abc/../admin"], ["healthz"], ["sessions", "id", "unknown"]]) assert.equal((await proxyCareerRequest(request(), path, user, config, fetcher)).status, 404);
  assert.equal((await proxyCareerRequest(request("PUT"), ["sessions"], user, config, fetcher)).status, 404);
  assert.equal(calls, 0);
});

test("insecure or credential-bearing upstreams fail closed", async () => {
  for (const baseUrl of ["http://public-server.example", "https://user:pass@example.com", "https://example.com/subpath", "https://example.com/?secret=value", "invalid"]) assert.equal((await proxyCareerRequest(request(), ["sessions"], user, { ...config, baseUrl })).status, 503);
  assert.equal((await proxyCareerRequest(request(), ["sessions"], user, { baseUrl: config.baseUrl })).status, 503);
});

test("loopback HTTP is opt-in for local integration only", async () => {
  const fetcher = async () => Response.json({ sessions: [] });
  assert.equal((await proxyCareerRequest(request(), ["sessions"], user, { ...config, baseUrl: "http://127.0.0.1:8787" }, fetcher)).status, 503);
  assert.equal((await proxyCareerRequest(request(), ["sessions"], user, { ...config, baseUrl: "http://127.0.0.1:8787", allowInsecureLoopback: true }, fetcher)).status, 200);
});

test("upstream failures are never automatically retried or leaked", async () => {
  let calls = 0;
  const result = await proxyCareerRequest(request("POST"), ["sessions"], user, config, async () => { calls++; throw new Error("private path or service key"); });
  assert.equal(result.status, 502); assert.equal(calls, 1);
  assert.doesNotMatch(await result.text(), /private path|test-only-service-key/);
});

test("backend conflicts are returned without retry and with no-store", async () => {
  let calls = 0;
  const result = await proxyCareerRequest(request("POST"), ["sessions"], user, config, async () => { calls++; return Response.json({ error: { code: "stale_state", message: "已过期" } }, { status: 409 }); });
  assert.equal(result.status, 409); assert.equal(calls, 1);
  assert.equal((await result.json()).error.code, "stale_state");
});

test("exports pass only the accepted format and download headers", async () => {
  const fetcher = async url => { assert.equal(String(url), "https://example-tunnel.trycloudflare.com/v1/sessions/id/export?format=markdown"); return new Response("# 匿名规划", { headers: { "Content-Type": "text/markdown; charset=utf-8", "Content-Disposition": "attachment; filename=plan.md", "X-Private": "secret" } }); };
  const result = await proxyCareerRequest(request("GET", {}, "/api/sessions/id/export?format=markdown&url=https://attacker.example"), ["sessions", "id", "export"], user, config, fetcher);
  assert.equal(result.status, 200); assert.equal(result.headers.get("content-disposition"), "attachment; filename=plan.md"); assert.equal(result.headers.get("X-Private"), null);
  assert.equal((await proxyCareerRequest(request("GET", {}, "/api/sessions/id/export?format=html"), ["sessions", "id", "export"], user, config, fetcher)).status, 400);
});

test("declared oversized uploads never reach the GPU", async () => {
  let calls = 0;
  const result = await proxyCareerRequest(request("POST", { "content-length": String(26 * 1024 * 1024) }), ["sessions", "id", "uploads"], user, config, async () => { calls++; return Response.json({}); });
  assert.equal(result.status, 413); assert.equal(calls, 0);
});

test("streamed uploads enforce the body cap even without Content-Length", async () => {
  let calls = 0;
  const body = new ReadableStream({ start(controller) { for (let i = 0; i < 27; i++) controller.enqueue(new Uint8Array(1024 * 1024)); controller.close(); } });
  const req = new Request("https://private-site.example/api/sessions/id/uploads", { method: "POST", headers: { origin: "https://private-site.example", "idempotency-key": "test-request-0002" }, body, duplex: "half" });
  const result = await proxyCareerRequest(req, ["sessions", "id", "uploads"], user, config, async (_, init) => { calls++; await new Response(init.body).arrayBuffer(); return Response.json({}); });
  assert.equal(result.status, 413); assert.equal(calls, 1);
});
