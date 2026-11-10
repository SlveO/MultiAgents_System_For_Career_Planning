import test from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { localDemoPlugin, LOCAL_DEMO_USER } from "../scripts/vite/local-demo-plugin.mjs";

const token = "local-test-token-which-is-at-least-32-characters";
async function listen(server) {
  await new Promise((resolve, reject) => { server.once("error", reject); server.listen(0, "127.0.0.1", resolve); });
  return `http://127.0.0.1:${server.address().port}`;
}
async function close(server) {
  server.closeAllConnections();
  await new Promise(resolve => server.close(resolve));
}

test("local demo enters its own identity and proxies writes without exposing its token", async () => {
  const received = [];
  const upstream = http.createServer(async (req, res) => {
    let body = "";
    for await (const chunk of req) body += chunk;
    received.push({ url: req.url, headers: req.headers, body });
    res.setHeader("Content-Type", "application/json"); res.end(JSON.stringify({ id: "demo-session" }));
  });
  let frontend;
  try {
    const baseUrl = await listen(upstream);
    let handler;
    localDemoPlugin({ enabled: true, baseUrl, token }).configureServer({ middlewares: { use(value) { handler = value; } } });
    frontend = http.createServer((req, res) => handler(req, res, () => {
      res.setHeader("Content-Type", "application/json");
      res.end(JSON.stringify({ id: req.headers["oai-authenticated-user-id"], email: req.headers["oai-authenticated-user-email"] }));
    }));
    const origin = await listen(frontend);
    const identity = await (await fetch(origin, { headers: { "oai-authenticated-user-id": "victim" } })).json();
    assert.equal(identity.id, LOCAL_DEMO_USER);
    assert.equal(identity.email, "demo@local.invalid");
    const response = await fetch(origin + "/api/sessions", { method: "POST", headers: { origin, "content-type": "application/json", "idempotency-key": "demo-write-00001", "x-career-user": "victim", authorization: "Bearer browser-forgery" }, body: JSON.stringify({ user_goal: "匿名目标" }) });
    assert.equal(response.status, 200);
    assert.doesNotMatch(await response.text(), new RegExp(token));
    assert.equal(received.length, 1);
    assert.equal(received[0].url, "/v1/sessions");
    assert.equal(received[0].headers.authorization, `Bearer ${token}`);
    assert.equal(received[0].headers["x-career-user"], LOCAL_DEMO_USER);
    assert.equal(JSON.parse(received[0].body).user_goal, "匿名目标");
    assert.equal((await fetch(origin + "/api/sessions", { method: "POST", headers: { origin: "https://attacker.invalid", "idempotency-key": "demo-write-00002" }, body: "{}" })).status, 403);
    assert.equal((await fetch(origin + "/api/sessions", { method: "POST", body: "{}" })).status, 403);
    assert.equal((await fetch(origin + "/api/admin")).status, 404);
    const badHostStatus = await new Promise((resolve, reject) => {
      http.get(origin, { headers: { host: "attacker.invalid" } }, response => { response.resume(); resolve(response.statusCode); }).on("error", reject);
    });
    assert.equal(badHostStatus, 403);
    assert.equal(received.length, 1);
    const redirect = await fetch(origin + "/signin-with-chatgpt", { redirect: "manual" });
    assert.equal(redirect.status, 302); assert.equal(redirect.headers.get("location"), "/");
  } finally {
    if (frontend) await close(frontend);
    await close(upstream);
  }
});

test("default local plugin installs no identity bypass and refuses non-loopback API config", () => {
  let installed = false;
  localDemoPlugin().configureServer({ middlewares: { use() { installed = true; } } });
  assert.equal(installed, false);
  assert.throws(() => localDemoPlugin({ enabled: true, baseUrl: "https://public.invalid", token }).configureServer({}), /loopback/);
});
