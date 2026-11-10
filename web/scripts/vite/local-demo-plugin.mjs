import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { proxyCareerRequest } from "../../lib/career-proxy.mts";

export const LOCAL_DEMO_USER = "career_local_demo";
const localHosts = new Set(["localhost", "127.0.0.1", "[::1]"]);
const localAddresses = new Set(["127.0.0.1", "::1", "::ffff:127.0.0.1"]);

function setHeader(request, name, value) {
  delete request.headers[name];
  for (let i = request.rawHeaders.length - 2; i >= 0; i -= 2) {
    if (request.rawHeaders[i].toLowerCase() === name) request.rawHeaders.splice(i, 2);
  }
  if (value !== undefined) {
    request.headers[name] = value;
    request.rawHeaders.push(name, value);
  }
}

/**
 * Only installed by the explicit local launcher; never active in a build.
 * @param {{ enabled?: boolean, baseUrl?: string, token?: string }} options
 */
export function localDemoPlugin({ enabled = false, baseUrl, token } = {}) {
  return {
    name: "career-local-demo",
    enforce: "pre",
    apply: "serve",
    configureServer(server) {
      if (!enabled) return;
      if (!token || token.length < 32 || !/^http:\/\/127\.0\.0\.1:\d+\/?$/.test(baseUrl || "")) {
        throw new Error("Local demo requires a loopback API and a private server token.");
      }
      server.middlewares.use(async (incoming, outgoing, next) => {
        let url;
        try { url = new URL(incoming.url || "/", `http://${incoming.headers.host}`); }
        catch { outgoing.writeHead(400); outgoing.end(); return; }
        if (!localHosts.has(url.hostname) || !localAddresses.has(incoming.socket.remoteAddress)) {
          outgoing.writeHead(403); outgoing.end("Local demo accepts loopback requests only."); return;
        }
        for (const name of Object.keys(incoming.headers)) {
          if (name.startsWith("oai-authenticated-user-")) setHeader(incoming, name);
        }
        // Reject cross-origin requests to the single-user local workspace.
        if ((incoming.headers.origin && incoming.headers.origin !== url.origin)
            || incoming.headers["sec-fetch-site"] === "cross-site") {
          outgoing.writeHead(403); outgoing.end("Use the local demo page."); return;
        }
        if (["/signin-with-chatgpt", "/signout-with-chatgpt"].includes(url.pathname)) {
          outgoing.writeHead(302, { Location: "/", "Cache-Control": "no-store" }); outgoing.end(); return;
        }
        if (url.pathname === "/api" || url.pathname.startsWith("/api/")) {
          try {
            const headers = new Headers();
            for (const [name, value] of Object.entries(incoming.headers)) {
              if (value !== undefined) headers.set(name, Array.isArray(value) ? value.join(", ") : value);
            }
            const method = incoming.method || "GET";
            const hasBody = !["GET", "HEAD"].includes(method);
            const request = new Request(url, { method, headers,
              ...(hasBody ? { body: Readable.toWeb(incoming), duplex: "half" } : {}),
            });
            const response = await proxyCareerRequest(request, url.pathname.slice(5).split("/"),
              { userId: LOCAL_DEMO_USER }, { baseUrl, token, allowInsecureLoopback: true });
            outgoing.writeHead(response.status, Object.fromEntries(response.headers));
            if (response.body) await pipeline(Readable.fromWeb(response.body), outgoing);
            else outgoing.end();
          } catch {
            if (outgoing.headersSent) outgoing.destroy();
            else { outgoing.writeHead(502, { "Content-Type": "application/json", "Cache-Control": "no-store" }); outgoing.end(JSON.stringify({ error: { code: "service_unavailable", message: "本地后端暂不可用，请检查启动终端。" } })); }
          }
          return;
        }
        setHeader(incoming, "oai-authenticated-user-id", LOCAL_DEMO_USER);
        setHeader(incoming, "oai-authenticated-user-email", "demo@local.invalid");
        setHeader(incoming, "oai-authenticated-user-full-name", encodeURIComponent("本地演示用户"));
        setHeader(incoming, "oai-authenticated-user-full-name-encoding", "percent-encoded-utf-8");
        next();
      });
    },
  };
}
