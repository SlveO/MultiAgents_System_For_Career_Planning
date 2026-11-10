import { mkdirSync, existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = fileURLToPath(new URL("../", import.meta.url));
// Allowlist source roots. Runtime state and local secrets are never traversed.
const entries = [".openai/hosting.json", ".env.example", ".gitignore", ".npmrc", "README.md", "app", "components", "db", "drizzle", "examples", "hooks", "lib", "public", "scripts", "tests", "vendor", "package.json", "package-lock.json", "tsconfig.json", "vite.config.ts", "next.config.ts", "postcss.config.mjs", "drizzle.config.ts", "eslint.config.mjs", "cloudflare-env.d.ts", "components.json", "worker"].filter(entry => existsSync(path.join(root, entry)));
const archive = path.join(root, "outputs", "career-compass-sites-source.tar.gz");
mkdirSync(path.dirname(archive), { recursive: true });
const result = spawnSync("tar", ["-czf", archive, "--", ...entries], { cwd: root, stdio: "inherit" });
if (result.error) throw result.error;
if (result.status !== 0) process.exit(result.status || 1);
process.stdout.write(`${archive}\n`);
