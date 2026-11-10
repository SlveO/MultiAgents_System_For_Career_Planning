# Repository Guide

## Current objective and entry points

The Demo functionality validation is complete per the user's latest instruction. The active handoff is preparing the verified workspace for GitHub and handing the web app to ChatGPT Sites on the user's local desktop. Read [README.md](README.md) for operation, [DEMO_PLAN.md](DEMO_PLAN.md) for Demo scope, [AGENT_WORKLOG.md](AGENT_WORKLOG.md) for actual status, [WEB_CONTRACT.md](WEB_CONTRACT.md) for the Sites/API contract, and [SITES_HANDOFF.md](SITES_HANDOFF.md) for the GitHub-to-Sites steps.

The prior cleanup of obsolete runtime materials and aliases is complete and is not being repeated. This handoff only synchronizes the already verified workspace; do not infer new functional changes from the old cleanup records.

## Scope and collaboration

Keep the Demo's main flow working: material → follow-up → plan → feedback → graph/source → export → restore. The main agent owns scheduling, shared interfaces, integration, and the worklog. The main agent may schedule up to 3 subagents at once; assign disjoint files and never let agents write the same file concurrently.

Use `gpt-6-luna max` for broad reading or scanning and simple document/config edits. Use `gpt-6-sol xhigh` only for necessary cross-module reasoning, complex fixes, or difficult integration.

## Runtime constraints

On the server, reuse the existing GPU 1 environment and `LOCAL_MODEL_DEVICE=cuda:1`; do not download model weights, replace the validated environment, or move local model/runtime profiles to the desktop. The GitHub handoff does not authorize GPU/API/model calls. The user has confirmed Demo validation is complete, so do not repeat the functional trial as part of this handoff.

For a clean desktop clone, the web app may install dependencies from its committed lockfile with `npm run install:ci` and build with `npm run build`, using Node.js 22.13+ and npm. This allowance applies to the frontend clone only; it does not authorize changing the server GPU environment. The web build needs no Python, CUDA, model weights, or API keys. Do not copy the server's `.env` to the desktop or repository.

## Checks and publication

For this explicit delivery, the user authorizes committing and pushing the current branch `codex/proposal-expansion-20261105` to the existing `origin` GitHub repository. Do not force-push, merge into or push `main`, change repository visibility, or invent a commit hash; report the result actually returned by Git. A GitHub push prepares source only. Create and publish the Site later through the local desktop Sites workflow; the server does not publish it directly.

The document/configuration agent does not run tests, builds, or installs. The main agent owns any assigned server source preparation/build and the authorized Git operations. Do not repeat the Demo trial or make model/API calls for this handoff. Later Git actions must follow the user's authorization and current task scope.
