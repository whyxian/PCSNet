# How to write an AGENTS.md file (with example)

There's no schema to memorize. The official guidance is to add sections that help an agent work effectively; 
popular choices include a project overview, build and test commands, code style guidelines, testing instructions, and security considerations.
Here's a compact example in the spirit of the official samples:

```markdown
# AGENTS.md

## Project overview
Node.js monorepo for the Acme billing platform. TypeScript everywhere.
Services live in /services, shared libraries in /packages.

## Setup commands
- Install deps: `pnpm install`
- Start dev server: `pnpm dev`
- Run all tests: `pnpm test`

## Code style
- TypeScript strict mode
- Single quotes, no semicolons
- Prefer functional patterns; avoid classes for new code

## Testing instructions
- CI config lives in .github/workflows
- Run `pnpm vitest run -t "<test name>"` to focus a single test
- The commit should pass lint and all tests before merge

## Security considerations
- Never commit .env files or anything in /secrets
- Payment code in /services/billing requires a human review tag

## PR instructions
- Title format: [<service>] <description>
- Always run `pnpm lint` and `pnpm test` before committing
```

A few rules of thumb that separate files agents follow from files agents ignore:
- Write commands, not prose. "Run pnpm test" beats "we have a comprehensive testing strategy." Agents act on the former.
- Tell it what you'd tell a new teammate. Commit message format, PR conventions, deploy quirks, and the directory nobody should touch. The spec is written in exactly this spirit.
- Keep it tight. Style minutiae belong in your linter, not your context file. 
Models follow focused instruction files more reliably than exhaustive ones, a point we documented in the CLAUDE.md guide, so spend that budget on architecture and commands rather than semicolon rules.

**For monorepos, nest the files.** Agents read the nearest AGENTS.md in the directory tree, so the closest file takes precedence, and every package can carry its own instructions. 
This scales further than you'd think. Large monorepos carry many nested AGENTS.md files, one per package. 
The main OpenAI repo had 88 of them when the standard's own site was written.

And when instructions collide? The spec's precedence rules are short and sensible. 
The AGENTS.md file closest to the edited file wins, and explicit user chat prompts override everything. 
Your file sets the defaults; the human in the loop keeps the steering wheel.

# AGENTS.md vs CLAUDE.md (and other tool-specific files)
This is the question we hear most, and the answer is less either-or than people expect.

CLAUDE.md is Claude Code's native context file, a Markdown file in your project root whose contents are automatically included at the start of every Claude Code conversation. It plays the same role AGENTS.md plays, but for one tool, and per-tool files like it are exactly the fragmentation the open standard set out to end. Notably, the official AGENTS.md compatibility list spans most major agents, but Claude Code isn't on it; Anthropic's tooling reads CLAUDE.md.

In practice, the most straightforward setup for most teams is:

| 文件          | 读取者                                                                               | 用途                                   |
|:------------|:----------------------------------------------------------------------------------|:-------------------------------------|
| `AGENTS.md` | Codex, Cursor, Jules, Gemini CLI, Aider, Copilot coding agent, Devin, 以及 20+ 其他代理 | 您的规范化、与工具无关的项目指令                     |
| `CLAUDE.md` | Claude Code                                                                       | 与 `AGENTS.md` 内容相同，加上任何 Claude 特定的指导 |

Keep the substance identical between them so no agent works from stale instructions, and let each file carry only the tool-specific extras. Some tools also make adoption a config switch rather than a convention. Aider reads AGENTS.md with one line in `.aider.conf.yml`, and Gemini CLI accepts it via a context file-name setting. The official site even documents a one-line symlink pattern for migrating older file names without breaking anything.

If you only maintain one file today, make it the open one, and mirror it for the tools that need their own.
