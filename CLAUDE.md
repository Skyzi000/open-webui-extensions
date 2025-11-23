# Claude Development Guidelines

## Repository Structure

This repository contains Open WebUI extensions, with some components managed as Git submodules.

### Submodules

- `graphiti/` - Graphiti memory system extensions (separate repository)
- `open-webui/` - Reference submodule

## Commit Message Rules

### Important: Check Submodule Commit Style First

Before committing to a submodule, **always check the existing commit history** using:

```bash
git log -5 --format=medium
```

Each submodule may have its own commit message style. Follow the existing pattern.

### Graphiti Submodule Style

Uses **Conventional Commits** format:

```text
type(scope): short description

Optional detailed description explaining what and why.
- Bullet points for multiple changes
- Keep each point concise
```

**Types:**

- `feat` - New feature
- `fix` - Bug fix
- `chore` - Maintenance (version updates, dependencies, etc.)
- `docs` - Documentation only
- `refactor` - Code refactoring without feature changes
- `test` - Adding or updating tests

**Scopes:**

- `filter` - Filter components
- `tools` - Tools components
- `pipeline` - Pipeline components
- `action` - Action components

**Examples:**

```text
feat(tools): add get_recent_episodes method for chronological episode retrieval

- Add get_recent_episodes() method with limit/offset pagination
- Sort episodes by created_at in descending order (newest first)
- Optimize database queries to fetch only offset+limit episodes
- Update version to 0.3
```

```text
fix(filter): Fix null pointer exception in D3.js graph tick handler

Add null checks before calling getBBox() on text nodes in the force
simulation tick event.
```

### Root Repository Style

Uses Japanese, simple descriptive messages:

```text
サブモジュールを更新
フルコンテキストモードトグルフィルターを追加
```

## Development Guidelines

### General Principles

1. **Version Management**
   - Update version numbers when adding features or fixing bugs
   - Follow semantic versioning principles

2. **Before Committing**
   - Verify changes work correctly
   - Check for linting errors
   - Review the commit message style of the target repository/submodule

### OpenWebUI Tools Development

1. **Error Handling**
   - Do not implement fallbacks without clear necessity
   - Return clear, actionable error messages
   - Provide specific information for AI to understand and address issues

2. **Documentation for AI Users**
   - Target audience is AI, running in Docker on OpenWebUI
   - AI only sees Tools class method docstrings and return values
   - Include all critical information in Tools class docstrings
   - Use Field descriptions to clarify data format requirements
