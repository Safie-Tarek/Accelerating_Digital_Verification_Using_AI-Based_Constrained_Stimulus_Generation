# Team Collaboration Guide

Welcome! This guide helps us work together smoothly, keep our work organized, and make collaboration easier across the AI and Verification subteams.

These are shared guidelines to help us maintain a clean repository, clear documentation, and reliable progress. When in doubt, communicate early and keep the relevant work documented.

## 1. Jira — Task Management

**Every task must be tracked in Jira.** If work needs to be done, it needs a Jira card.

Each task should be a small, meaningful piece of work and include:

- **Title:** A short, descriptive name.
- **Description:** Explain what needs to be done and why.
- **To-do list:** Break the task into steps when useful (Not necessary).
- **Acceptance criteria:** Clearly define what must be achieved for the task to be considered complete.
- **Assignee:** Every task must have an owner.

The description and acceptance criteria must be sufficient for the assignee to start working without needing to guess the expected outcome.

Documentation is expected for every task unless it is a bug fix. But document if the fix introduced something worth keeping.

Jira issue types such as Epics, Stories, Tasks, and Subtasks are available as a reference for organizing work, not as mandatory classifications.

## 2. Git and GitHub — Version Control

### Branching

- Both `main` and `dev` are protected. Do not push directly to either branch.
- Create all working branches from the latest `dev`.
- Use the following naming convention:

  `domain/feature-or-topic`

  Examples:
  - `ai/implement-adapter`
  - `verification/regression-script`
  - `ai/fix-agent-response`

- Delete feature branches after their pull requests have been merged.
- Always pull the latest changes before pushing your work.

### Commits

Commits should be meaningful, focused, and frequent.

- Avoid bulk commits that combine unrelated changes.
- Commit frequently, but don't split a single coherent feature into unnecessary commits.
- Use the standardized commit format below.
- Document the reasoning behind changes in the commit message or the relevant file under `docs/`.
- Not every commit needs a Jira ID.

**Commit message format:**

`type(scope): description [JIRA-ID]`

The Jira ID is optional. When additional explanation is needed, include a body explaining the reasoning.

Example:

```text
feat(verification): add AXI burst coverage [PROJ-123]

Improve visibility into burst-length coverage and help
identify scenarios that remain uncovered.
```

Use these standard types:

- `feat` — New functionality
- `fix` — Bug fixes
- `docs` — Documentation changes
- `test` — Tests and verification
- `refactor` — Code restructuring without intended behavior changes
- `build` — Build system or dependency changes
- `chore` — Maintenance and housekeeping

### Pull Requests

**Every contribution must go through a pull request targeting `dev`. Never open a pull request directly to `main`.**

Before opening a PR:

- Ensure the branch is based on the latest `dev`, resolving conflicts when necessary.
- Run the relevant tests and confirm they pass.
- Update the documentation when applicable.
- Link the related Jira task and repository changes.

Each PR should have:

- A clear title and description.
- A summary of the changes and their purpose.
- Testing performed and its results.
- A link to the relevant Jira task.
- At least one approval from another team member before merging.

The PR author may merge their own PR once it has been approved and all required checks have passed.

**Merge strategy: Merge commit.** This preserves the individual commits made by each contributor while recording the integration of the branch into `dev`. Keep individual commits meaningful so the history remains useful.

### Main Branch and Milestones

- `main` is updated only through a pull request from `dev`, never through direct pushes.
- The team decides when a milestone is ready to be merged into `main`.
- Create a Git tag for each milestone to mark the corresponding version.
- Update the README whenever the milestone introduces changes that affect setup, usage, features, or project status.

## 3. Repository and Documentation

### File Types

Markdown, text, JSON, code, and the configuration formats required by the project are allowed.

Please avoid committing Word, PDF, or PowerPoint documents. Upload these files to the team's shared Drive instead.

Also, keep generated files, logs, caches, virtual environments, local configuration, and secrets out of Git. Use `.gitignore` where appropriate.

**Never commit API keys, access tokens, passwords, credentials, or other secrets to GitHub.**

### Documentation Structure

Keep documentation organized by responsibility:

- `docs/ai/` — AI-related documentation.
- `docs/verification/` — Verification-related documentation.
- `docs/` — Shared project documentation, architecture, and cross-subteam decisions.

Use lowercase kebab-case for documentation filenames, such as `verification-plan.md`.

Update documentation when changes affect setup, behavior, architecture, usage, or other information teammates need to understand or reproduce your work. Bug fixes don't automatically require new documentation unless the fix changes documented behavior or reveals something worth recording.

Link relevant Drive documents from Jira cards or Markdown documentation so teammates can find them easily.

## 4. Python Environment

Whenever you start Python work, and whenever dependencies change, run:

```bash
uv sync --all-extras
```

- Use the Python version declared by the repository.
- Manage dependencies through `uv`.
- Commit the updated `uv.lock` whenever dependency changes require it.
- Don't silently change dependency versions or bypass the project's declared environment.

This helps everyone work with a consistent, reproducible setup.

## 5. Communication and Handoffs

### Raise Blockers Early

If you're stuck, encounter an unexpected issue, or need input from another teammate, communicate early through the team's usual communication channel.

Link the relevant Jira task and briefly explain what is blocking you. Asking for help early is better than silently losing time.

### Make Handoffs Easy

Before handing work over to another teammate, update the Jira card with:

- What has been completed.
- What remains to be done.
- Known issues or blockers.
- Relevant code, pull requests, and Drive documents.

The goal is to let the next person continue without having to reconstruct your progress from scratch.

### Cross-Subteam Integration

AI and Verification tasks may be developed independently, but their work must integrate successfully where required.

Communicate interface changes, dependencies, and integration issues early. Document shared decisions in `docs/` so both subteams can refer to the same information.

## Final Reminder

Keep tasks visible in Jira, commits meaningful, pull requests reviewable, and documentation up to date. Communicate early when something is unclear or blocked.

The purpose of these guidelines is to make our collaboration smoother and help everyone build on each other's work with confidence.
