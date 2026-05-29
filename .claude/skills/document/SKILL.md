---
name: document
description: >
  Generate documentation for a module, class, or function.
  Use when the user says "document this file", "generate docs for X",
  "I need documentation for this module".
argument-hint: "[file or module to document]"
allowed-tools: Read Write Edit Grep Glob
context: fork
agent: Explore
---

# Document

Generate a complete documentation file for a project module.
Follow the standards defined in `docs-style.md` rule.

## Procedure

### 1. Identify the target

- If the user passed `$ARGUMENTS`, use it as the target.
- Read the complete source file to understand what it does.

### 2. Create the document

- Location: **`documentation/`** (per `docs-style.md` — `docs/` is reserved for GitHub Pages)
- Name: `NN_<slug>.md` following `file-naming.md`

### 3. Required sections

```markdown
# <Module name>

## Purpose
One sentence describing the module's role.

## Workflow
Description of the sequence of operations.
Use numbered steps or descriptive text.
Include a Mermaid diagram if the flow is complex.

## Inputs and Outputs
- param_name (type): description
- return (type): description

## Mathematical explanation (if applicable)
Formulas or logic in LaTeX/pseudocode.

## Code reference
Source: <path to source file>
```

## Rules

- Document only what exists in the current code — no plans or assumptions.
- Write clearly and concisely. Short paragraphs, examples when helpful.
- Each document must reflect the actual behavior of the code.
- Avoid TODOs and speculative notes.
- If the module has dependencies, list them.
- **Output goes to `documentation/`, never to `docs/`** (which is reserved for the GitHub Pages landing site).
