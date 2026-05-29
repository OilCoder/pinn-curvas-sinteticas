<!-- description: Differentiate human bitácora from Claude's auto memory; enforce what goes where -->

# Memory Policy

The project has **two memory systems** with different purposes, voices, and audiences.
Confusing them dilutes both. This rule defines the boundary.

## Two systems, one project

| Eje | **Bitácora** | **MEMORY.md** |
|---|---|---|
| Owner | The user | Claude |
| Location | `todo/bitacora-YYYY-MM-DD.md` | `~/.claude/projects/<project>/memory/MEMORY.md` |
| Voice | First-person, narrative | Imperative, factual |
| Purpose | User's learning process and project history | Make Claude effective in future sessions |
| Lifetime | Forever (exported to Obsidian for knowledge management) | Per machine, per project |
| Loaded into context | Only via `/bitacora` or manual reference | First 200 lines / 25KB at every session start |

## What goes in the bitácora

Narrative artifacts of the user's process:

- Actions taken in the session (what was implemented, what was changed)
- Design decisions and the alternatives discarded
- Failures and dead ends, with the reason they failed
- Learnings (technical insights, gotchas understood, mental models acquired)
- Pending items (`- [ ]`) for future sessions or for `/plan-writing` to promote

The bitácora is a **journal**. It is in Spanish for prose by default (configurable),
written by the user for the user, and exportable to Obsidian via Cowork. It documents
the **journey**, not the operational state.

**Example bitácora entry:**
```
## 14:32 — Intenté pasar el pipeline a polars

### Cambios técnicos
- src/pipeline/load.py — probé reescribir el load con polars

### Decisiones de diseño
- Mantengo pandas: polars no soporta MultiIndex (depth+well) que necesito
- Alternativa polars con structs requiere reescribir filtrado entero — no vale la pena ahora

### Errores
- Asumí que polars soportaba MultiIndex y empecé a portar el load antes de validar.
  Perdí 1h. Lo descubrí al primer `groupby([depth, well])` que tiró TypeError.
- Pasé el datatype `pl.Categorical` para `well_id` antes de revisar si el filtro
  posterior lo aceptaba — no lo acepta, tuve que revertir.

### Pendientes
- [ ] Evaluar polars en Phase 4 cuando el dataset crezca

### Aprendizajes
- polars usa structs en vez de MultiIndex — distinto modelo mental
- Para series temporales paralelas (pozos) MultiIndex sigue siendo más natural
- Antes de portar a una lib nueva: validar primero los 2-3 patrones más usados,
  no porter completo y descubrir incompatibilidades a mitad
```

**The `Errors` section is not optional venting** — it is structured input for the
user's knowledge management. Each error must include (1) what was wrong,
(2) how it was noticed, (3) what the actual right answer was. That's what makes
it useful when re-read months later in Obsidian.

## What goes in MEMORY.md

Operational facts Claude needs to be effective:

- Build / test / lint / type-check commands
- Project-specific gotchas (ports, env vars, hardware quirks, version pinning)
- Patterns Claude discovered while working (e.g., "the pipeline expects depth+well as MultiIndex; do not flatten")
- Corrections the user made repeatedly (so Claude internalizes them)
- Operational shortcuts (where the data lives, which file is the entry point)

MEMORY.md is a **manual**. Claude writes it for itself. It is in English by default,
imperative, and contains no narrative.

**Example MEMORY.md entry:**
```
## Build
- `make build` requires WSL + GDAL installed at system level
- Tests skip GPU-only ones unless `CUDA_VISIBLE_DEVICES` is set
- Pipeline expects depth+well as MultiIndex; do not flatten

## Gotchas
- lasio crashes on files where ~Other appears before ~Curve — guard at load
- Streamlit cache invalidates on `.streamlit/config.toml` changes only
```

## What does NOT go in MEMORY.md

Hard rule: MEMORY.md must not contain narrative prose, exploration, fallback decisions,
or any "today I learned" framing. Those belong in the bitácora.

If Claude finds itself wanting to write `Hoy intenté X y falló porque Y`, the entry
goes to the bitácora, not to MEMORY.md.

## Maintenance

The bundled `/consolidate-memory` skill (provided by Claude Code) periodically reviews
MEMORY.md, merges duplicates, and prunes stale entries. Run it monthly or whenever
MEMORY.md grows beyond 200 lines.

The bitácora has its own maintenance via Cowork: each `bitacora-YYYY-MM-DD.md` is
exported to the Obsidian vault as `log-YYYY-MM-DD.md` with frontmatter and wikilinks.
The source file in `todo/` stays as the canonical record.

## Cross-references

- See `bitacora/SKILL.md` for the bitácora write procedure.
- See `plan-format.md` for how pending items from bitácora flow into PLAN.md.
- Official docs: <https://code.claude.com/docs/en/memory> for the auto memory system.
