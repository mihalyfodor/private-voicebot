Status: implementation

# 22b — Rail switches views only; settings live in the Settings view

Follow-up to [22-sidebar-navigation.md](22-sidebar-navigation.md), user decision 2026-08-29: no contextual panel.

## Changes
- Remove `#panel-wrap`/`#side-panel`, `#scrim`, pin button, key `b`, `set_setting panel_pinned` (action, whitelist,
  `/api/config` field, tests) and the panel CSS (incl. narrow-screen sheet). Rail click = `setView` only; clicking the
  active item does nothing. Keys 1–5 switch views (5 = Settings).
- Settings view sections (existing `.set-row` style): **Listening** (hands-free switch), **Reply length** (slider),
  **Backdrop** (chips), **Memory** (decay windows read-only; danger zone: Forget everything with confirm),
  **Wiki** (vault path, Open vault), **Characters** (Reload), **Tools** (screenshot helper), **Session** (Shut down).
- View-local actions: Memory header keeps "model view" and gains "Save now" (moved from the panel); Wiki footer
  regains "Re-ingest last session" + lint findings; Sources view keeps "file a source…".
- Every control keeps its WS payload (parity test from PRD 22 extended to the new homes).

## Test scenarios
1. No element with ids `side-panel`, `pin-btn`, `scrim` exists; `set_setting` action rejects `panel_pinned` (unknown key).
2. Every moved control sends the same payload as before (hands-free, verbosity, backdrop, memory_reset, wiki_ingest,
   reload_characters, shutdown, save_memory).
3. Keys 1–5 switch views; ignored while typing; Esc only closes the intake popup.
4. Screenshots at 1440 and 640: Settings shows all sections without overlap; Memory header shows model view + Save now;
   Wiki footer shows re-ingest + lint.

## Out of scope
Any change to the Chat layout (PRD 24).
