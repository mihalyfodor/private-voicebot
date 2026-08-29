Status: done

# 22 — Sidebar & interaction review: navigation and settings in one place

## Overview

The workspace has grown views (Chat, Memory, Wiki; Sources with PRD 20) and the ☰ drawer has grown settings
(listening, reply length, avatar, backdrop, characters, memory/wiki, shutdown). The view tabs live in the main
area, settings in the drawer — two places for "where am I / what can I change". The user wants most configuration
and navigation to live in the sidebar, including the view switcher. This PRD analyses the current interactions
and picks a sidebar pattern that scales without becoming a settings dump.

## Findings (audit of current UI, 2026-08-29)

- Drawer: 6 sections, 12 controls; mixes **navigation** (Memory, Wiki), **session state** (hands-free, reply
  length), **appearance** (avatar, backdrop), **maintenance** (reload characters, shut down). It is an overlay
  (closes on selection) — good for occasional settings, bad for navigation you switch often.
- View tabs top-right of the workspace: the only navigation that is visible without opening anything; three
  tabs today, five with Sources + a possible Settings view.
- Avatar column is 25 % and fixed; the drawer overlays it, so the two never compete for width.
- Keyboard: Esc closes drawer, Space = push-to-talk; no shortcuts for views.

## Pattern research (applicable to a small local app)

| pattern | seen in | fit |
|---|---|---|
| **Icon rail + expandable panel** (activity bar) | VS Code, Discord, Obsidian left ribbon | Rail always visible with view icons (Chat, Memory, Wiki, Sources, Settings); click expands a panel for that item's options. Scales to many items; two levels max. Best fit. |
| Persistent wide sidebar with sections | Slack, Linear, Notion | Always ~240–280 px; nav on top, settings at the bottom. Simple, but permanently steals width from a workspace that already gives 25 % to the avatar. |
| Overlay drawer (current) | mobile apps, Material nav drawer | Fine for settings, poor for frequent view switching. |
| Command palette (⌘K) | VS Code, Linear, Obsidian | Great complement: "switch avatar", "open wiki page…", "retire Anna"; not a replacement for visible navigation. |
| Settings as a view | ChatGPT, Obsidian | Rarely-used config (characters reload, shutdown, windows, vault path) moves out of the sidebar into a Settings page reached from the rail. |

## Proposal

- **Left rail (56 px, always visible) inside the avatar column's left edge**: icons for Chat, Memory, Wiki,
  Sources, and at the bottom Settings + ☰-style "more". Active icon highlighted; hover tooltip; keys `1–4`
  switch views (when not typing).
- **Contextual panel (280 px, toggled by clicking the active rail icon again or via a pin)**: the options for
  the current view — Chat: listening toggle, reply length, avatar, backdrop (what the drawer has today);
  Memory: model view toggle, forget everything; Wiki: re-ingest, lint, open vault; Sources: inbox list.
  Panel pushes the workspace (not an overlay) when pinned; overlays when unpinned; remembers pin state in
  settings.json.
- **Settings view** (in the workspace): characters reload, shutdown, decay windows, vault path, screenshot
  tool — everything touched less than weekly.
- **Remove the top-right tabs** once the rail exists. The avatar column keeps its 25 %, the rail overlaps its
  left edge, so the avatar shifts right by ~56 px only.
- ⌘K palette: out of scope here, listed as PRD 23 candidate.
- Below 700 px: rail becomes a bottom tab bar; panel becomes a sheet.

### Decisions
- Rail + contextual panel over a wide persistent sidebar: navigation stays visible without a permanent
  280 px cost; settings are grouped by the view they affect instead of one long list.
- Panel pushes rather than overlays when pinned: the user wants config visible while working, not a modal.
- Keep the DOM: existing drawer sections become panel fragments keyed by view; no framework.
- Assume ≤ 6 rail items; if false, group with a "more" menu.

## Test scenarios
1. Rail renders one icon per registered view; clicking switches `#workspace` view and highlights; keys 1–4
   switch views only when no input is focused.
2. Clicking the active icon toggles the panel; panel content equals the section list for that view; pin state
   persists via `set_setting` and is restored on reload.
3. Every existing drawer control still sends the same WS action from its new home (hands-free, verbosity,
   avatar, backdrop, reload, shutdown) — assert payloads unchanged.
4. Narrow layout: bottom tab bar visible, rail hidden, panel opens as a sheet.
5. Manual: no control from the old drawer is unreachable; screenshots at 1440 and 900 px widths.

## Out of scope
Command palette (PRD 23 candidate); theming; touch gestures; any change to the avatar rendering.

## Decisions on the prototype's open questions (2026-08-29, defaults unless the user overrides)
1. Rail far left (rail | panel | avatar | workspace) — the avatar column keeps its 25 %, controls stay together.
2. Floating by default, one global pin state persisted in settings.json; pinning is opt-in.
3. 72 px rail with icon + tiny label (self-explanatory for an occasional-use app); collapses to icon-only below 1100 px.
4. Chat gets a panel too (the user wants configuration in the sidebar, not a separate drawer); the ☰ drawer is removed.
5. Settings view holds the rarely-touched items (characters reload, shutdown, vault path, screenshot tool); decay windows
   sit in the Memory panel next to the data they govern.
6. One global open/closed flag; Settings view hides the panel.
Prototype: `docs/prototypes/sidebar/index.html`.
