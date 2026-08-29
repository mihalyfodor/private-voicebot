# Sidebar navigation prototype (PRD 22)

Clickable mock of the rail + contextual panel proposal in
[docs/22-sidebar-navigation.md](../../22-sidebar-navigation.md). Fake data, no backend, no build step —
one self-contained HTML file. Throwaway: it exists to argue about layout, not to be reused as code.

## How to open it

```
open docs/prototypes/sidebar/index.html
```

Or drop the file on a browser window. Screenshots:

```
.venv/bin/python scripts/screenshot.py /tmp/proto.png \
  --url file://$PWD/docs/prototypes/sidebar/index.html --wait 500
.venv/bin/python scripts/screenshot.py /tmp/proto-narrow.png \
  --url file://$PWD/docs/prototypes/sidebar/index.html --wait 500 --width 640
```

(`--view` does nothing here — the prototype has no `.view-tab` elements, that's the point.)

## What to look at

- **Rail (56 px, far left)** — Chat / Memory / Wiki / Sources on top, Settings at the bottom.
  Hover shows a tooltip, the small digit is the keyboard shortcut, Sources carries an unread badge.
  The old top-right tabs are gone.
- **Clicking the active icon toggles the panel**; clicking a different icon switches view and keeps the
  panel open. `1`–`4` switch views (ignored while an input is focused), `b` toggles the panel,
  `Esc` closes it when floating.
- **Pin toggle** (top right of the panel). *pinned* = the panel is a flex column and pushes the avatar
  and workspace right. *floating* = it overlays with a scrim, the workspace reflows to full width and
  the panel covers the avatar's left edge — the same relationship the drawer has today.
- **Panel contents change per view**: Chat (hands-free switch, reply length, avatar, backdrop — the
  current drawer), Memory (edit/model view, show stale, forget everything), Wiki (re-ingest, lint list,
  open vault), Sources (inbox list + triage). Settings has no panel, so entering it folds the panel away
  and leaving restores it.
- **Settings as a workspace view**: characters reload, decay windows, vault path, screenshot tool,
  shut down. This is the "less than weekly" bucket the PRD wants out of the sidebar.
- **Avatar column** is a dashed placeholder at 25 %, pushed right of the rail — check how much room
  the workspace actually loses when the panel is pinned.
- **Narrow (< 700 px)**: resize the window under 700 px. The rail becomes a bottom tab bar with
  icon + label, the panel becomes a bottom sheet above it, and the avatar shrinks to a 22 vh band on top.

Buttons are wired to a toast so clicks feel real; nothing is persisted.

## Open questions

1. **Rail position** — far left as prototyped (rail | panel | avatar | workspace), or between the avatar
   and the workspace, so the avatar keeps the window edge and the controls sit next to the content
   they affect?
2. **Pinned by default?** Pinned costs a permanent 280 px on top of the avatar's 25 % — at 1440 px the
   workspace drops from ~1080 to ~800 px. Should the default be floating, with pin as opt-in per view,
   or one global pin state as the PRD assumes?
3. **Icon-only rail, or icon + label?** Icon-only needs tooltips and is unlearnable for a five-item rail
   used occasionally; a 72–80 px rail with tiny labels underneath (as the narrow tab bar already has)
   is self-explanatory but wider.
4. **Does Chat need a panel at all?** Its four sections are exactly today's drawer. An alternative:
   Chat's controls stay a transient overlay (the current drawer behaviour) and only Memory/Wiki/Sources
   get pinnable panels.
5. **Settings split** — is the Settings *view* the right home for decay windows and vault path, or
   should those live in the Memory / Wiki panels next to the data they govern, leaving Settings with
   only reload + shutdown + tools?
6. *(minor)* Should the panel remember a per-view open/closed state, or a single global one? The
   prototype uses one global flag plus a special case for Settings.
