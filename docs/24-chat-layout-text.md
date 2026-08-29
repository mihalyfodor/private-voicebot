Status: refactoring

# 24 — Chat layout around Wanko + text input

## Overview
With one avatar (PRD 23), the shell is rebuilt around a **companion column** (avatar, conversation history, input)
that is half the screen in Chat and a ~25 % vertical slice elsewhere. Typed input joins voice as a first-class
way to talk; push-to-talk is retired in favour of hands-free + a mute toggle.

## User stories
- As a user, in Chat I see Wanko large on the left and the conversation on the right, with a text box I can type into.
- As a user, on Memory/Wiki I still see Wanko and the running conversation in a narrow column and can keep talking
  or typing while editing.
- As a user, typed and spoken messages behave identically (memory, tools, wiki); replies are spoken unless muted.
- As the developer, there is one history element and one input element, re-parented between layouts.

## UI/UX
- `#companion` = `#avatar` (square canvas, backdrop behind) + `#history` + `#composer` (text input, send, mute toggle,
  listening indicator). Chat view: `#companion` spans the full workspace as a 2-column grid (avatar | history+composer).
  Other views: `#companion` is the left column (25 %, min 280 px), stacked vertically (avatar / history / composer),
  content on the right. Below 700 px: avatar band on top, content below, composer docked at the bottom.
- Composer: single-line input growing to 4 lines, Enter sends, Shift+Enter newline; mic/mute button with state
  (listening / muted / speaking); Space toggles mute when not typing. The Speak/PTT button and `ptt` action are removed.
- History shows both roles as today; typed turns are marked with a small keyboard glyph.
- Backdrop switcher on the avatar itself (Chat view only): a row of small thumbnail icons (one per backdrop, ~28 px,
  current one highlighted) in the bottom-left corner of the avatar area, fading in on hover; sends the same
  `set_backdrop` payload as the panel chips (which stay). Hidden in the narrow layout.

## Technical approach
- New WS action `text_turn {text}` → same path as a final transcript (`submit_turn`), origin "typed"; reply TTS is
  skipped when `muted` (client-side: still receive `speech` but don't play — or server flag `set_muted`; pick the
  simpler: client-side skip, so barge-in logic is untouched).
- `fit()` handles the two container shapes (square-ish vs tall) — it already scales by min(width, height).
- Layout switch = moving `#companion` between `#view-chat` and the shell's left slot on `setView`; the Live2D
  canvas survives re-parenting (PIXI `resizeTo` parent + `fit()` after move).
- Remove: `#btn` PTT, hands-free toggle becomes "voice on/off" in the composer; keep `set_hands_free` action
  semantics (voice on = hands-free listening).

### Decisions
- One companion element re-parented, not two copies: single source of history and input state.
- Client-side mute over a server flag: no change to turn/interrupt logic.
- Text turns reuse `submit_turn`: zero divergence between typed and spoken.

## Test scenarios
1. `text_turn` produces the same controller events/log line as a spoken turn (`session.log` shows `typed:`), and
   memory save includes it.
2. Muted client ignores `speech` messages (unit via fixture render) and still renders the transcript.
3. `setView('memory')` moves `#companion` to the slice and calls `fit()`; back to chat restores the grid (DOM assertions).
4. Backdrop thumbnails send `set_backdrop` identical to the panel chips and reflect the current key; payload parity for remaining controls; `ptt` action absent; Space toggles mute only when no input focused.
5. Manual: type a question → answer spoken; mute → next answer silent; talk while on Wiki tab → history updates in the slice.

## Out of scope
Streaming token display, markdown rendering in history, multiple conversations.
