# AI transcripts — how I directed the AI

I used AI two ways, as the brief expects: to **understand** an unfamiliar problem (cadastral
georeferencing drift in Indian land records) and to **build** the solution (the alignment +
calibration method in `solver/`).

## Coding session (Claude Code)

The method was built in a Claude Code (CLI) session, directed in phases:
understand the brief and data → design the approach → build the alignment engine →
add self-supervised confidence calibration → wire the end-to-end pipeline → debug the real
failures it surfaced (over-flagging, a saturated confidence, dense-village neighbour-snapping)
and fix each with a principled change (adaptive search, IoU regression, scale-aware shift cap).

**Export:** run `/export` in the Claude Code session and commit the exported file here, e.g.
`transcripts/claude-code-session.txt`.

## Web chats (problem understanding)

List any web-chat share links here (ChatGPT "Share" / Claude.ai "Share"):

- _(add share links, or remove this section if all direction was in the coding session)_

## What to read for "how I direct AI"

The interesting signal is the **debugging arc**: the first full run flagged 74% of plots; a later
change collapsed Malatavadi's accuracy to ~0.03 by snapping small plots onto neighbours. Each was
diagnosed from the data and fixed with a targeted, generalising change rather than a hand-tuned
constant — that reasoning is what the transcripts capture.
