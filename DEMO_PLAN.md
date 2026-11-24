# ChessLM web demo plan

Build a playable chess workspace with the layout from the supplied chess mockup, Glaze for the main interface, and Sodium for the AI inspector and terminal.

## Reference decisions

| Source | What carries into the demo |
| --- | --- |
| `System-Preview (1).html` · Glaze | Dark neutral shell, raised controls, thin inset borders, rounded surfaces, and a reserved glass treatment for the main action. Use the bundled dark tokens as the starting point. The preview uses backgrounds such as `#0c0c0c`, `#161616`, and `#1f1f1f`, with 26–36 px controls. |
| `System-Preview (1).html` · Sodium | Rectangular controls, flat dark wells, hairline rules, compact monospaced labels and numeric columns. Keep Sodium's yellow accent local to console focus, selected rows and meaningful highlights. |
| `chess.svg` and `chess.png` | The same 1440 × 1120 IDE-style composition. Keep the top toolbar, left AI inspector, centered grayscale board, right move list, bottom terminal and thin status bar. Use the SVG board fills `#BFC0C5` and `#7D7E86` and the preview's restrained highlight treatments. |

Use the SVG and PNG as layout references. Render the squares, pieces, move list and console as interactive elements. The board and analysis areas in the mockup contain different example positions, so the demo will derive both from a single game history. Start with Glaze's `New game` treatment and leave the slow accent animation out of the first pass. System font fallbacks keep the local demo usable when preview fonts are unavailable.

## Workspace layout

| Area | Content and interaction |
| --- | --- |
| Glaze toolbar | ChessLM, New game, analysis toggle, and visible-state buttons for AI, Terminal and Moves. Use one glass primary action and quiet secondary controls. |
| Sodium AI inspector, left | Checkpoint and temperature, Only legal toggle, model top picks, raw illegal probability, and a Stockfish comparison when connected. Use shared move labels when comparing ranks. |
| Glaze board stage, center | Square board, side-to-move and player labels, move hints and last-move markers. Keep the actual board centered when panels change size. Reserve a slim evaluation rail for positions with engine data. |
| Glaze move panel, right | Compact numbered SAN rows and first, previous, next and latest navigation. Replay shows a historical position and pauses input until the user returns to the latest position. |
| Sodium terminal, bottom | Terminal and Model output tabs. Show timestamp-free commands and readable event lines for moves, invalid attempts, retries and service status. A focused prompt accepts SAN or UCI through the board's move handler, plus `/help`, `/new`, `/undo`, and `/retry`. |
| Status strip | Model connection and analysis state. Show session persistence only if implemented. |

Use the reference's approximately 300 px left pane and 330 px right pane at 1440 px as desktop anchors. Let the center board fit the remaining width and height. At narrow widths, keep the board first, move inspector and history into switchable panes, and collapse the terminal until requested. Keep labels readable and preserve keyboard focus.

## Implementation order

1. **Lay out the new demo at a separate route.** Reuse the existing local server and checkpoint loader. Add scoped Glaze and Sodium styles so typography, radius, borders and accents stay consistent in their regions. Verify the desktop composition at the reference size and then at a narrow viewport.
2. **Wire one live game.** Reuse the existing position and model attempt endpoints for both board and terminal actions. Keep promotions, undo and replay, raw and legal-only inference, temperature, the top-picks list, and illegal-attempt highlighting. A failed raw attempt leaves the board and history at the last legal position. The same position can be retried in either mode.
3. **Connect analysis when Stockfish is available.** Show model probabilities and Stockfish evaluations for the same input position, with a consistent evaluation perspective. The model's last-turn predictions belong to the position immediately before its move. An analysis toggle controls engine requests. If the engine is unavailable, show that state and keep the model playable. Display scores as evaluations. A probability comparison requires an explicit score-to-distribution conversion and per-move engine coverage.
4. **Validate on the real checkpoint.** Play both colors. Exercise raw invalid attempts, switching to legal-only and retrying, promotions, move replay, and terminal entry. Check that closing and reopening panes preserves the game. Verify that the inspector, terminal and move list refer to the same position and that a delayed engine response cannot overwrite a newer position's analysis.

The first demo is a local playable workspace with the existing model. Training and Stockfish distribution matching are separate from the interface work.
