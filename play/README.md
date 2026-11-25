# Play against your ChessGPT checkpoint

In Google Drive, open `My Drive/chessgpt/run01` (or the folder you set as ROOT in Colab). Download `base.pt` for the baseline, or `A.pt`, `B.pt`, or `C.pt` for a fine-tune. An intermediate checkpoint such as `base-60min.pt` works too. Download the matching `settings-*.json` if you changed the attention head count or vocabulary.

From the project root, with PyTorch and python-chess installed:

```sh
python chessgpt/play/serve.py ~/Downloads/base.pt
```

For a new Python environment, install the app dependencies with `python -m pip install -r chessgpt/play/requirements.txt`. To load a custom run, add `--settings /path/to/settings.json` to the launch command. Use `--port 8766` if the default port is busy.

Open [the local board](http://127.0.0.1:8765). Choose White or Black, then click a piece and its destination.

- **Free generation** samples from the whole model vocabulary. Illegal moves highlight their source and destination in red without advancing the game. **Retry AI move** samples again from the same history. **Revert attempt** clears the red markers first.
- **Legal moves only** filters the entire vocabulary against the current board and rescales the probabilities of the remaining moves. Switch modes and click **Retry AI move** to continue from an invalid attempt.

The last AI turn shows up to five top picks in the mode it used, alongside the raw probability of an illegal move before any filtering. Temperature zero picks the highest-scoring allowed candidate. The displayed percentages at that setting use the model's scores at temperature one.

The app performs inference on your computer. It accepts checkpoint files from the notebook as saved, including partially trained runs. It treats the complete sequence of legal moves from the initial position as the model input. Play stops at the checkpoint's context limit.

## ChessLM workspace

Start the same server and open [the demo](http://127.0.0.1:8765/demo/). If you launched it with another port, use that port in the link. Toggle AI, Terminal, and Moves in the toolbar. The terminal accepts SAN such as `Nf3` or UCI such as `g1f3`, along with `/help`, `/new`, `/undo`, and `/retry`.

For optional live evaluation, have `stockfish` on your PATH or pass `--stockfish /path/to/stockfish` when launching. The Analysis button then enables candidate evaluations from White's perspective. Model picks and the adjacent Stockfish lines refer to the position before the AI's move. The evaluation bar follows the board you're currently viewing.
