# ChessLM

ChessLM is an autoregressive transformer trained from chess move histories. It includes the Colab notebooks, a standalone resumable trainer, and a local playable web demo.

## Train

Install dependencies with `pip install torch python-chess requests zstandard`, then run:

```bash
python train.py --source /path/to/games.pgn --data-dir run01 --hours 3
```

The default model is approximately 26M parameters. Checkpoints are saved to `run01/base.pt` every five minutes and resume automatically. Use a Lichess PGN or a `.pgn.zst` archive. A CUDA GPU is recommended.

## Play

Install `python-chess` and PyTorch, then start the demo with a checkpoint:

```bash
python play/serve.py /path/to/base.pt --port 8768
```

Open `http://127.0.0.1:8768/demo/`. The interface supports free generation, legal-only filtering, top-pick inspection, move replay, and PGN/FEN sharing.

The notebooks contain the full experiment including automated legality feedback, Stockfish preference labels, fine-tuning branches A/B/C, and evaluation.
