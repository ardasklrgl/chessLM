"""Serve a local board for a ChessGPT.ipynb checkpoint."""

import argparse
import json
import re
import shutil
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

import chess
import chess.engine
import torch
from torch import nn
from torch.nn import functional as F


def notebook_vocabulary():
    moves = set()
    for start in chess.SQUARES:
        for end in chess.SQUARES:
            dx = abs(chess.square_file(start) - chess.square_file(end))
            dy = abs(chess.square_rank(start) - chess.square_rank(end))
            if start == end or not (dx == dy or min(dx, dy) == 0 or sorted((dx, dy)) == [1, 2]):
                continue
            move = chess.Move(start, end).uci()
            moves.add(move)
            ranks = chess.square_rank(start), chess.square_rank(end)
            if dx <= 1 and ranks in [(1, 0), (6, 7)]:
                moves.update(move + piece for piece in "qrbn")
    return sorted(moves) + ["<start>", "<pad>"]


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.out = nn.Linear(width, width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width))

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, length, 3, self.heads, width // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind()
        attention = F.scaled_dot_product_attention(query, key, value, is_causal=True)
        x = x + self.out(attention.transpose(1, 2).reshape(batch, length, width))
        return x + self.mlp(self.norm2(x))


class ChessGPT(nn.Module):
    def __init__(self, vocab_size, width, context, heads, layers):
        super().__init__()
        self.tokens = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(context, width)
        self.blocks = nn.Sequential(*(Block(width, heads) for _ in range(layers)), nn.LayerNorm(width))

    def forward(self, tokens):
        positions = torch.arange(tokens.size(1), device=tokens.device)
        x = self.tokens(tokens) + self.position(positions)
        return F.linear(self.blocks(x), self.tokens.weight)


class Player:
    def __init__(self, path, settings_file, heads):
        settings = json.loads(settings_file.read_text()) if settings_file else {}
        self.vocabulary = settings.get("vocabulary", notebook_vocabulary())
        self.ids = {move: index for index, move in enumerate(self.vocabulary)}
        self.heads = heads if heads is not None else settings.get("heads", 8)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        weights = checkpoint.get("model", checkpoint)
        vocab_size, width = weights["tokens.weight"].shape
        self.context = weights["position.weight"].shape[0]
        layers = sum(bool(re.fullmatch(r"blocks\.\d+\.qkv\.weight", key)) for key in weights)
        if vocab_size != len(self.vocabulary) or "<start>" not in self.ids:
            raise ValueError("Vocabulary mismatch. Pass the Colab run's settings JSON with --settings.")
        if self.heads <= 0 or width % self.heads:
            raise ValueError("Invalid attention head count. Pass the Colab run's --settings or --heads.")
        self.model = ChessGPT(vocab_size, width, self.context, self.heads, layers)
        self.model.load_state_dict(weights)
        self.model.eval()
        self.info = dict(name=path.name, step=checkpoint.get("step"), context=self.context)

    @torch.inference_mode()
    def attempt(self, board, history, temperature, mode="free"):
        if len(history) >= self.context:
            raise ValueError("The move history has reached the model's context limit.")
        tokens = torch.tensor([[self.ids["<start>"], *(self.ids[move] for move in history)]])
        logits = self.model(tokens)[0, -1].float()
        legal_moves = {self.ids[move.uci()]: move for move in board.legal_moves if move.uci() in self.ids}
        legal_mask = torch.zeros_like(logits, dtype=torch.bool)
        legal_mask[list(legal_moves)] = True
        scale = temperature or 1.0
        raw_probabilities = ((logits - logits.max()) / scale).softmax(0)
        ranked_logits = logits
        probabilities = raw_probabilities
        candidate_count = len(self.vocabulary)
        if mode == "legal":
            if not legal_moves:
                raise ValueError("No legal moves are available in the model vocabulary.")
            ranked_logits = logits.masked_fill(~legal_mask, -torch.inf)
            probabilities = ((ranked_logits - ranked_logits.max()) / scale).softmax(0)
            candidate_count = len(legal_moves)
        choice = int(torch.multinomial(probabilities, 1)) if temperature else int(ranked_logits.argmax())
        picks = [
            dict(token=self.vocabulary[index], probability=float(probabilities[index]),
                 legal=index in legal_moves, selected=index == choice)
            for index in ranked_logits.topk(min(5, candidate_count)).indices.tolist()
        ]
        prediction = dict(mode=mode, temperature=temperature, picks=picks,
                          illegalMass=float(raw_probabilities[~legal_mask].sum()))
        token = self.vocabulary[choice]
        move = legal_moves.get(choice)
        if move is not None:
            san = board.san(move)
            board.push(move)
            return dict(attempt=token, legal=True, san=san, prediction=prediction, **position(board))
        move = chess.Move.from_uci(token) if re.fullmatch(r"[a-h][1-8][a-h][1-8][qrbn]?", token) else None
        return dict(attempt=token, legal=False, fen=board.fen(), reason=illegal_reason(board, move),
                    prediction=prediction)


def illegal_reason(board, move):
    if move is None:
        return "The model emitted a special token."
    piece = board.piece_at(move.from_square)
    if piece is None:
        return "The starting square is empty."
    if piece.color != board.turn:
        return "That piece belongs to the other side."
    if board.is_pseudo_legal(move):
        return "The move would leave the king in check."
    return "That piece cannot make this move from this position."


def position(board):
    outcome = board.outcome()
    return dict(fen=board.fen(), legalMoves=[move.uci() for move in board.legal_moves],
                outcome=outcome.result() if outcome else None)


def analyse_position(stockfish, board):
    if board.is_game_over():
        return {"lines": []}
    with chess.engine.SimpleEngine.popen_uci(stockfish, timeout=5) as engine:
        evaluations = engine.analyse(board, chess.engine.Limit(time=0.3), multipv=3)
    lines = []
    for evaluation in evaluations:
        line = evaluation["pv"]
        score = evaluation["score"].white()
        lines.append(dict(move=line[0].uci(), san=board.san(line[0]),
                          sequence=board.variation_san(line[:7]), cp=score.score(), mate=score.mate()))
    return {"lines": lines}


DEMO_ASSETS = {
    "/demo": ("demo.html", "text/html"),
    "/demo/": ("demo.html", "text/html"),
    "/demo/style.css": ("demo.css", "text/css"),
    "/demo/app.js": ("demo.js", "text/javascript"),
}


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, player, stockfish=None, **kwargs):
        self.player = player
        self.stockfish = stockfish
        super().__init__(*args, **kwargs)

    def respond(self, status, content, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def respond_json(self, status, value):
        self.respond(status, json.dumps(value).encode(), "application/json")

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/model":
            self.respond_json(200, {**self.player.info, **position(chess.Board()), "stockfish": bool(self.stockfish)})
        elif path == "/":
            self.respond(200, Path(__file__).with_name("index.html").read_bytes(), "text/html; charset=utf-8")
        elif path in DEMO_ASSETS:
            filename, content_type = DEMO_ASSETS[path]
            self.respond(200, Path(__file__).with_name(filename).read_bytes(), f"{content_type}; charset=utf-8")
        else:
            self.respond_json(404, {"error": "Page unavailable."})

    def do_POST(self):
        try:
            if self.headers.get_content_type() != "application/json":
                raise ValueError("Expected a JSON request.")
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= 65536:
                raise ValueError("Invalid request size.")
            body = json.loads(self.rfile.read(length))
            history = body["history"]
            if not isinstance(history, list) or len(history) > self.player.context:
                raise ValueError("Invalid move history.")
            board = chess.Board()
            san = None
            for index, token in enumerate(history):
                if not isinstance(token, str) or token not in self.player.ids or token.startswith("<"):
                    raise ValueError("Invalid move token.")
                move = chess.Move.from_uci(token)
                if not board.is_legal(move):
                    raise ValueError(f"Illegal move in history: {token}.")
                if index == len(history) - 1 and self.path == "/api/position":
                    san = board.san(move)
                board.push(move)
            if self.path == "/api/position":
                token = None
                if "move" in body:
                    if len(history) >= self.player.context or board.is_game_over():
                        raise ValueError("This game has ended.")
                    notation = body["move"]
                    if not isinstance(notation, str):
                        raise ValueError("Enter a move in SAN or UCI notation.")
                    move = board.parse_san(notation)
                    token = move.uci()
                    if token not in self.player.ids or not board.is_legal(move):
                        raise ValueError("That move is unavailable.")
                    san = board.san(move)
                    board.push(move)
                result = dict(san=san, move=token, **position(board))
            elif self.path == "/api/analysis":
                if not self.stockfish:
                    self.respond_json(503, {"error": "Stockfish is unavailable."})
                    return
                result = analyse_position(self.stockfish, board)
            elif self.path == "/api/attempt":
                if board.is_game_over():
                    raise ValueError("The game has ended.")
                temperature = float(body["temperature"])
                if not 0 <= temperature <= 2:
                    raise ValueError("Temperature must be between zero and two.")
                mode = body.get("mode", "free")
                if mode not in ("free", "legal"):
                    raise ValueError("Mode must be free or legal.")
                result = self.player.attempt(board, history, temperature, mode)
            else:
                self.respond_json(404, {"error": "Endpoint unavailable."})
                return
            self.respond_json(200, result)
        except ValueError as error:
            self.respond_json(400, {"error": str(error)})
        except (KeyError, TypeError):
            self.respond_json(400, {"error": "Invalid move history or temperature."})
        except (chess.engine.EngineError, OSError):
            self.respond_json(503, {"error": "Stockfish did not return an evaluation."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Colab .pt file")
    parser.add_argument("--settings", type=Path, help="Optional settings-*.json from the same Colab run")
    parser.add_argument("--heads", type=int, help="Attention heads for custom runs without a settings file")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--stockfish", help="Path to Stockfish for optional analysis")
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        player = Player(args.checkpoint, args.settings, args.heads)
    except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
        parser.exit(1, f"Could not load {args.checkpoint.name}: {error}\n")
    print(f"Loaded {args.checkpoint.name}. Open http://127.0.0.1:{args.port}", flush=True)
    stockfish = args.stockfish or shutil.which("stockfish")
    with ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, player=player, stockfish=stockfish)) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
