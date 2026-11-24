#!/usr/bin/env python3
"""Standalone ChessLM baseline trainer.

Example: python train.py --source games.pgn --data-dir run01 --hours 3
The checkpoint is resumable and saved every five minutes.
"""
from __future__ import annotations
import argparse, hashlib, io, json, os, random, time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
import chess, chess.pgn, torch
from torch import nn
from torch.nn import functional as F

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

def atomic(path, writer):
    temporary = path.with_name(path.name + '.tmp')
    writer(temporary)
    os.replace(temporary, path)

def vocabulary():
    moves = set()
    for start in chess.SQUARES:
        for end in chess.SQUARES:
            dx = abs(chess.square_file(start)-chess.square_file(end))
            dy = abs(chess.square_rank(start)-chess.square_rank(end))
            if start == end or not (dx == dy or min(dx,dy) == 0 or sorted((dx,dy)) == [1,2]):
                continue
            move = chess.Move(start,end).uci(); moves.add(move)
            if dx <= 1 and (chess.square_rank(start),chess.square_rank(end)) in ((1,0),(6,7)):
                moves.update(move + piece for piece in 'qrbn')
    return sorted(moves) + ['<start>', '<pad>']

class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__(); self.heads = heads
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.qkv, self.out = nn.Linear(width,width*3), nn.Linear(width,width)
        self.mlp = nn.Sequential(nn.Linear(width,width*4), nn.GELU(), nn.Linear(width*4,width))
    def forward(self, x):
        b,t,d = x.shape
        q,k,v = self.qkv(self.norm1(x)).reshape(b,t,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind()
        x = x + self.out(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(b,t,d))
        return x + self.mlp(self.norm2(x))

class ChessLM(nn.Module):
    def __init__(self, vocab_size, context, width, heads, layers):
        super().__init__(); self.tokens = nn.Embedding(vocab_size,width); self.position = nn.Embedding(context,width)
        self.blocks = nn.Sequential(*(Block(width,heads) for _ in range(layers)), nn.LayerNorm(width))
        nn.init.normal_(self.tokens.weight,std=.02); nn.init.normal_(self.position.weight,std=.02)
    def forward(self, tokens):
        positions = torch.arange(tokens.size(1), device=tokens.device)
        return F.linear(self.blocks(self.tokens(tokens)+self.position(positions)), self.tokens.weight)

def read_games(source, ids, max_games, context, seed):
    def read(stream):
        games, seen = defaultdict(list), set()
        while len(seen) < max_games:
            game = chess.pgn.read_game(stream)
            if game is None: break
            if game.errors or game.headers.get('Variant','Standard') != 'Standard': continue
            sequence = [m.uci() for m in game.mainline_moves()][:context]
            if not sequence: continue
            key = digest([seed,sequence])
            if key in seen: continue
            seen.add(key); bucket = int(key[:8],16) % 100
            split = 'test' if bucket == 0 else 'valid' if bucket == 1 else 'train'
            games[split].append([ids[m] for m in sequence])
        if any(not games[s] for s in ('train','valid','test')): raise ValueError('PGN needs more distinct games.')
        return dict(games)
    if str(source).startswith(('http://','https://')):
        import requests
        response = requests.get(source,stream=True,timeout=(15,120)); response.raise_for_status()
        raw = response.raw
        if str(source).endswith('.zst'):
            import zstandard
            raw = zstandard.ZstdDecompressor().stream_reader(raw)
        return read(io.TextIOWrapper(raw))
    if str(source).endswith('.zst'):
        import zstandard
        with open(source,'rb') as compressed, zstandard.ZstdDecompressor().stream_reader(compressed) as raw:
            return read(io.TextIOWrapper(raw))
    with open(source,encoding='utf-8') as stream: return read(stream)

def padded(rows, pad_id, device):
    tensors = [torch.tensor(row,dtype=torch.long) for row in rows]
    return nn.utils.rnn.pad_sequence(tensors,batch_first=True,padding_value=pad_id).to(device)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True,help='PGN, .pgn.zst, or URL')
    parser.add_argument('--data-dir',type=Path,default=Path('run01'))
    parser.add_argument('--hours',type=float,default=3)
    parser.add_argument('--max-games',type=int,default=100000)
    parser.add_argument('--context',type=int,default=256); parser.add_argument('--width',type=int,default=512)
    parser.add_argument('--heads',type=int,default=8); parser.add_argument('--layers',type=int,default=8)
    parser.add_argument('--batch-size',type=int,default=16); parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',choices=('auto','cpu','cuda'),default='auto'); args = parser.parse_args()
    if args.width % args.heads: parser.error('--width must be divisible by --heads')
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    if device == 'cuda' and not torch.cuda.is_available(): parser.error('CUDA is unavailable')
    root = args.data_dir; root.mkdir(parents=True,exist_ok=True)
    vocab = vocabulary(); ids = {token:i for i,token in enumerate(vocab)}; bos,pad_id = ids['<start>'],ids['<pad>']
    data_signature = digest([args.source,args.max_games,args.seed,args.context,vocab]); games_path = root/'games.pt'
    if games_path.exists():
        artifact = torch.load(games_path,weights_only=False)
        if artifact['signature'] != data_signature: raise ValueError('games.pt settings differ. Choose a new --data-dir.')
        games = artifact['games']
    else:
        games = read_games(args.source,ids,args.max_games,args.context,args.seed)
        atomic(games_path,lambda p: torch.save({'games':games,'signature':data_signature},p))
    print({s:len(g) for s,g in games.items()},'device:',device,flush=True)
    model = ChessLM(len(vocab),args.context,args.width,args.heads,args.layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=3e-4); scaler = torch.amp.GradScaler('cuda',enabled=device=='cuda')
    path = root/'base.pt'; run_signature = digest([data_signature,args.width,args.heads,args.layers,args.batch_size])
    step = tokens = 0; elapsed_before = 0.; rng = random.Random(args.seed)
    if path.exists():
        state = torch.load(path,map_location='cpu',weights_only=False)
        if state['signature'] != run_signature: raise ValueError('base.pt settings differ. Choose a new --data-dir.')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        step,tokens,elapsed_before = state['step'],state['tokens'],state['elapsed']; rng.setstate(state['rng'])
    budget = args.hours*3600; print('parameters:',sum(p.numel() for p in model.parameters()),'starting at',step,flush=True)
    started,last_save = time.monotonic(),elapsed_before
    while elapsed_before + time.monotonic()-started < budget:
        batch = rng.choices(games['train'],k=args.batch_size); x = padded([[bos]+row[:-1] for row in batch],pad_id,device); y = padded(batch,pad_id,device)
        optimizer.zero_grad(set_to_none=True); context = torch.autocast('cuda',dtype=torch.float16) if device=='cuda' else nullcontext()
        with context: loss = F.cross_entropy(model(x).reshape(-1,len(vocab)),y.flatten(),ignore_index=pad_id)
        scaler.scale(loss).backward(); scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(),1.); scaler.step(optimizer); scaler.update()
        if device=='cuda': torch.cuda.synchronize()
        step += 1; tokens += sum(map(len,batch)); elapsed = elapsed_before + time.monotonic()-started
        if elapsed-last_save >= 300 or elapsed >= budget:
            state = {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),'step':step,'tokens':tokens,'elapsed':elapsed,'rng':rng.getstate(),'signature':run_signature}
            atomic(path,lambda p: torch.save(state,p)); print('base',step,round(elapsed),'seconds',tokens,'targets','loss',round(loss.item(),3),flush=True); last_save = elapsed

if __name__ == '__main__': main()
