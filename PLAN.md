# ChessGPT experiment plan

## Research question

Can automated chess feedback improve a transformer trained from scratch on move sequences within a free Colab budget? The first experiment measures two effects separately: learning to play legally from history, and choosing stronger moves when several legal moves are available.

**Hypothesis.** Legality supervision will reduce illegal predictions. Engine preferences will improve move quality beyond legality supervision alone. Playing strength is an empirical outcome. A three-hour run establishes a baseline without assuming a target Elo.

## 1. Prepare games and evaluation positions

- Use standard Lichess games from the normal starting position. Parse the main line, validate it by replay, deduplicate move sequences and split whole games into training, validation and test sets with a fixed seed. Record the source, filters, counts and split seed.
- Encode each half-move as one UCI token, for example `e2e4 e7e5 g1f3`. Include promotion moves, a game-start token and a padding token. Train on move targets only. Recorded games can end by resignation or timeout, so use the chess rules to recognize board-terminal positions during play.
- Use prefixes starting at the initial position. For the initial 256-token context, truncate long games at the context limit and report evaluation coverage. Keep board states, legal lists and engine evaluations available to the trainer and evaluator. The transformer receives only the move history.
- Freeze evaluation positions and rollout starting histories before training. Sample across early, middle and late plies. Reserve enough context for each rollout and exclude training positions from the scored engine-quality set by board-state key.

## 2. Train the autoregressive baseline

Start with a decoder-only transformer with 8 layers, width 512, 8 attention heads, a feed-forward width of 2048 and tied input/output embeddings (roughly 26M parameters). Model size is a tunable choice. The fixed resource is **about three GPU hours of baseline training**, with preprocessing, fine-tuning and evaluation budgeted separately.

Benchmark full training steps on the assigned GPU after warm-up. Record GPU type, peak memory, non-padding move tokens per second and projected token count over three hours. Reduce the microbatch if memory is tight. Use next-move cross-entropy, causal attention, and masks that exclude padding. Save resumable checkpoints outside the temporary runtime, including optimizer state, training progress and data order. Log metrics at 30, 60, 120 and 180 minutes. Use the final checkpoint for every branch.

## 3. Compare three fine-tuning objectives

Budget about 30 GPU minutes per branch, or **4.5 GPU hours total** including baseline training, before evaluation. Prepare engine labels on CPU separately. Use the same sampled training prefixes, starting weights, optimizer settings and effective batch size. Record examples and tokens processed because equal time can yield different sample counts. Compare at matched update counts as a secondary check.

| Branch | Training objective | Comparison |
|---|---|---|
| A | Continue human next-move cross-entropy | Controls for extra training |
| B | A plus a legality loss | B versus A measures the effect of rule feedback |
| C | B plus engine move preferences | C versus B measures the effect of strategic feedback |

**Legality.** Replay each training prefix with a rules library. Sum the model probabilities of all legal moves and minimize the negative log of that sum. Use probabilities from the full unmasked move distribution. Start with unit weight on this loss alongside cross-entropy. Skip terminal positions.

**Engine preferences.** On training positions, take the frozen baseline's top three legal moves. Include an engine-selected candidate so the teacher can surface a tactic the baseline missed. Score candidates with a fixed Stockfish version, equal per-candidate node budgets and scores oriented to the side to move. Begin with a small CPU pilot and record labeling throughput. Keep clear preferences, initially a gap of at least 100 centipawns or a forced mate over a non-mating continuation. Skip ambiguous pairs. Cache labels and record coverage. For C, add a pairwise loss that increases the preferred move's logit relative to the weaker move's logit. Start with unit weight and use validation data for any adjustments.

## 4. Evaluate legality and strength

| Question | Test |
|---|---|
| Does the model track the board? | On held-out histories, measure illegal probability mass and top-choice legality without filtering. Report by ply range. |
| Does it stay legal on its own moves? | From fixed held-out histories, sample both sides at temperature 1 with fixed seeds and no filtering. Count rollouts reaching 40 further plies or ending legally earlier. Record the first illegal ply. |
| Does it retain human move prediction? | Report held-out next-move cross-entropy for the baseline and A, B and C. |
| Does it choose better legal moves? | On the frozen test positions, score each model's highest-ranked legal move against an engine reference at a deeper fixed budget. Report centipawn loss and missed forced mates separately. |
| Does better move choice translate into wins? | Play C versus B from identical opening histories with colors swapped. Mask illegal moves for both, use the same decoding settings, and publish wins, draws, losses and score. Log context-limit cutoffs separately. |

Start with 100 paired openings for head-to-head evaluation. Estimate uncertainty by resampling opening pairs. A small or uncertain score advantage calls for more games before drawing a strength conclusion. Archive split identifiers, model configurations, checkpoints, labeling settings and per-position results so comparisons can be reproduced.

## 5. Decide what to try next

- If B improves raw legality over A while preserving human-move prediction, retain the legality objective.
- If B performs well on recorded histories but fails during rollouts, collect the positions immediately before its first illegal moves. Mix those histories into training and evaluate on fresh rollouts.
- If C improves engine-assessed move quality and beats B across paired openings, repeat with a second training seed before scaling the model.
- If C shows little gain, inspect candidate diversity, preference gaps and label coverage. Scale data or model size after identifying the limiting factor.

The initial fine-tunes use exact rule labels and automated engine preferences. A subsequent self-play RL experiment can test rewards based on game outcomes once the baseline comparisons establish how much each simpler signal contributes.
