const $ = id => document.getElementById(id);
const files = "abcdefgh";
const referenceOpening = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6"];
const params = new URLSearchParams(location.search);
const embedded = params.has("position");
const pieces = {
  k: ["♚", "king"], q: ["♛", "queen"], r: ["♜", "rook"],
  b: ["♝", "bishop"], n: ["♞", "knight"], p: ["♟", "pawn"],
};
const pieceUrls = JSON.parse($("piece-art").textContent);
const motionPreference = matchMedia("(prefers-reduced-motion: reduce)");
const artImages = Object.fromEntries(Object.entries(pieceUrls).map(([name, src]) => {
  const img = new Image();
  img.src = src;
  return [name, img];
}));
for (const icon of document.querySelectorAll("[data-piece-art]")) icon.src = pieceUrls[icon.dataset.pieceArt];
const percent = new Intl.NumberFormat(undefined, {style: "percent", maximumFractionDigits: 1});
const modelOutput = [];
const terminalLog = [];
const engineCache = new Map();
const enginePending = new Set();

let model;
let frames = [];
let moves = [];
let cursor = 0;
let renderedPly = 0;
let selected = "";
let promotion = "";
let busy = false;
let rejected = null;
let lastPrediction = null;
let analysis = false;
let online = false;
let outputTab = false;
let notice = "";

async function request(endpoint, data) {
  const settings = data ? {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(data),
  } : undefined;
  const response = await fetch("/api/" + endpoint, settings);
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "The server returned an error.");
  return result;
}

function element(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function log(text, type = "event") {
  terminalLog.push({text, type});
  renderConsole();
}

function renderConsole() {
  const entries = outputTab ? modelOutput : terminalLog;
  const rows = entries.map(entry => element("li", entry.text, entry.type));
  if (!rows.length) rows.push(element("li", "Model output appears after its first turn."));
  $("console-lines").replaceChildren(...rows);
  $("console-output").scrollTop = $("console-output").scrollHeight;
}

function history(ply = moves.length) {
  return moves.slice(0, ply).map(move => move.uci);
}

function current() { return frames[cursor]; }
function live() { return cursor === moves.length; }
function playerSide() { return $("side").value; }
function turn(position = current()) { return position.fen.split(" ")[1]; }
function canPlay() {
  const position = current();
  return position && online && !busy && live() && !position.outcome &&
    cursor < model.context && turn() === playerSide();
}

function setPane(id, open, restoreFocus = false) {
  $(id).hidden = !open;
  $("workspace").dataset[id] = open;
  const toggle = document.querySelector('[data-pane="' + id + '"]');
  toggle.setAttribute("aria-pressed", String(open));
  if (restoreFocus) toggle.focus();
}

for (const button of document.querySelectorAll("[data-pane]")) {
  button.addEventListener("click", () => setPane(button.dataset.pane, $(button.dataset.pane).hidden));
}
for (const button of document.querySelectorAll("[data-close]")) {
  button.addEventListener("click", () => setPane(button.dataset.close, false, true));
}

const workspace = $("workspace");
const minimumPaneWidth = 175;
const collapseWidth = 145;

for (const handle of document.querySelectorAll("[data-resize]")) {
  const id = handle.dataset.resize;
  const pane = $(id);
  const direction = id === "ai" ? 1 : -1;
  const widthProperty = "--" + id + "-width";
  const maxWidth = () => Math.floor((workspace.clientWidth - 10) * .33);
  const paneWidth = () => Math.round(pane.getBoundingClientRect().width);
  let drag;

  function resize(width) {
    workspace.style.setProperty(widthProperty, Math.max(minimumPaneWidth, Math.min(maxWidth(), width)) + "px");
  }

  new ResizeObserver(() => {
    if (pane.hidden) return;
    handle.setAttribute("aria-valuemin", minimumPaneWidth);
    handle.setAttribute("aria-valuemax", maxWidth());
    handle.setAttribute("aria-valuenow", paneWidth());
  }).observe(pane);

  handle.addEventListener("pointerdown", event => {
    if (event.button !== 0 || drag) return;
    event.preventDefault();
    drag = {id: event.pointerId, x: event.clientX, width: paneWidth(),
            previous: workspace.style.getPropertyValue(widthProperty)};
    handle.setPointerCapture(event.pointerId);
    handle.classList.add("active");
    workspace.classList.add("resizing");
  });
  const targetWidth = event => drag.width + (event.clientX - drag.x) * direction;
  handle.addEventListener("pointermove", event => {
    if (drag?.id === event.pointerId) resize(targetWidth(event));
  });
  function finish(event) {
    if (drag?.id !== event.pointerId) return;
    if (event.type === "pointerup") {
      if (Math.abs(event.clientX - drag.x) > 1 && targetWidth(event) < collapseWidth) {
        workspace.style.setProperty(widthProperty, drag.previous);
        setPane(id, false, true);
      }
    } else {
      workspace.style.setProperty(widthProperty, drag.previous);
    }
    drag = null;
    handle.classList.remove("active");
    workspace.classList.remove("resizing");
  }
  for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) {
    handle.addEventListener(type, finish);
  }
  handle.addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const change = (event.key === "ArrowRight" ? 1 : -1) * 24 * direction;
    if (event.key === "Home" || (change < 0 && paneWidth() <= minimumPaneWidth && event.key !== "End")) {
      setPane(id, false, true);
    } else {
      resize(event.key === "End" ? maxWidth() : paneWidth() + change);
    }
  });
}

if (matchMedia("(max-width: 700px)").matches) {
  setPane("ai", false);
  setPane("moves", false);
  setPane("terminal", false);
}

function playerLabel(id, color) {
  const name = color === playerSide() ? "You" : "chess-lm";
  const opponentIsWhite = color === "b";
  const placement = current().fen.split(" ")[0];
  const initial = {p: 8, n: 2, b: 2, r: 2, q: 1};
  const captured = Object.entries(initial).flatMap(([piece, count]) => {
    const token = opponentIsWhite ? piece.toUpperCase() : piece;
    return Array(Math.max(0, count - placement.split(token).length + 1)).fill(pieces[piece][0]);
  }).join("");
  $(id).replaceChildren(element("span", name));
  if (captured) $(id).append(element("span", captured, "captured"));
}

function renderBoard() {
  const board = $("board");
  const focusedSquare = board.contains(document.activeElement) ? document.activeElement.dataset.square : null;
  const position = current();
  const layout = position.fen.split(" ")[0].replace(/[1-8]/g, n => ".".repeat(Number(n))).replaceAll("/", "");
  const last = moves[cursor - 1]?.uci;
  const attempt = live() ? rejected?.attempt : null;
  const flipped = playerSide() === "b";
  const piecesOnBoard = [];
  for (let row = 0; row < 8; row++) {
    for (let col = 0; col < 8; col++) {
      const rank = flipped ? 7 - row : row;
      const file = flipped ? 7 - col : col;
      const coordinate = files[file] + (8 - rank);
      const piece = layout[rank * 8 + file];
      const kind = pieces[piece.toLowerCase()];
      const white = piece === piece.toUpperCase();
      const square = element("button", undefined, "square " + ((rank + file) % 2 ? "dark" : "light"));
      square.type = "button";
      square.dataset.square = coordinate;
      square.disabled = embedded || !canPlay();
      square.setAttribute("aria-label", kind ? coordinate + " " + (white ? "white " : "black ") + kind[1] : coordinate);
      square.classList.toggle("last", [last?.slice(0, 2), last?.slice(2, 4)].includes(coordinate));
      square.classList.toggle("illegal", [attempt?.slice(0, 2), attempt?.slice(2, 4)].includes(coordinate));
      square.classList.toggle("selected", selected === coordinate);
      square.classList.toggle("target", Boolean(selected) && position.legalMoves.some(move => move.startsWith(selected + coordinate)));
      if (kind) {
        square.classList.add("occupied");
        const glyph = element("img", undefined, "piece");
        glyph.alt = "";
        glyph.draggable = false;
        glyph.dataset.art = (white ? "w" : "b") + piece.toUpperCase();
        glyph.src = pieceUrls[glyph.dataset.art];
        square.append(glyph);
      }
      if (row === 7) square.append(element("span", coordinate[0], "coord file"));
      if (col === 0) square.append(element("span", coordinate[1], "coord rank"));
      piecesOnBoard.push(square);
    }
  }
  board.replaceChildren(...piecesOnBoard);
  if (last && cursor === renderedPly + 1 && !motionPreference.matches) {
    const source = piecesOnBoard.findIndex(square => square.dataset.square === last.slice(0, 2));
    const target = piecesOnBoard.findIndex(square => square.dataset.square === last.slice(2, 4));
    const square = piecesOnBoard[target];
    const image = square?.querySelector(".piece");
    if (source >= 0 && image) {
      const x = ((source % 8) - (target % 8)) * 100;
      const y = (Math.floor(source / 8) - Math.floor(target / 8)) * 100;
      square.classList.add("moving");
      image.animate([{transform: "translate(" + x + "%, " + y + "%)"}, {transform: "none"}],
        {duration: 160, easing: "cubic-bezier(.2,.75,.25,1)"})
        .finished.then(() => square.classList.remove("moving"), () => {});
    }
  }
  renderedPly = cursor;
  if (focusedSquare && canPlay()) board.querySelector('[data-square="' + focusedSquare + '"]').focus();
  playerLabel("top-player", flipped ? "w" : "b");
  playerLabel("bottom-player", flipped ? "b" : "w");
}

function renderMoves() {
  const list = $("move-list");
  const focusedPly = list.contains(document.activeElement) ? document.activeElement.dataset.ply : null;
  const rows = [];
  for (let i = 0; i < moves.length; i += 2) {
    const row = element("li");
    row.append(element("span", String(i / 2 + 1) + ".", "move-number"));
    for (const index of [i, i + 1]) {
      if (!moves[index]) continue;
      const button = element("button", moves[index].san);
      button.type = "button";
      button.dataset.ply = index + 1;
      button.disabled = busy;
      button.setAttribute("aria-current", String(cursor === index + 1));
      row.append(button);
    }
    rows.push(row);
  }
  list.replaceChildren(...rows);
  if (focusedPly && !busy) list.querySelector('[data-ply="' + focusedPly + '"]')?.focus({preventScroll: true});
  if (live()) $("move-scroll").scrollTop = $("move-scroll").scrollHeight;
  $("moves-empty").hidden = Boolean(moves.length);
  const lengthLabel = moves.length === 1 ? "1 ply" : moves.length + " plies";
  $("ply-count").textContent = moves.length ? (live() ? lengthLabel : "Ply " + cursor + " / " + moves.length) : "";
  for (const button of document.querySelectorAll("[data-replay]")) {
    const forward = ["next", "live"].includes(button.dataset.replay);
    button.disabled = busy || (forward ? live() : cursor === 0);
  }
  $("undo").disabled = busy || !live() || (!moves.length && !rejected);
  $("undo").title = rejected ? "Revert attempt" : "Undo";
  $("undo").setAttribute("aria-label", $("undo").title);
  $("opening-name").textContent = referenceOpening.slice(0, 5).every((move, index) => history(cursor)[index] === move)
    ? "Ruy López" + (history(cursor)[5] === "a7a6" ? " · Morphy Defense" : "")
    : cursor ? "Standard chess" : "Starting position";
}

function evaluationLabel(line) {
  if (line.mate !== null) return (line.mate > 0 ? "+" : "") + "M" + line.mate;
  return (line.cp >= 0 ? "+" : "") + (line.cp / 100).toFixed(2);
}

function renderInspector() {
  const snapshot = lastPrediction && (live() || lastPrediction.ply === cursor) ? lastPrediction : null;
  const prediction = snapshot?.data;
  $("picks-caption").textContent = prediction
    ? (prediction.mode === "legal" ? "Legal picks" : "Raw picks") + " · " +
      (snapshot.preview ? "current position" : "before ply " + (snapshot.ply + 1))
    : "Play a move to see the model's picks.";
  $("illegal-mass").hidden = !prediction;
  if (prediction) $("illegal-mass").textContent = "Raw illegal probability " + percent.format(prediction.illegalMass) +
    (prediction.temperature === 0 ? " · displayed at T=1" : "");
  const modelRows = (prediction?.picks ?? []).map(pick => {
    const row = element("li", undefined, pick.legal ? (pick.selected ? "chosen" : "") : "invalid");
    row.append(element("span", pick.token + (pick.legal ? "" : " · illegal")), element("span", percent.format(pick.probability)));
    return row;
  });
  $("picks").replaceChildren(...modelRows);
  const currentKey = history(cursor).join(" ");
  const key = snapshot ? snapshot.key : currentKey;
  const engine = analysis ? engineCache.get(key) : null;
  const currentBest = analysis ? engineCache.get(currentKey)?.lines?.[0] : null;
  $("evaluation").hidden = false;
  $("evaluation-fill").style.height = "50%";
  $("evaluation").setAttribute("aria-label", "No engine evaluation for this position");
  if (currentBest) {
    const white = currentBest.mate === null ? 50 + 48 * Math.tanh(currentBest.cp / 420) : currentBest.mate > 0 ? 98 : 2;
    $("evaluation-fill").style.height = (playerSide() === "w" ? white : 100 - white) + "%";
    $("evaluation-fill").style.background = playerSide() === "w" ? "#efefef" : "#353535";
    $("evaluation").style.background = playerSide() === "w" ? "#353535" : "#efefef";
    $("evaluation").setAttribute("aria-label", "Stockfish evaluation for White " + evaluationLabel(currentBest));
  }
  $("engine-caption").textContent = !model.stockfish ? "Stockfish unavailable." : !analysis ? "Analysis off." :
    engine?.error || (engine?.lines ? (engine.lines.length ? "White perspective · " +
      (snapshot ? "before ply " + (snapshot.ply + 1) : "board position") : "Game over.") : "Analysing…");
  $("engine-lines").replaceChildren(...(engine?.lines ?? []).map(line => {
    const row = element("li");
    const pick = prediction?.picks.find(candidate => candidate.token === line.move);
    row.append(element("span", line.san + " · " + line.move), element("span", evaluationLabel(line)),
               element("span", line.sequence + (pick ? " · model " + percent.format(pick.probability) : ""), "sequence"));
    return row;
  }));
  const currentEngine = analysis ? engineCache.get(currentKey) : null;
  $("engine-status").textContent = !model.stockfish ? "Stockfish unavailable" : "Stockfish";
  $("analysis-lines").replaceChildren(...(currentEngine?.lines ?? []).map(line => {
    const row = element("li");
    row.append(element("strong", evaluationLabel(line)), element("span", line.sequence));
    return row;
  }));
}

async function fetchAnalysis(line) {
  if (!analysis) return;
  const key = line.join(" ");
  if (enginePending.has(key) || engineCache.has(key)) return;
  enginePending.add(key);
  try {
    engineCache.set(key, await request("analysis", {history: line}));
  } catch (error) {
    engineCache.set(key, {error: error.message});
  } finally {
    enginePending.delete(key);
    // Render only the currently selected history, even when an older request finishes last.
    renderInspector();
  }
}

function refreshAnalysis() {
  fetchAnalysis(history(cursor));
  if (live() && lastPrediction) fetchAnalysis(history(lastPrediction.ply));
}

function render() {
  if (!model) return;
  renderBoard();
  renderMoves();
  renderInspector();
  $("new-game").disabled = busy;
  $("side").disabled = busy;
  $("only-legal").disabled = busy;
  $("temperature").disabled = busy;
  $("retry").hidden = !live() || turn() === playerSide() || Boolean(current().outcome) || cursor >= model.context;
  $("retry").disabled = busy;
  $("game-status").classList.toggle("error", live() && Boolean(rejected));
  $("game-status").textContent = !live() ? "Reviewing ply " + cursor :
    rejected ? "Illegal AI move · " + rejected.attempt :
    busy && turn() !== playerSide() ? "ChessLM is thinking…" :
    current().outcome ? "Game over · " + current().outcome :
    cursor >= model.context ? "Model context limit reached" :
    notice || (turn() === playerSide() ? "Your move" : "ChessLM to move");
  $("promotion-panel").hidden = !promotion;
  $("analysis-toggle").setAttribute("aria-pressed", String(analysis));
  $("analysis-label").textContent = analysis ? "Analysis on" : "Analysis off";
  $("terminal-analysis").textContent = analysis ? "ANALYSIS ENABLED" : "ANALYSIS OFF";
  $("share-button").disabled = busy;
  for (const button of document.querySelectorAll("[data-side]")) {
    button.disabled = busy;
    if (button.dataset.side !== "random") button.setAttribute("aria-pressed", String(button.dataset.side === playerSide()));
  }
  $("status-analysis").textContent = analysis ? "Analysis on" : "Analysis off";
  $("status-mode").textContent = $("only-legal").checked ? "Only legal moves" : "Free generation";
}

function appendMove(uci, san, position) {
  moves.push({uci, san});
  frames.push(position);
  cursor = moves.length;
}

async function aiMove() {
  if (!model || busy || !live() || turn() === playerSide() || current().outcome || cursor >= model.context) return;
  busy = true;
  rejected = null;
  notice = "";
  const prefix = history();
  const mode = $("only-legal").checked ? "legal" : "free";
  render();
  try {
    const result = await request("attempt", {history: prefix, mode, temperature: Number($("temperature").value)});
    if (mode === "legal" && (!result.legal || result.prediction?.mode !== "legal")) {
      throw new Error("Restart the ChessLM server to enable legal-only play.");
    }
    if (result.prediction) {
      lastPrediction = {ply: prefix.length, key: prefix.join(" "), data: result.prediction};
      const picks = result.prediction.picks.map(pick => pick.token + " " + percent.format(pick.probability)).join("  ");
      modelOutput.push({type: "model", text: "[ply " + (prefix.length + 1) + " · " + mode + "] " + picks});
      renderConsole();
    }
    if (result.legal) {
      appendMove(result.attempt, result.san, result);
      log("[model] " + result.san + " · " + result.attempt, "model");
    } else {
      rejected = result;
      log("[model] Illegal " + result.attempt + " · " + result.reason, "error");
    }
  } catch (error) {
    notice = error.message;
    log("[error] " + error.message, "error");
  } finally {
    busy = false;
    render();
    refreshAnalysis();
  }
}

async function playMove(notation) {
  if (!canPlay()) return;
  busy = true;
  selected = promotion = notice = "";
  render();
  let played = false;
  try {
    const result = await request("position", {history: history(), move: notation});
    appendMove(result.move, result.san, result);
    lastPrediction = null;
    log("[you] " + result.san + " · " + result.move);
    played = true;
  } catch (error) {
    notice = "Invalid move · " + notation;
    log("[error] " + error.message, "error");
  } finally {
    busy = false;
    render();
  }
  if (played) {
    refreshAnalysis();
    await aiMove();
  }
}

function selectSquare(square) {
  if (!canPlay()) return;
  const options = current().legalMoves;
  if (selected && selected !== square) {
    const matches = options.filter(move => move.startsWith(selected + square));
    if (matches.length === 1) return playMove(matches[0]);
    if (matches.length > 1) {
      promotion = selected + square;
      render();
      $("promotion-panel").querySelector("button").focus();
      return;
    }
  }
  promotion = "";
  selected = selected !== square && options.some(move => move.startsWith(square)) ? square : "";
  render();
}

function newGame() {
  if (!model || busy) return;
  moves = [];
  frames = [model];
  cursor = 0;
  selected = promotion = notice = "";
  rejected = lastPrediction = null;
  log("[game] New game · playing " + (playerSide() === "w" ? "White" : "Black") + ".");
  render();
  refreshAnalysis();
  aiMove();
}

function undo() {
  if (!model || busy || !live() || (!rejected && !moves.length)) return;
  if (rejected) {
    rejected = null;
    log("[game] Illegal attempt cleared.");
  } else {
    // In a finished game the last move may belong to either side.
    const lastWasHuman = (moves.length % 2 === 1) === (playerSide() === "w");
    const remove = !lastWasHuman && moves.length >= 2 ? 2 : 1;
    moves.splice(-remove);
    frames.splice(-remove);
    cursor = moves.length;
    log("[game] Move undone.");
  }
  lastPrediction = null;
  selected = promotion = notice = "";
  render();
  refreshAnalysis();
}

function replay(ply) {
  if (!model || busy || ply < 0 || ply > moves.length) return;
  cursor = ply;
  selected = promotion = "";
  render();
  refreshAnalysis();
}

function showOutput(useOutput) {
  outputTab = useOutput;
  for (const [id, active] of [["terminal-tab", !useOutput], ["output-tab", useOutput]]) {
    $(id).setAttribute("aria-selected", String(active));
    $(id).tabIndex = active ? 0 : -1;
  }
  $("console-output").setAttribute("aria-labelledby", useOutput ? "output-tab" : "terminal-tab");
  renderConsole();
}

$("board").addEventListener("click", event => {
  const button = event.target.closest("[data-square]");
  if (button) selectSquare(button.dataset.square);
});
$("promotion-panel").addEventListener("click", event => {
  const piece = event.target.closest("[data-promotion]")?.dataset.promotion;
  if (piece && promotion) playMove(promotion + piece);
});
$("promotion-panel").addEventListener("keydown", event => {
  if (event.key !== "Escape") return;
  promotion = "";
  render();
  $("board").querySelector('[data-square="' + selected + '"]')?.focus();
});
$("move-list").addEventListener("click", event => {
  const target = event.target.closest("[data-ply]");
  if (target) replay(Number(target.dataset.ply));
});
for (const button of document.querySelectorAll("[data-replay]")) {
  button.addEventListener("click", () => replay({
    first: 0, previous: cursor - 1, next: cursor + 1, live: moves.length,
  }[button.dataset.replay]));
}
$("new-game").addEventListener("click", newGame);
for (const button of document.querySelectorAll("[data-side]")) {
  button.addEventListener("click", () => {
    const side = button.dataset.side;
    $("side").value = side === "random" ? (Math.random() < .5 ? "w" : "b") : side;
    newGame();
  });
}
$("retry").addEventListener("click", aiMove);
$("undo").addEventListener("click", undo);
$("only-legal").addEventListener("change", render);
$("analysis-toggle").addEventListener("click", () => {
  analysis = !analysis;
  if (analysis) {
    for (const [key, value] of engineCache) if (value.error) engineCache.delete(key);
  }
  render();
  refreshAnalysis();
});
$("terminal-tab").addEventListener("click", () => showOutput(false));
$("output-tab").addEventListener("click", () => showOutput(true));
for (const tab of [$("terminal-tab"), $("output-tab")]) {
  tab.addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const selectOutput = event.key === "End" || (event.key !== "Home" && !outputTab);
    showOutput(selectOutput);
    $(selectOutput ? "output-tab" : "terminal-tab").focus();
  });
}
$("command-form").addEventListener("submit", async event => {
  event.preventDefault();
  const input = $("command");
  const text = input.value.trim();
  if (!text) return;
  input.value = "";
  showOutput(false);
  log("› " + text, "command");
  if (text === "/help") log("Moves: e4, Nf3, e2e4 · /new · /undo · /retry");
  else if (!model || busy) log("[game] Wait for the current request.", "error");
  else if (text === "/new") newGame();
  else if (!live()) log("[game] Return to the latest position first.", "error");
  else if (text === "/undo") undo();
  else if (text === "/retry") await aiMove();
  else if (text.startsWith("/")) log("[error] Unknown command. Enter /help.", "error");
  else if (canPlay()) await playMove(text);
  else log("[game] Move input is unavailable on this turn.", "error");
});

request("model").then(async result => {
  model = result;
  online = true;
  $("checkpoint").textContent = result.name;
  $("checkpoint").title = result.step == null ? result.name : result.name + " · step " + result.step;
  $("connection").dataset.state = "ready";
  $("connection-text").textContent = "Model connected";
  $("analysis-toggle").disabled = !result.stockfish;
  $("analysis-toggle").title = result.stockfish ? "Toggle Stockfish analysis" : "Stockfish unavailable";
  analysis = Boolean(result.stockfish) && !embedded;
  document.body.dataset.embedded = String(embedded);
  frames = [model];
  busy = true;
  render();
  const start = embedded ? params.get("position").split(" ").filter(Boolean) : referenceOpening;
  try {
    const positions = await Promise.all(start.slice(0, model.context).map((move, index) =>
      request("position", {history: start.slice(0, index), move})));
    for (const position of positions) appendMove(position.move, position.san, position);
    if (start.length && !embedded) {
      log("› position startpos moves " + history().join(" "), "command");
      log("[model] Ruy López · Morphy Defense. White to move.");
      log("[stream] Complete. Waiting for your next move.");
      const prefix = history();
      const key = prefix.join(" ");
      request("attempt", {history: prefix, mode: "free", temperature: Number($("temperature").value)})
        .then(({prediction}) => {
          if (!prediction || history().join(" ") !== key || lastPrediction) return;
          for (const pick of prediction.picks) pick.selected = false;
          lastPrediction = {ply: prefix.length, key, data: prediction, preview: true};
          renderInspector();
        })
        .catch(() => {});
    }
  } catch {
    log("[game] Position could not be restored.");
  } finally {
    busy = false;
    render();
    refreshAnalysis();
  }
}).catch(error => {
  $("checkpoint").textContent = "Unavailable";
  $("game-status").textContent = "Checkpoint unavailable.";
  $("connection").dataset.state = "error";
  $("connection-text").textContent = "Model disconnected";
  log("[error] " + error.message, "error");
});

let shareFormat = "pgn";
let shareObjectUrl;

function selectShareFormat(format) {
  shareFormat = format;
  for (const tab of document.querySelectorAll("[data-share]")) {
    const active = tab.dataset.share === format;
    tab.setAttribute("aria-selected", String(active));
    tab.tabIndex = active ? 0 : -1;
    $("share-" + tab.dataset.share).hidden = !active;
  }
  $("share-notice").textContent = "";
  if (shareObjectUrl) URL.revokeObjectURL(shareObjectUrl);
  shareObjectUrl = undefined;
  const link = $("share-download");
  if (format === "image") {
    link.href = $("share-canvas").toDataURL("image/png");
    link.download = "chesslm-position.png";
    return;
  }
  const source = format === "pgn" ? "share-pgn-value" : "share-embed-value";
  const extension = format === "pgn" ? "pgn" : "html";
  shareObjectUrl = URL.createObjectURL(new Blob([$(source).value], {type: "text/plain;charset=utf-8"}));
  link.href = shareObjectUrl;
  link.download = "chesslm." + extension;
}

function paintShareImage() {
  const canvas = $("share-canvas");
  const ctx = canvas.getContext("2d");
  const size = canvas.width / 8;
  const squares = [...$("board").querySelectorAll(".square")];
  for (const [i, square] of squares.entries()) {
    const x = (i % 8) * size;
    const y = Math.floor(i / 8) * size;
    ctx.fillStyle = getComputedStyle(square).backgroundColor;
    ctx.fillRect(x, y, size, size);
    if (square.classList.contains("last")) {
      ctx.fillStyle = "rgba(155,199,0,.41)";
      ctx.fillRect(x, y, size, size);
    }
    const piece = square.querySelector(".piece");
    if (piece) ctx.drawImage(artImages[piece.dataset.art], x, y, size, size);
  }
}

$("share-button").addEventListener("click", async () => {
  await Promise.all(Object.values(artImages).map(image => image.decode()));
  if ($("share-panel").open) return;
  $("share-fen").value = current().fen;
  const result = current().outcome || "*";
  const tags = [
    ["Event", "ChessLM"], ["Site", "?"], ["Date", "????.??.??"], ["Round", "?"],
    ["White", playerSide() === "w" ? "You" : "ChessLM"],
    ["Black", playerSide() === "b" ? "You" : "ChessLM"], ["Result", result],
  ];
  const notation = moves.slice(0, cursor).flatMap((move, index) =>
    index % 2 === 0 ? [String(index / 2 + 1) + ".", move.san] : [move.san]);
  $("share-pgn-value").value = tags.map(([key, value]) => '[' + key + ' "' + value + '"]').join("\n") +
    "\n\n" + [...notation, result].join(" ") + "\n";
  const url = new URL(location.pathname, location.origin);
  url.searchParams.set("position", history(cursor).join(" "));
  $("share-embed-value").value = '<iframe src="' + url.href +
    '" title="ChessLM position" width="480" height="540" loading="lazy"></iframe>';
  paintShareImage();
  selectShareFormat("pgn");
  $("share-panel").showModal();
  centerSharePanel();
});

function centerSharePanel() {
  const panel = $("share-panel");
  if (!panel.open) return;
  const board = $("board").getBoundingClientRect();
  const inset = 8;
  const clampCenter = (center, size, viewport) =>
    Math.max(size / 2 + inset, Math.min(center, viewport - size / 2 - inset));
  panel.style.setProperty("--share-x", clampCenter(board.left + board.width / 2, panel.offsetWidth, innerWidth) + "px");
  panel.style.setProperty("--share-y", clampCenter(board.top + board.height / 2, panel.offsetHeight, innerHeight) + "px");
}

new ResizeObserver(centerSharePanel).observe($("share-panel"));
window.addEventListener("resize", centerSharePanel);
for (const tab of document.querySelectorAll("[data-share]")) {
  tab.addEventListener("click", () => selectShareFormat(tab.dataset.share));
  tab.addEventListener("keydown", event => {
    const formats = ["pgn", "image", "embed"];
    const index = formats.indexOf(shareFormat);
    const next = {ArrowLeft: (index + 2) % 3, ArrowRight: (index + 1) % 3, Home: 0, End: 2}[event.key];
    if (next === undefined) return;
    event.preventDefault();
    selectShareFormat(formats[next]);
    $("share-" + formats[next] + "-tab").focus();
  });
}

for (const button of document.querySelectorAll("[data-copy]")) {
  button.addEventListener("click", async () => {
    const field = $(button.dataset.copy);
    try {
      await navigator.clipboard.writeText(field.value);
      $("share-notice").textContent = "Copied";
    } catch {
      field.focus();
      field.select();
      $("share-notice").textContent = "Copy the selected text with your keyboard.";
    }
  });
}

$("share-panel").addEventListener("close", () => {
  if (shareObjectUrl) URL.revokeObjectURL(shareObjectUrl);
  shareObjectUrl = undefined;
});
