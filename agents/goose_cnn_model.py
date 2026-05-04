"""goose_cnn_model.py - PyTorch CNN + experience buffer for Track G (Goose CNN).

This module ports StochasticGoose (1st place ARC-AGI-3 Developer Preview, Tufa Labs,
12.58% private LB; public Kaggle sample = 0.25). Reference:
- https://github.com/DriesSmit/ARC3-solution
- https://medium.com/@dries.epos/1st-place-in-the-arc-agi-3-agent-preview-competition-49263f6287db

Key architectural choices (mirroring the upstream design):

1. **Input**: 16-channel one-hot encoding of the 64x64 grid (one channel per palette
   color). Preserves discrete palette structure as 2D feature maps.

2. **Backbone**: 4-layer CNN with channel progression 32 -> 64 -> 128 -> 256.
   Stride=1 throughout so spatial dim stays 64x64 -- this is the "convolutional
   2D inductive bias" the upstream uses for the coord head.

3. **Two heads**:
   - **Action head**: global average pool over 256x64x64 -> 256-d vector, then
     linear -> 5 logits (one per ACTION1..ACTION5). Sigmoid output: each logit
     is *independently* the predicted probability of "action will change frame".
   - **Coord head**: 1x1 conv on 256x64x64 -> 1x64x64. Sigmoid output: each
     pixel = predicted P(ACTION6 click at (x,y) changes frame).

4. **Loss**: binary cross-entropy on observed (state, action, frame_changed) tuples.
   Light entropy regularization (configurable) to prevent collapse.

5. **Experience buffer**: hash-de-duplicated dict capped at MAX_BUFFER (default
   50k; upstream uses 200k but we conserve memory for the 6h Kaggle wall).
   Each entry is keyed by (frame_hash, action_id, [click_x, click_y]); each
   value is the binary frame-changed label.

6. **Per-level reset**: model weights and buffer are wiped when a new level is
   reached (gotcha #4). Different levels in the same game can have radically
   different mechanics; cross-level training data poisons the prediction.

This module imports torch lazily inside class methods so that callers without
torch (smoke runner, CI) can still construct an agent (it falls back to a no-op
predictor that returns uniform probabilities).
"""

from __future__ import annotations

import hashlib
import random
from collections import OrderedDict
from typing import Any

# Public constants
NUM_COLORS = 16
GRID_SIZE = 64
NUM_SIMPLE_ACTIONS = 5  # ACTION1..ACTION5; ACTION6 has its own head; ACTION7=Undo
MAX_BUFFER_DEFAULT = 50000


def hash_frame_grid(grid_2d: Any) -> bytes:
    """Stable 8-byte digest of a single 64x64 int grid."""
    h = hashlib.blake2b(digest_size=8)
    if hasattr(grid_2d, "tobytes"):
        h.update(grid_2d.tobytes())
    else:
        for row in grid_2d:
            for v in row:
                h.update(int(v).to_bytes(2, "little", signed=False))
    return h.digest()


def encode_one_hot(grid_2d: Any) -> Any:
    """Convert a 64x64 int grid -> torch.FloatTensor [16, 64, 64]."""
    import numpy as np
    import torch

    arr = grid_2d if isinstance(grid_2d, np.ndarray) else np.asarray(grid_2d, dtype=np.int64)
    arr = np.clip(arr, 0, NUM_COLORS - 1)
    if arr.shape != (GRID_SIZE, GRID_SIZE):
        raise ValueError(f"expected (64, 64) grid, got {arr.shape}")
    onehot = np.zeros((NUM_COLORS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    for c in range(NUM_COLORS):
        onehot[c] = (arr == c).astype(np.float32)
    return torch.from_numpy(onehot)


class ExperienceBuffer:
    """Hash-de-duplicated buffer of (state_hash, action_id, x, y, frame_changed) tuples.

    Backed by an `OrderedDict` so we can evict oldest when the cap is hit.

    For ACTION1..ACTION5, the key is (state_hash, action_id, -1, -1).
    For ACTION6, the key is (state_hash, 6, x, y).
    """

    def __init__(self, max_size: int = MAX_BUFFER_DEFAULT) -> None:
        self.max_size = max_size
        self._data: OrderedDict[tuple[bytes, int, int, int], int] = OrderedDict()

    def __len__(self) -> int:
        return len(self._data)

    def add(
        self,
        state_hash: bytes,
        action_id: int,
        frame_changed: bool,
        click_xy: tuple[int, int] | None = None,
    ) -> None:
        x, y = click_xy or (-1, -1)
        key = (state_hash, int(action_id), int(x), int(y))
        if key in self._data:
            self._data.move_to_end(key)
            self._data[key] = int(bool(frame_changed))
            return
        if len(self._data) >= self.max_size:
            self._data.popitem(last=False)
        self._data[key] = int(bool(frame_changed))

    def clear(self) -> None:
        self._data.clear()

    def sample(self, batch_size: int, rng: random.Random) -> list[tuple[bytes, int, int, int, int]]:
        """Sample up to `batch_size` (state_hash, action_id, x, y, label) tuples."""
        if not self._data:
            return []
        keys = list(self._data.keys())
        n = min(batch_size, len(keys))
        picked = rng.sample(keys, n)
        return [(k[0], k[1], k[2], k[3], self._data[k]) for k in picked]


def _make_torch_model(num_simple: int = NUM_SIMPLE_ACTIONS) -> Any:
    """Build the PyTorch nn.Module. Lazy-imported torch."""
    from torch import nn

    class GooseCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            ch = [NUM_COLORS, 32, 64, 128, 256]
            blocks = []
            for i in range(4):
                blocks.append(nn.Conv2d(ch[i], ch[i + 1], kernel_size=3, padding=1, bias=False))
                blocks.append(nn.BatchNorm2d(ch[i + 1]))
                blocks.append(nn.ReLU(inplace=True))
            self.backbone = nn.Sequential(*blocks)
            self.action_head = nn.Linear(256, num_simple)
            self.coord_head = nn.Conv2d(256, 1, kernel_size=1)

        def forward(self, x: Any) -> tuple[Any, Any]:
            feat = self.backbone(x)  # [B, 256, 64, 64]
            pooled = feat.mean(dim=(2, 3))  # [B, 256]
            action_logits = self.action_head(pooled)  # [B, num_simple]
            coord_logits = self.coord_head(feat).squeeze(1)  # [B, 64, 64]
            return action_logits, coord_logits

    return GooseCNN()


class GooseCNNPredictor:
    """Wraps the torch model with a graceful CPU/no-torch fallback.

    Public API:
        .available -> bool        : True if torch could be imported and model built
        .device    -> str         : "cuda" | "cpu"
        .reset(seed)              : fresh model + optimizer; called on level transitions
        .predict(grid_2d) -> dict : {"action_probs": [5], "coord_probs": [64, 64]}
                                    Always returns numpy arrays in [0, 1]; uniform if
                                    torch is unavailable.
        .update(buffer, n_steps)  : runs `n_steps` mini-batch BCE updates
        .buffer                   : ExperienceBuffer instance (so the agent can add)
    """

    def __init__(
        self,
        seed: int = 0,
        learning_rate: float = 5e-4,
        batch_size: int = 32,
        entropy_weight: float = 0.001,
        device: str | None = None,
        max_buffer: int = MAX_BUFFER_DEFAULT,
    ) -> None:
        self._seed = seed
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._entropy_weight = entropy_weight
        self._desired_device = device
        self._rng = random.Random(seed)
        self.buffer = ExperienceBuffer(max_size=max_buffer)
        self._torch = None
        self._model = None
        self._optim = None
        self.device = "cpu"
        self.available = False
        self._try_init_torch()

    def _try_init_torch(self) -> None:
        try:
            import torch

            self._torch = torch
            torch.manual_seed(self._seed)
            if self._desired_device:
                self.device = self._desired_device
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._build_model_and_optim()
            self.available = True
        except Exception:
            self.available = False
            self._model = None
            self._optim = None

    def _build_model_and_optim(self) -> None:
        if self._torch is None:
            return
        torch = self._torch
        self._model = _make_torch_model().to(self.device)
        self._optim = torch.optim.Adam(
            self._model.parameters(), lr=self._learning_rate, weight_decay=0.0
        )

    def reset(self, seed: int | None = None) -> None:
        """Re-initialize model weights + clear buffer (call on level transition)."""
        self.buffer.clear()
        if seed is not None:
            self._seed = seed
            self._rng = random.Random(seed)
        if self._torch is None:
            return
        self._torch.manual_seed(self._seed)
        self._build_model_and_optim()

    def predict(self, grid_2d: Any) -> dict[str, Any]:
        """Returns {'action_probs': [5,], 'coord_probs': [64, 64]} as numpy arrays."""
        import numpy as np

        if not self.available or self._model is None or self._torch is None:
            return {
                "action_probs": np.full((NUM_SIMPLE_ACTIONS,), 0.5, dtype=np.float32),
                "coord_probs": np.full((GRID_SIZE, GRID_SIZE), 0.5, dtype=np.float32),
            }
        torch = self._torch
        self._model.eval()
        with torch.no_grad():
            x = encode_one_hot(grid_2d).unsqueeze(0).to(self.device)
            action_logits, coord_logits = self._model(x)
            ap = torch.sigmoid(action_logits).squeeze(0).detach().cpu().numpy()
            cp = torch.sigmoid(coord_logits).squeeze(0).detach().cpu().numpy()
        return {"action_probs": ap.astype(np.float32), "coord_probs": cp.astype(np.float32)}

    def update(self, n_steps: int, grids_by_hash: dict[bytes, Any]) -> dict[str, float]:
        """Mini-batch BCE on the experience buffer.

        `grids_by_hash` maps state_hash -> the most recent 64x64 grid that produced
        it (we re-encode lazily; we keep the dict keyed on hash so we don't have
        to store the grid in the buffer itself).

        Returns a small stats dict for logging.
        """
        if not self.available or self._model is None or self._optim is None or self._torch is None:
            return {"loss": 0.0, "n_updates": 0}
        if n_steps <= 0 or len(self.buffer) < self._batch_size:
            return {"loss": 0.0, "n_updates": 0}
        torch = self._torch
        from torch.nn import functional as nn_func

        self._model.train()
        total_loss = 0.0
        n_done = 0
        for _ in range(n_steps):
            samples = self.buffer.sample(self._batch_size, self._rng)
            xs, action_targets, action_mask, coord_targets, coord_mask = [], [], [], [], []
            for state_hash, action_id, x, y, label in samples:
                grid = grids_by_hash.get(state_hash)
                if grid is None:
                    continue
                xs.append(encode_one_hot(grid))
                a_t = [0.0] * NUM_SIMPLE_ACTIONS
                a_m = [0.0] * NUM_SIMPLE_ACTIONS
                c_t = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]
                c_m = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]
                if 1 <= action_id <= NUM_SIMPLE_ACTIONS:
                    a_t[action_id - 1] = float(label)
                    a_m[action_id - 1] = 1.0
                elif action_id == 6 and 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    c_t[y][x] = float(label)
                    c_m[y][x] = 1.0
                action_targets.append(a_t)
                action_mask.append(a_m)
                coord_targets.append(c_t)
                coord_mask.append(c_m)
            if not xs:
                continue
            x_batch = torch.stack(xs, dim=0).to(self.device)
            at = torch.tensor(action_targets, dtype=torch.float32, device=self.device)
            am = torch.tensor(action_mask, dtype=torch.float32, device=self.device)
            ct = torch.tensor(coord_targets, dtype=torch.float32, device=self.device)
            cm = torch.tensor(coord_mask, dtype=torch.float32, device=self.device)
            action_logits, coord_logits = self._model(x_batch)
            loss_a_raw = nn_func.binary_cross_entropy_with_logits(
                action_logits, at, reduction="none"
            )
            loss_c_raw = nn_func.binary_cross_entropy_with_logits(
                coord_logits, ct, reduction="none"
            )
            loss_a = (loss_a_raw * am).sum() / (am.sum() + 1e-6)
            loss_c = (loss_c_raw * cm).sum() / (cm.sum() + 1e-6)
            ap = torch.sigmoid(action_logits)
            ent = -(ap * torch.log(ap + 1e-6) + (1 - ap) * torch.log(1 - ap + 1e-6)).mean()
            loss = loss_a + loss_c - self._entropy_weight * ent
            self._optim.zero_grad()
            loss.backward()
            self._optim.step()
            total_loss += float(loss.detach().cpu())
            n_done += 1
        return {"loss": total_loss / max(n_done, 1), "n_updates": n_done}


__all__ = [
    "GRID_SIZE",
    "MAX_BUFFER_DEFAULT",
    "NUM_COLORS",
    "NUM_SIMPLE_ACTIONS",
    "ExperienceBuffer",
    "GooseCNNPredictor",
    "encode_one_hot",
    "hash_frame_grid",
]
