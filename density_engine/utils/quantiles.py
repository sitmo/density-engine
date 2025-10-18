from bisect import bisect_left
from math import floor
from typing import List, Tuple


# ---- Minimal GK quantile sketch ----
class GKQuantiles:
    """
    Greenwald–Khanna (GK) streaming quantile sketch.

    Purpose:
        Maintain ε-approximate quantiles of a data stream using sublinear memory.
        For any q in [0,1], `query(q)` returns a value whose true rank differs
        from q*N by at most ε*N (N is the number of inserted items).

    Typical use:
        gk = GKQuantiles(eps=1e-2)
        for x in stream: gk.insert(x)
        p95 = gk.query(0.95)

    Notes:
        - Deterministic; no randomness.
        - Time: amortized O(1) per insert, O(#summaries) per query.
        - Memory: O((1/ε) * log(ε*N)).
        - Works on numeric values; values are coerced to float.
    """

    def __init__(self, eps: float = 1e-2):
        """
        Create an empty GK sketch.

        Args:
            eps: Error parameter ε in (0,1). Rank error ≤ ε * N.

        Raises:
            ValueError: If eps is not in (0,1).

        Attributes:
            eps (float): Configured ε.
            n (int): Number of inserted items.
            S (list[tuple[float,int,int]]): Summary tuples (value, g, Δ),
                kept sorted by value. Internal.
        """
        if not (0 < eps < 1):
            raise ValueError("eps must be in (0,1)")
        self.eps = eps
        self.n = 0  # total count
        self.S: list[tuple[float, int, int]] = []  # (value, g, Δ), sorted by value

    def insert(self, x: float) -> None:
        """
        Insert one value into the sketch.

        Args:
            x: Numeric value (coerced to float).

        Notes:
            - Occasionally triggers a `_compress()` to keep the summary small.
            - Amortized O(1) time; worst case O(#summaries) when compressing.
        """
        x = float(x)
        self.n += 1
        if not self.S:
            self.S.append((x, 1, 0))
            return

        # find position by value
        i = bisect_left([v for (v, _, _) in self.S], x)
        if i == 0:
            self.S.insert(0, (x, 1, 0))
        elif i == len(self.S):
            self.S.append((x, 1, 0))
        else:
            # GK invariant: g_i + Δ_i ≤ floor(2εN)
            delta = floor(2 * self.eps * self.n) - 1
            if delta < 0:
                delta = 0
            self.S.insert(i, (x, 1, delta))

        # occasional compress keeps size small
        if self.n % max(1, int(1 / (2 * self.eps))) == 0:
            self._compress()

    def _compress(self) -> None:
        """
        Merge adjacent summary entries when the GK invariant allows it.

        Internal:
            - Scans summaries once; O(#summaries).
            - Preserves correctness guarantees.
        """
        if not self.S:
            return
        S2 = [self.S[0]]
        for j in range(1, len(self.S)):
            v, g, d = self.S[j]
            v_prev, g_prev, d_prev = S2[-1]
            if g_prev + g + d <= floor(2 * self.eps * self.n):
                # merge into previous
                S2[-1] = (v, g_prev + g, d_prev)
            else:
                S2.append((v, g, d))
        self.S = S2

    def query(self, q: float) -> float:
        """
        Query an approximate q-quantile.

        Args:
            q: Quantile in [0,1]. Example: q=0.5 (median), 0.95 (95th).

        Returns:
            A value v whose rank is within ±ε*N of q*N.

        Raises:
            ValueError: If q not in [0,1] or the sketch is empty.

        Notes:
            - Iterates the compact summary; O(#summaries).
        """
        if not (0.0 <= q <= 1.0):
            raise ValueError("q must be in [0,1]")
        if not self.S:
            raise ValueError("empty sketch")

        r = 0
        rank_target = q * self.n
        tol = self.eps * self.n
        for v, g, d in self.S:
            r_next = r + g
            r_max = r_next + d
            if r_max >= rank_target - tol:
                return v
            r = r_next
        return self.S[-1][0]


# ---- Two-buffer, flip-every-window_step wrapper ----
class WindowQuantiles:
    """
    Sliding-window-ish quantiles via two GK buffers with periodic flipping.

    Idea:
        Maintain two independent GK sketches. New inserts go to the "active"
        sketch. Every `window_step` inserts, flip which sketch is active and
        reset the new active one. Queries prefer the "holdover" (non-active)
        sketch if it has data; otherwise fall back to the active sketch.

    What window does this approximate?
        Roughly the last [window_step, 2*window_step) items:
        - After a flip: holdover has ~window_step most recent items,
          active is filling (0..window_step). Using holdover yields a stable
          ~window_step-sized window. During early fill, active is used.

    Typical use:
        wq = WindowQuantiles(eps=1e-2, window_step=5_000)
        for x in stream: wq.insert(x)
        p98 = wq.query(0.98)
    """

    def __init__(self, eps: float = 1e-2, window_step: int = 5_000):
        """
        Initialize the two-buffer windowed quantile tracker.

        Args:
            eps: GK ε parameter for both internal sketches.
            window_step: Number of inserts between flips (>0).

        Raises:
            ValueError: If window_step <= 0.

        Attributes:
            window_step (int): Flip period in inserts.
            buffers (list[GKQuantiles, GKQuantiles]): Two sketches.
            active (int): Index of the active buffer (0 or 1).
            count_since_flip (int): Inserts written since last flip.
        """
        if window_step <= 0:
            raise ValueError("window_step must be > 0")
        self.window_step = int(window_step)
        self.buffers = [GKQuantiles(eps), GKQuantiles(eps)]
        self.active = 0
        self.count_since_flip = 0

    def insert(self, x: float) -> None:
        """
        Insert a value and manage flips when the step is reached.

        Args:
            x: Numeric value (coerced to float).

        Behavior:
            - Writes to the active GK sketch.
            - When `count_since_flip >= window_step`, flips active index,
              clears the new active sketch, and resets the counter.
        """
        # rotate if current window is full
        if self.count_since_flip >= self.window_step:
            self.active = 1 - self.active  # flip active
            # clear the new active by replacing with a fresh GKQuantiles
            self.buffers[self.active] = GKQuantiles(self.buffers[self.active].eps)
            self.count_since_flip = 0

        # add to active
        self.buffers[self.active].insert(x)
        self.count_since_flip += 1

    def query(self, q: float = 0.98) -> float:
        """
        Query an approximate q-quantile from the recent window.

        Args:
            q: Quantile in [0,1]. Default 0.98 (98th percentile).

        Returns:
            Approximate windowed quantile. Prefers the holdover buffer if it
            has data; otherwise uses the active buffer.

        Raises:
            ValueError: If no data has been inserted yet.
        """
        hold = 1 - self.active
        if self.buffers[hold].n > 0:
            return self.buffers[hold].query(q)
        # early phase: use whatever we have
        if self.buffers[self.active].n > 0:
            return self.buffers[self.active].query(q)
        raise ValueError("no data yet")
