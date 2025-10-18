from typing import Any, Optional, Self, Tuple

import torch


class SampleRingBuffer:
    """
    Fixed-capacity ring buffer for storing (X, Y) samples and sampling
    uniformly at random. Optimized for preallocated storage and
    wrap-around writes.

    Typical use:
        buf = SampleRingBuffer(200_000, x_shape=(8,), y_shape=(512,), device="cpu")
        buf.append(x_batch, y_batch)        # add data (single row or batches)
        xb, yb, idx = buf.sample(1024)      # random batch
        buf.to("cuda")                      # move storage across devices if needed
    """

    def __init__(
        self,
        capacity: int,
        x_shape: tuple[int, ...] = (8,),
        y_shape: tuple[int, ...] = (512,),
        *,
        dtype_x: torch.dtype = torch.float32,
        dtype_y: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
        pin_memory: bool = False,
    ) -> None:
        """
        Create an empty ring buffer with fixed capacity and shapes.

        Args:
            capacity: Maximum number of rows the buffer can hold (>0).
            x_shape: Shape of a single X row (excluding batch dim).
            y_shape: Shape of a single Y row (excluding batch dim).
            dtype_x: Storage dtype for X.
            dtype_y: Storage dtype for Y.
            device: Device where storage lives initially ("cpu", "cuda", etc.).
            pin_memory: If True and on CPU, allocate pinned memory to speed
                        host->GPU transfers.

        Notes:
            - Storage is preallocated as tensors of shape (capacity, *shape).
            - The buffer starts empty (size==0) and grows until full, then
              wraps around and overwrites oldest rows.
        """
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.pin_memory = bool(pin_memory) and (self.device.type == "cpu")

        alloc_kwargs: dict[str, Any] = {"device": self.device}
        if self.pin_memory:
            alloc_kwargs["pin_memory"] = True

        self.X = torch.empty((self.capacity, *x_shape), dtype=dtype_x, **alloc_kwargs)
        self.Y = torch.empty((self.capacity, *y_shape), dtype=dtype_y, **alloc_kwargs)
        self._size = 0
        self._write = 0  # next write position

        self._x_shape = x_shape
        self._y_shape = y_shape
        self._dtype_x = dtype_x
        self._dtype_y = dtype_y

    @property
    def size(self) -> int:
        """
        Current number of valid rows stored (0..capacity).

        Returns:
            The number of rows available for sampling.
        """
        return self._size

    def __len__(self) -> int:
        """
        Alias for `size` so `len(buffer)` works.

        Returns:
            Current number of valid rows.
        """
        return self._size

    def is_full(self) -> bool:
        """
        Whether the buffer has reached its capacity.

        Returns:
            True if size == capacity, else False.
        """
        return self._size == self.capacity

    def to(self, device: torch.device | str) -> Self:
        """
        Move the underlying storage (X and Y) to another device in-place.

        Args:
            device: Target device ("cpu", "cuda", "cuda:1", etc.).

        Returns:
            self (for chaining).

        Notes:
            - Uses non_blocking transfers where possible.
            - If moving to GPU, pin_memory flag is ignored automatically.
        """
        device = torch.device(device)
        self.X = self.X.to(device, non_blocking=True)
        self.Y = self.Y.to(device, non_blocking=True)
        self.device = device
        self.pin_memory = self.pin_memory and (self.device.type == "cpu")
        return self

    def append(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Append one or many rows to the buffer, overwriting old rows when full.

        Args:
            x: Tensor whose trailing shape equals `x_shape`.
               Leading dims (if any) are flattened into batch size.
               Examples: (8,), (B, 8), (T, B, 8) for x_shape=(8,).
            y: Tensor whose trailing shape equals `y_shape`.
               Must be broadcast-compatible with x's leading batch dims.

        Raises:
            TypeError: If dtypes do not match configured dtypes.
            ValueError: If tail shapes don't match x_shape/y_shape.

        Behavior:
            - If x/y are on a different device than storage, they are copied
              once to the storage device.
            - At most two contiguous copies are performed due to wrap-around.
            - Size increases until capacity is reached; further appends
              overwrite the oldest rows (ring behavior).
        """
        if x.dtype != self._dtype_x or y.dtype != self._dtype_y:
            raise TypeError("dtype mismatch with buffer dtypes")
        if tuple(x.shape[-len(self._x_shape) :]) != self._x_shape:
            raise ValueError(
                f"x tail shape must be {self._x_shape}, got {tuple(x.shape)}"
            )
        if tuple(y.shape[-len(self._y_shape) :]) != self._y_shape:
            raise ValueError(
                f"y tail shape must be {self._y_shape}, got {tuple(y.shape)}"
            )

        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)
        if y.device != self.device:
            y = y.to(self.device, non_blocking=True)

        # Flatten leading dims into batch dimension
        b = int(torch.tensor(x.shape[: -len(self._x_shape)] or (1,)).prod().item())
        x = x.reshape(b, *self._x_shape)
        y = y.reshape(b, *self._y_shape)

        # Write with wrap-around
        first = min(b, self.capacity - self._write)
        self.X[self._write : self._write + first].copy_(x[:first], non_blocking=True)
        self.Y[self._write : self._write + first].copy_(y[:first], non_blocking=True)

        remain = b - first
        if remain > 0:
            self.X[0:remain].copy_(x[first:], non_blocking=True)
            self.Y[0:remain].copy_(y[first:], non_blocking=True)

        self._write = (self._write + b) % self.capacity
        self._size = min(self.capacity, self._size + b)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        *,
        replace: bool = False,
        out_device: torch.device | str | None = None,
        indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Uniformly sample rows from the buffer.

        Args:
            batch_size: Number of rows to sample.
            replace: If False, samples without replacement (requires
                     batch_size <= size). If True, allows duplicates.
            out_device: Optional device to move the sampled batch to.
                        Defaults to storage device for zero-copy returns.
            indices: Optional LongTensor of explicit indices to gather.
                     If provided, RNG is bypassed and `replace` is ignored.

        Returns:
            (x_batch, y_batch, idxs)
            - x_batch: Tensor of shape (batch_size, *x_shape)
            - y_batch: Tensor of shape (batch_size, *y_shape)
            - idxs: LongTensor of chosen indices (on storage device unless
                    out_device is CUDA, then moved accordingly)

        Raises:
            ValueError: If buffer is empty or batch_size too large with replace=False.
            TypeError: If indices is provided but not long dtype.

        Notes:
            - Indices are created on the storage device to avoid device
              mismatches on CUDA for indexing ops.
            - If `out_device` differs from storage, returned batches are
              copied using non_blocking transfers.
        """
        if self._size == 0:
            raise ValueError("buffer is empty")
        if not replace and batch_size > self._size:
            raise ValueError("batch_size > size and replace=False")

        if indices is None:
            if replace:
                idxs = torch.randint(
                    self._size, (batch_size,), device=self.device, dtype=torch.long
                )
            else:
                perm = torch.randperm(self._size, device=self.device)
                idxs = perm[:batch_size]
        else:
            if indices.dtype != torch.long:
                raise TypeError("indices must be torch.LongTensor")
            if indices.device != self.device:
                indices = indices.to(self.device, non_blocking=True)
            idxs = indices

        x = torch.index_select(self.X, dim=0, index=idxs)
        y = torch.index_select(self.Y, dim=0, index=idxs)

        if out_device is None or torch.device(out_device) == self.device:
            return x, y, idxs
        else:
            out_device = torch.device(out_device)
            return (
                x.to(out_device, non_blocking=True),
                y.to(out_device, non_blocking=True),
                idxs.to(out_device if out_device.type == "cuda" else "cpu"),
            )

    def clear(self) -> None:
        """
        Reset the buffer to empty state.

        Notes:
            - Does not deallocate storage; just resets size and write pointer.
            - Existing data will be overwritten by future appends.
        """
        self._size = 0
        self._write = 0

    def state_dict(self) -> dict:
        """
        Create a lightweight checkpoint of the current buffer content.

        Returns:
            A dict containing:
              - "X", "Y": Copies of valid rows (size x *shape)
              - "size": Current size
              - "write": Write pointer
              - "capacity", "x_shape", "y_shape", "dtype_x", "dtype_y"

        Notes:
            - Uses `.clone()` to decouple the checkpoint from live storage.
            - Intended for saving minimal state; shapes/capacity must match when loading.
        """
        return {
            "X": self.X[: self._size].clone(),
            "Y": self.Y[: self._size].clone(),
            "size": self._size,
            "write": self._write,
            "capacity": self.capacity,
            "x_shape": self._x_shape,
            "y_shape": self._y_shape,
            "dtype_x": self._dtype_x,
            "dtype_y": self._dtype_y,
        }

    def load_state_dict(self, state: dict) -> None:
        """
        Restore buffer content from a state dict produced by `state_dict()`.

        Args:
            state: Dictionary with the keys documented in `state_dict()`.

        Raises:
            ValueError: If capacity or shapes differ from this buffer.

        Notes:
            - Only valid rows are copied back.
            - Write pointer is restored; subsequent appends will continue
              from that position (with wrap-around semantics).
        """
        if (
            state["capacity"] != self.capacity
            or state["x_shape"] != self._x_shape
            or state["y_shape"] != self._y_shape
        ):
            raise ValueError("shape/capacity mismatch")
        self.clear()
        n = int(state["size"])
        if n > 0:
            self.X[:n].copy_(state["X"].to(self.device), non_blocking=True)
            self.Y[:n].copy_(state["Y"].to(self.device), non_blocking=True)
        self._size = n
        self._write = int(state["write"]) % self.capacity
