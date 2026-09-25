import io
import logging
import multiprocessing as mp
import queue
import sys
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from typing import Any, Callable, Iterable, List, Optional, Sequence

from tqdm import tqdm

_worker_slot: Optional[int] = None
_current_seed: Optional[int] = None
_message_queue = None


def _init_worker(slot_queue, tqdm_lock, message_queue):
    global _worker_slot, _message_queue
    _worker_slot = slot_queue.get()
    _message_queue = message_queue
    tqdm.set_lock(tqdm_lock)
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)


def worker_slot() -> Optional[int]:
    return _worker_slot


def set_current_seed(seed: int):
    global _current_seed
    _current_seed = seed


class _Utf8Buffer(io.StringIO):
    # wandb drops the "Run history" sparklines for streams without a unicode encoding
    encoding = "utf-8"


@contextmanager
def captured_output():
    """In a pool worker, capture stdout/stderr into a buffer; outside a pool, do nothing and yield None."""
    if _worker_slot is None:
        yield None
        return
    buffer = _Utf8Buffer()
    with redirect_stdout(buffer), redirect_stderr(buffer):
        yield buffer


def send_to_console(text: str):
    """Print text; from a pool worker, hand it to the main process so it isn't drawn over the progress bars."""
    if _message_queue is None:
        print(text)
    else:
        _message_queue.put(text)


def _print_messages(message_queue, lock, timeout: float = 0.0):
    while True:
        try:
            text = message_queue.get(timeout=timeout) if timeout else message_queue.get_nowait()
        except queue.Empty:
            return
        with lock:
            # \r\x1b[J clears the worker bars below the cursor; they redraw beneath the text
            sys.stdout.write("\r\x1b[J" + text.rstrip("\n") + "\n")
            sys.stdout.flush()


def _seed_label(desc: str) -> str:
    return f"seed {_current_seed} | {desc}" if desc else f"seed {_current_seed}"


class _SeedTqdm(tqdm):
    def set_description(self, desc=None, refresh=True):
        super().set_description(_seed_label(desc), refresh)


def progress_bar(iterable: Iterable) -> tqdm:
    if _worker_slot is None:
        return tqdm(iterable)
    return _SeedTqdm(
        iterable, desc=_seed_label(""), position=_worker_slot, leave=False, dynamic_ncols=True
    )


def lightning_progress_callbacks() -> list:
    if _worker_slot is None:
        return []

    from lightning.pytorch.callbacks import TQDMProgressBar
    from lightning.pytorch.callbacks.progress.tqdm_progress import Tqdm

    class _SeedLightningTqdm(Tqdm):
        def set_description(self, desc=None, refresh=True):
            super().set_description(_seed_label(desc), refresh)

    class _SeedProgressBar(TQDMProgressBar):
        def init_train_tqdm(self):
            return _SeedLightningTqdm(
                desc=_seed_label(self.train_description),
                position=_worker_slot,
                disable=self.is_disabled,
                leave=False,
                dynamic_ncols=True,
                file=sys.stdout,
                smoothing=0,
                bar_format=self.BAR_FORMAT,
            )

        # only the training bar is shown per worker; sanity/validation/test bars are hidden
        def init_sanity_tqdm(self):
            return Tqdm(disable=True)

        def init_validation_tqdm(self):
            return Tqdm(disable=True)

        def init_test_tqdm(self):
            return Tqdm(disable=True)

    return [_SeedProgressBar(process_position=_worker_slot)]


def run_seeds(fn: Callable, args_list: Sequence[tuple], num_concurrent: int = 1) -> List[Any]:
    if num_concurrent <= 1:
        return [fn(*args) for args in args_list]

    ctx = mp.get_context("spawn")
    slot_queue = ctx.Queue()
    for slot in range(num_concurrent):
        slot_queue.put(slot)
    message_queue = ctx.Queue()
    console_lock = ctx.RLock()
    pool = ProcessPoolExecutor(
        max_workers=num_concurrent,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(slot_queue, console_lock, message_queue),
    )
    futures = [pool.submit(fn, *args) for args in args_list]
    try:
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=1.0, return_when=FIRST_EXCEPTION)
            _print_messages(message_queue, console_lock)
            for future in done:
                future.result()
    except BaseException:
        workers = list((pool._processes or {}).values())
        pool.shutdown(wait=False, cancel_futures=True)
        for worker in workers:
            worker.terminate()
        raise

    pool.shutdown()
    _print_messages(message_queue, console_lock, timeout=0.5)
    return [future.result() for future in futures]
