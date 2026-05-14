from typing import Any, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.rich_progress import (
    RichProgressBar,
    MetricsTextColumn,
)

# Rich imports used inside custom progress column
from rich import get_console, reconfigure
from rich.table import Table
from rich.progress import ProgressColumn, Task, RenderableType
from rich.console import Group
from rich.live import Live
from pytorch_lightning.callbacks import TQDMProgressBar


class CompactTQDMProgressBar(TQDMProgressBar):
    # Purpose: Keep only selected metrics, merge into one short line and trim length.
    # Additionally, allow per-metric display formats (e.g., PSNR with 3 decimals, LR full precision).
    def __init__(
        self,
        keep_keys=(
            "train/loss",
            "train/RUsage",
            "train/psnr_0",
            "train/psnr_1",
            "train/psnr_2",
            "train/psnr_3",
        ),
        max_len=160,
        format_map: Optional[dict[str, str]] = None,
    ):
        super().__init__()
        self.keep_keys = set(keep_keys)
        self.max_len = max_len
        # format_map: metric_name -> format string, e.g. "{:.3f}" or "{}"
        self.format_map = format_map or {}

    def _format_value(self, key: str, value: float) -> str:
        # Purpose: Format a single metric according to format_map or built-in heuristics.
        fmt = self.format_map.get(key)
        if fmt is not None:
            try:
                return fmt.format(value)
            except Exception:
                return str(value)

        key_lower = key.lower()
        # Heuristic defaults: more precision for lr, 3 decimals for PSNR, 2 decimals otherwise.
        if "lr" in key_lower:
            return f"{value:.2e}"  # show full float representation for learning rate
        if "psnr" in key_lower:
            return f"{value:.3f}"
        return f"{value:.2f}"

    # Purpose: Filter, format and compact metrics into a single short string.
    def get_metrics(self, trainer, pl_module):
        m = super().get_metrics(trainer, pl_module)
        m.pop("v_num", None)  # hide version
        # keep only selected numeric metrics
        kept = []
        keep_keys = self.keep_keys
        for k, v in m.items():
            if k in keep_keys and isinstance(v, (int, float)):
                # strip prefix/suffix for readability, but avoid mutating the original metrics dict
                if "psnr" in k:
                    k = k.replace("psnr", "P")
                kept.append((k.split("/")[-1].replace("_step", ""), self._format_value(k, v)))
        # build one compact string, e.g. "loss=1.23 psnr_0=16.81 psnr_0_pseudo=16.89"
        parts = []
        # Sort metrics by key before outputting
        for k, v in sorted(kept, key=lambda item: item[0]):
            parts.append(f"{k}:{v}")
        line = " ".join(parts)
        if len(line) > self.max_len:
            line = line[: self.max_len - 1] + "…"
        return {"train": line}  # only one concise field

    # Optional: fix width so tqdm does not wrap when the terminal width changes.
    def init_train_tqdm(self):
        bar = super().init_train_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 160  # Adjust according to terminal width.
        return bar

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.dynamic_ncols = False
        bar.ncols = 160
        return bar

    def _is_rank0(self, trainer) -> bool:
        return trainer.is_global_zero

    def on_train_batch_end(self, trainer, pl_module, *args, **kwargs):
        if not self._is_rank0(trainer):
            return  # Skip non-rank-0 processes.
        return super().on_train_batch_end(trainer, pl_module, *args, **kwargs)

    def on_validation_batch_end(self, trainer, pl_module, *args, **kwargs):
        if not self._is_rank0(trainer):
            return
        return super().on_validation_batch_end(trainer, pl_module, *args, **kwargs)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not self._is_rank0(trainer):
            return
        return super().on_validation_epoch_end(trainer, pl_module)


class MetricsTableColumn(MetricsTextColumn):
    # Purpose: Render metrics as a Rich Table instead of plain text.
    def __init__(self, trainer: "pl.Trainer", style: str, metrics_format: str) -> None:
        super().__init__(trainer, style=style, text_delimiter=" ", metrics_format=metrics_format)

    def render(self, task: "Task") -> RenderableType:  # type: ignore[override]
        # Purpose: Build and return a Rich Table for the current training task.
        assert isinstance(self._trainer.progress_bar_callback, RichProgressBar)
        if (
            self._trainer.state.fn != "fit"
            or self._trainer.sanity_checking
            or self._trainer.progress_bar_callback.train_progress_bar_id != task.id
        ):
            return Table.grid()

        if self._trainer.training and task.id not in self._tasks:
            self._tasks[task.id] = Table.grid()
            if self._renderable_cache:
                self._current_task_id = self._current_task_id  # keep current id
                self._tasks[self._current_task_id] = self._renderable_cache[self._current_task_id][1]
            self._current_task_id = task.id

        if self._trainer.training and task.id != self._current_task_id:
            return self._tasks[task.id]

        table = Table(show_header=True, header_style=str(self._style) if self._style else "")
        table.add_column("Metric", justify="left")
        table.add_column("Value", justify="right")

        for name, value in self._metrics.items():
            if not isinstance(value, (str, int)):
                value = f"{value:{self._metrics_format}}"
            table.add_row(str(name), str(value))

        return table


class TableRichProgressBar(RichProgressBar):
    # Purpose: Stack a metrics table ABOVE the progress bar using Rich Live layout.
    def _init_progress(self, trainer: "pl.Trainer") -> None:
        # Purpose: Initialize a Live(Group(metrics_table, progress)) layout.
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            reconfigure(**self._console_kwargs)
            self._console = get_console()

            if hasattr(self._console, "_live_stack"):
                if len(self._console._live_stack) > 0:
                    self._console.clear_live()
            else:
                self._console.clear_live()

            # Metrics will be rendered above the bar; keep a local store
            self._stored_metrics: dict[str, Any] = {}

            # Build the underlying progress WITHOUT the metrics column
            from pytorch_lightning.callbacks.progress.rich_progress import CustomProgress

            self.progress = CustomProgress(
                *self.configure_columns(trainer),
                auto_refresh=False,
                disable=self.is_disabled,
                console=self._console,
            )

            # Create a top-level Live that stacks table over progress
            self._live = Live(
                Group(self._build_metrics_table(), self.progress),
                console=self._console,
                refresh_per_second=12,
            )
            self._live.start()
            self._progress_stopped = False

    def _build_metrics_table(self) -> Table:
        # Purpose: Build a Rich Table from the stored metrics.
        table = Table(show_header=True, header_style=str(self.theme.metrics) if self.theme.metrics else "")
        table.add_column("Metric", justify="left")
        table.add_column("Value", justify="right")
        for name, value in self._stored_metrics.items():
            if not isinstance(value, (str, int)):
                value = f"{value:{self.theme.metrics_format}}"
            table.add_row(str(name), str(value))
        return table

    def refresh(self) -> None:  # type: ignore[override]
        # Purpose: Refresh Live layout with latest metrics and progress state.
        if getattr(self, "_live", None) is not None:
            self._live.update(Group(self._build_metrics_table(), self.progress), refresh=True)  # type: ignore[arg-type]

    def _get_train_description(self, current_epoch: int) -> str:
        # Purpose: Show global iteration/step instead of epoch.
        desc = f"Iteration"
        if len(self.validation_description) > len(desc):
            desc = f"{desc:{len(self.validation_description)}}"
        return desc

    def _update_metrics(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        current: Optional[int] = None,
        total_batches: bool = False,
    ) -> None:  # type: ignore[override]
        # Purpose: Store metrics and trigger a refresh so table updates.
        if not self.is_enabled:
            return
        if current is not None and not total_batches:
            total = self.total_train_batches
            if not self._should_update(current, total):
                return
        metrics = self.get_metrics(trainer, pl_module)
        self._stored_metrics = metrics
        self.refresh()

    def _stop_progress(self) -> None:  # type: ignore[override]
        # Purpose: Stop Live properly.
        if getattr(self, "_live", None) is not None:
            self._live.stop()  # type: ignore[union-attr]
            self._live = None  # type: ignore[assignment]
            self._progress_stopped = True