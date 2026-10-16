"""Shared infrastructure for feature extraction and archive persistence.

The functions in this module deliberately do not know about audio features.
They provide extraction checkpointing, atomic persistence, chunked NPZ writing,
and cleanup.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
import inspect
import logging
from time import perf_counter, sleep
from pathlib import Path
from typing import Any, Callable, Generic, Iterator, Mapping, ParamSpec, Protocol, TypeVar, cast
from zipfile import ZIP_DEFLATED, ZipFile
import json
import shutil

import numpy as np


def iter_chunks(size: int, chunk_size: int) -> Iterator[tuple[int, int]]:
	"""Yield half-open row ranges covering an array of ``size`` rows."""
	for start in range(0, size, chunk_size):
		yield start, min(start + chunk_size, size)


def atomic_write(
	path: Path,
	writer: Callable[[Path], Any],
	temporary_suffix: str,
) -> None:
	"""Write a file through a temporary path and replace it safely on Windows."""
	temporary_path = path.with_suffix(temporary_suffix)
	try:
		writer(temporary_path)
		replace_with_retries(temporary_path, path)
	finally:
		if temporary_path.exists():
			temporary_path.unlink()


def replace_with_retries(temporary_path: Path, path: Path) -> None:
	"""Replace ``path``, retrying transient Windows file locks."""
	for attempt in range(3):
		try:
			temporary_path.replace(path)
			return
		except PermissionError:
			if attempt == 2:
				break
			sleep(0.2 * (attempt + 1))

	path.unlink(missing_ok=True)
	temporary_path.replace(path)


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
	"""Persist a JSON checkpoint atomically."""
	atomic_write(
		path,
		lambda temporary_path: temporary_path.write_text(
			json.dumps(state), encoding="utf-8"
		),
		".tmp",
	)


def load_checkpoint(path: Path) -> dict[str, Any]:
	"""Load a JSON checkpoint."""
	return json.loads(path.read_text(encoding="utf-8"))


def close_memmaps(memmaps: Mapping[str, np.memmap]) -> None:
	"""Flush and close memory maps so their files can be removed on Windows."""
	for mapped_array in memmaps.values():
		mapped_array.flush()
		mmap_handle = getattr(mapped_array, "_mmap", None)
		if mmap_handle is not None:
			mmap_handle.close()


def _write_array_to_zip(
	archive: ZipFile,
	name: str,
	array: np.ndarray,
	*,
	rows: np.ndarray | None,
	chunk_size: int,
) -> None:
	shape = (len(rows), *array.shape[1:]) if rows is not None else array.shape
	# ZIP64 is required because individual feature members can exceed 4 GiB.
	with archive.open(f"{name}.npy", "w", force_zip64=True) as stream:
		header = np.lib.format.header_data_from_array_1_0(array)
		header["shape"] = shape
		np.lib.format.write_array_header_1_0(stream, header)
		row_indices = np.arange(array.shape[0]) if rows is None else rows
		for start, end in iter_chunks(len(row_indices), chunk_size):
			values = np.asarray(array[row_indices[start:end]])
			stream.write(values.tobytes(order="C"))


def write_feature_archive(
	path: Path,
	arrays: Mapping[str, np.ndarray],
	feature_names: tuple[str, ...],
	*,
	rows: np.ndarray | None = None,
	chunk_size: int = 1024,
) -> None:
	"""Write feature arrays to a chunked, ZIP64-compatible NPZ archive."""
	logging.info("Starting archive write: %s", path.name)
	with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
		for feature_number, name in enumerate(feature_names, start=1):
			logging.info(
				"Writing archive feature %d/%d: %s",
				feature_number,
				len(feature_names),
				name,
			)
			_write_array_to_zip(
				archive,
				name,
				arrays[name],
				rows=rows,
				chunk_size=chunk_size,
			)
			logging.info("Finished archive feature %d/%d: %s", feature_number, len(feature_names), name)
	logging.info("Finished archive write: %s", path.name)


class ResumableJob:
	"""Manage checkpoints, memory maps, atomic finalization, and timing."""

	def __init__(
		self,
		output_dir: Path,
		output_path: Path,
		checkpoint_name: str,
		configuration: dict[str, Any],
		overwrite: bool,
	) -> None:
		self.output_path = output_path
		self.checkpoint_dir = output_dir / checkpoint_name
		self.checkpoint_path = self.checkpoint_dir / "manifest.json"
		self.memmaps: dict[str, np.memmap] = {}
		self.finished = False
		self.started_at = perf_counter()

		if overwrite and self.checkpoint_dir.exists():
			shutil.rmtree(self.checkpoint_dir)
		if self.checkpoint_path.exists():
			self.state = load_checkpoint(self.checkpoint_path)
			if self.state["configuration"] != configuration:
				raise ValueError(
					f"{checkpoint_name} configuration differs; use overwrite=True"
				)
		else:
			self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
			self.state = {"configuration": configuration, "completed": [], "specs": {}}
			self.save()

		specs = cast(dict[str, dict[str, Any]], self.state["specs"])
		for name, spec in specs.items():
			self.open_memmap(
				name,
				dtype=spec["dtype"],
				shape=tuple(spec["shape"]),
				mode="r+",
			)

	def save(self) -> None:
		"""Persist the current manifest."""
		save_checkpoint(self.checkpoint_path, self.state)

	def open_memmap(
		self,
		name: str,
		*,
		dtype: str | np.dtype[Any],
		shape: tuple[int, ...],
		mode: str = "w+",
	) -> np.memmap:
		"""Create or reopen a checkpoint-backed NumPy memory map."""
		mapped_array = np.lib.format.open_memmap(
			self.checkpoint_dir / f"{name}.npy",
			mode=mode,
			dtype=dtype,
			shape=shape,
		)
		self.memmaps[name] = mapped_array
		specs = cast(dict[str, dict[str, Any]], self.state["specs"])
		specs[name] = {"dtype": str(mapped_array.dtype), "shape": list(shape)}
		self.save()
		return mapped_array

	def mark_completed(self, item: int | str) -> None:
		"""Record one completed file or feature and persist the checkpoint."""
		completed = cast(list[int | str], self.state["completed"])
		if item not in completed:
			completed.append(item)
		self.save()

	def finalize(self, writer: Callable[[Path], Any]) -> None:
		"""Publish one output artifact atomically."""
		self.finalize_artifacts({self.output_path: writer})

	def finalize_artifacts(
		self, writers: Mapping[Path, Callable[[Path], Any]]
	) -> None:
		"""Publish multiple artifacts only after all temporary writes succeed."""
		temporary_paths: dict[Path, Path] = {}
		try:
			for path, writer in writers.items():
				temporary_path = path.with_suffix(f".tmp{path.suffix}")
				temporary_paths[path] = temporary_path
				logging.info("Finalizing artifact: %s", path.name)
				writer(temporary_path)
				logging.info("Finished writing artifact: %s", path.name)
			for path, temporary_path in temporary_paths.items():
				logging.info("Publishing artifact: %s", path.name)
				replace_with_retries(temporary_path, path)
				logging.info("Published artifact: %s", path.name)
			self.finished = True
		finally:
			for temporary_path in temporary_paths.values():
				if temporary_path.exists():
					temporary_path.unlink()

	def __enter__(self) -> ResumableJob:
		return self

	def __exit__(
		self, exc_type: Any, _exc_value: Any, _traceback: Any
	) -> None:
		del _exc_value, _traceback
		close_memmaps(self.memmaps)
		self.memmaps.clear()
		if self.finished:
			shutil.rmtree(self.checkpoint_dir)
		if exc_type is None:
			elapsed = perf_counter() - self.started_at
			logging.info("Completed %s in %.2f seconds", self.output_path.name, elapsed)


def get_output_dir(dataset_name: str) -> Path:
	"""Return the output directory for a dataset."""
	project_dir = Path(__file__).resolve().parent
	output_dir = project_dir / "training" / dataset_name
	output_dir.mkdir(parents=True, exist_ok=True)
	return output_dir

T = TypeVar("T")

@dataclass
class TaskResult(Generic[T]):
	result: T
	results_file_saver: Callable[..., object]
	timer_log: Callable[[float], object]
	model_file_saver: Callable[..., object] | None = None

P = ParamSpec("P")
R = TypeVar("R")


class DataFromFileCallable(Protocol[P, R]):
	compute: Callable[..., TaskResult[R]]

	def __call__(
		self, *args: P.args, **kwargs: P.kwargs
	) -> object: ...

def data_from_file(
	*,
	dataset_name: str,
	input_filename: str | Callable,
	output_filename: str | Callable,
	overwrite: bool = False,
	model_filename: str | None = None,
) -> Callable[
	[Callable[..., TaskResult[R]]],
	DataFromFileCallable[..., R],
]:
	def decorator(
		func: Callable[..., TaskResult[R]],
	) -> DataFromFileCallable[..., R]:
		signature = inspect.signature(func)
		accepts_output_dir = "output_dir" in signature.parameters


		@wraps(func)
		def wrapper(
			*args: Any,
			dataset_name: str = dataset_name,
			input_filename: str | Callable = input_filename,
			output_filename: str | Callable = output_filename,
			model_filename: str | None = model_filename,
			overwrite: bool = overwrite,
			**kwargs: Any,
		) -> object:
			output_dir = get_output_dir(dataset_name)

			bound = signature.bind_partial(*args, **kwargs)
			bound.apply_defaults()
			params = bound.arguments
			# Support either fixed filenames or callable resolvers
			input_name = (
				input_filename(**params)
				if callable(input_filename)
				else input_filename
			)

			output_name = (
				output_filename(**params)
				if callable(output_filename)
				else output_filename
			)
			input_path = output_dir / input_name
			output_path = output_dir / output_name
			model_path = output_dir / model_filename if model_filename is not None else None

			if not input_path.is_file():
				raise FileNotFoundError(
					f"Archive does not exist: {input_path}"
				)

			if output_path.exists() and not overwrite:
				raise FileExistsError(
					f"{output_path} already exists; "
					"pass overwrite=True to replace it"
				)

			started_at = perf_counter()

			with np.load(input_path, allow_pickle=False) as source:
				if accepts_output_dir:
					taskresult = func(
						source,
						*args,
						output_dir=output_dir,
						**kwargs,
					)
				else:
					taskresult = func(source, *args, **kwargs)

			logging.info("Saving results")
			

			atomic_write(output_path, taskresult.results_file_saver, ".tmp.npz")
			if (taskresult.model_file_saver is not None and model_path is not None):
				atomic_write(model_path, taskresult.model_file_saver, ".tmp.joblib")

			taskresult.timer_log(perf_counter() - started_at)

			if (callable(taskresult.result)):
				return taskresult.result(output_path)
			return taskresult.result
		setattr(wrapper, "compute", func)
		return cast(DataFromFileCallable[..., R], wrapper)
	return decorator