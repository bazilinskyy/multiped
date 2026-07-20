"""Single-file cache for processed human-experiment data.

The cache contains all participant-derived inputs required by the analysis:
questionnaires, trial ratings, slider tables, trigger matrices, HMD quaternion
matrices, and derived horizontal head-heading series. It intentionally excludes
figures because those should be regenerated from the cached data.
"""

from __future__ import annotations

import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from utils.HMD_helper import HMD_yaw
from utils.tools import Tools


CACHE_SCHEMA_VERSION = 2
REQUIRED_CACHE_KEYS = {
    "schema_version",
    "created_utc",
    "settings",
    "mapping",
    "intake_questionnaire",
    "post_questionnaire",
    "trial_ratings",
    "slider_tables",
    "trigger_matrices",
    "quaternion_matrices",
    "head_heading_average",
    "head_heading_bins",
    "head_heading_values",
    "statistical_tables",
}


class ProcessedExperimentCache:
    """Build, validate, save, load, and restore one processed-data pickle."""

    def __init__(self, cache_path: str, output_dir: str):
        self.cache_path = Path(cache_path).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()

    def delete_existing(self) -> bool:
        """Delete only the configured pickle file, if present."""
        if self.cache_path.exists():
            if not self.cache_path.is_file():
                raise ValueError(f"Processed cache path is not a file: {self.cache_path}")
            self.cache_path.unlink()
            return True
        return False

    @staticmethod
    def _read_questionnaire(source) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return source.copy()
        return pd.read_csv(source)

    @staticmethod
    def _safe_video_ids(mapping: pd.DataFrame):
        if "video_id" not in mapping.columns:
            raise ValueError("mapping must contain video_id")
        return list(dict.fromkeys(mapping["video_id"].dropna().astype(str).tolist()))

    def build(
        self,
        helper,
        mapping: pd.DataFrame,
        data_folder: str,
        intake_questionnaire,
        post_questionnaire,
        settings: Dict[str, Any],
        response_col_index: int = 2,
        n_participants: int = 50,
    ) -> Dict[str, Any]:
        """Reprocess raw human data and return a complete cache payload."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        mapping = mapping.copy()
        mapping["video_id"] = mapping["video_id"].astype(str)

        intake_df = self._read_questionnaire(intake_questionnaire)
        post_df = self._read_questionnaire(post_questionnaire)
        trial_ratings = helper.load_trial_ratings(
            responses_root=data_folder,
            n_participants=n_participants,
            response_col_index=response_col_index,
        )

        # Rebuild the wide slider tables from raw participant responses.
        helper.read_slider_data(data_folder, str(self.output_dir))
        slider_tables = {}
        for name in ("behaviour", "distance", "intention"):
            path = self.output_dir / f"slider_input_{name}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Expected slider table was not created: {path}")
            slider_tables[name] = pd.read_csv(path)

        trigger_matrices = {}
        quaternion_matrices = {}
        head_heading_average = {}
        head_heading_bins = {}
        head_heading_values = {}
        heading_converter = HMD_yaw()
        tools = Tools()

        for video_id in self._safe_video_ids(mapping):
            trigger_path = self.output_dir / f"participant_TriggerValueRight_{video_id}.csv"
            helper.export_participant_trigger_matrix(
                data_folder=data_folder,
                video_id=video_id,
                output_file=str(trigger_path),
                column_name="TriggerValueRight",
                mapping=mapping,
                overwrite=True,
            )
            if not trigger_path.is_file():
                raise FileNotFoundError(f"Trigger matrix was not created: {trigger_path}")
            trigger_matrices[video_id] = pd.read_csv(trigger_path)

            quaternion_path = self.output_dir / f"participant_Yaw_{video_id}.csv"
            helper.export_participant_quaternion_matrix(
                data_folder=data_folder,
                video_id=video_id,
                output_file=str(quaternion_path),
                mapping=mapping,
                overwrite=True,
            )
            if not quaternion_path.is_file():
                raise FileNotFoundError(f"Quaternion matrix was not created: {quaternion_path}")
            quaternion_df = pd.read_csv(quaternion_path)
            quaternion_matrices[video_id] = quaternion_df

            heading_path = self.output_dir / f"yaw_avg_{video_id}.csv"
            head_heading_average[video_id] = heading_converter.compute_avg_yaw_from_matrix_csv(
                input_csv=str(quaternion_path),
                output_csv=str(heading_path),
                force=True,
            )
            heading_bins = tools.all_yaws_per_bin_from_dataframe(quaternion_df)
            head_heading_bins[video_id] = heading_bins
            headings = tools.flatten_trial_matrix(heading_bins)
            head_heading_values[video_id] = headings[np.isfinite(headings)]

        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "settings": dict(settings),
            "mapping": mapping,
            "intake_questionnaire": intake_df,
            "post_questionnaire": post_df,
            "trial_ratings": trial_ratings,
            "slider_tables": slider_tables,
            "trigger_matrices": trigger_matrices,
            "quaternion_matrices": quaternion_matrices,
            "head_heading_average": head_heading_average,
            "head_heading_bins": head_heading_bins,
            "head_heading_values": head_heading_values,
            "statistical_tables": {},
        }
        self.validate(payload, expected_settings=settings)
        return payload

    @staticmethod
    def validate(
        payload: Dict[str, Any],
        expected_settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Reject incomplete, incompatible, or partial cache payloads."""
        if not isinstance(payload, dict):
            raise TypeError("Processed experiment cache must contain a dictionary")
        missing = sorted(REQUIRED_CACHE_KEYS.difference(payload))
        if missing:
            raise ValueError(f"Processed experiment cache is incomplete; missing: {missing}")
        if payload["schema_version"] != CACHE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported cache schema {payload['schema_version']}; "
                f"expected {CACHE_SCHEMA_VERSION}. Set always_analyse to true to rebuild it."
            )

        mapping = payload["mapping"]
        if not isinstance(mapping, pd.DataFrame) or mapping.empty:
            raise ValueError("Cached mapping is missing or empty")
        if "video_id" not in mapping.columns:
            raise ValueError("Cached mapping is missing video_id")

        for key in ("intake_questionnaire", "post_questionnaire", "trial_ratings"):
            if not isinstance(payload[key], pd.DataFrame):
                raise TypeError(f"Cache section {key} must be a pandas DataFrame")
        required_rating_columns = {"participant", "video_id", "Q1", "Q2", "Q3"}
        missing_rating_columns = sorted(
            required_rating_columns.difference(payload["trial_ratings"].columns)
        )
        if missing_rating_columns:
            raise ValueError(
                "Cached trial ratings are incomplete; missing columns: "
                f"{missing_rating_columns}"
            )

        slider_tables = payload["slider_tables"]
        missing_slider_tables = sorted(
            {"behaviour", "distance", "intention"}.difference(slider_tables)
        )
        if missing_slider_tables:
            raise ValueError(
                f"Processed cache is missing slider tables: {missing_slider_tables}"
            )
        if any(not isinstance(table, pd.DataFrame) for table in slider_tables.values()):
            raise TypeError("Every cached slider table must be a pandas DataFrame")

        video_ids = set(mapping["video_id"].dropna().astype(str))
        for key in (
            "trigger_matrices",
            "quaternion_matrices",
            "head_heading_average",
            "head_heading_bins",
            "head_heading_values",
        ):
            missing_videos = sorted(video_ids.difference(payload[key]))
            if missing_videos:
                raise ValueError(f"Cache section {key} is missing videos: {missing_videos}")
        for key in ("trigger_matrices", "quaternion_matrices", "head_heading_average"):
            if any(not isinstance(table, pd.DataFrame) for table in payload[key].values()):
                raise TypeError(f"Every entry in cache section {key} must be a DataFrame")
        if not isinstance(payload["statistical_tables"], dict):
            raise TypeError("Cache section statistical_tables must be a dictionary")
        if any(
            not isinstance(table, pd.DataFrame)
            for table in payload["statistical_tables"].values()
        ):
            raise TypeError("Every cached statistical table must be a DataFrame")

        if expected_settings is not None:
            cached_settings = payload.get("settings", {})
            mismatches = {
                key: (cached_settings.get(key), value)
                for key, value in expected_settings.items()
                if cached_settings.get(key) != value
            }
            if mismatches:
                raise ValueError(
                    "Processed cache settings do not match the current configuration: "
                    f"{mismatches}. Set always_analyse to true to rebuild it."
                )

    def save(self, payload: Dict[str, Any]) -> None:
        """Atomically replace the configured pickle after successful processing."""
        self.validate(payload)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        try:
            with temporary_path.open("wb") as stream:
                pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, self.cache_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def load(self, expected_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Load only the configured pickle; never fall back to raw data."""
        if not self.cache_path.is_file():
            raise FileNotFoundError(
                f"Processed cache not found: {self.cache_path}. "
                "Set always_analyse to true for one run to create it."
            )
        with self.cache_path.open("rb") as stream:
            payload = pickle.load(stream)
        upgraded = self._upgrade_payload(payload)
        self.validate(payload, expected_settings=expected_settings)
        if upgraded:
            self.save(payload)
        return payload

    @staticmethod
    def _upgrade_payload(payload: Dict[str, Any]) -> bool:
        """Upgrade the previous cache using data already stored in the pickle."""
        if not isinstance(payload, dict):
            return False
        version = payload.get("schema_version")
        if version == CACHE_SCHEMA_VERSION:
            return False
        if version != 1:
            return False

        quaternion_matrices = payload.get("quaternion_matrices")
        if not isinstance(quaternion_matrices, dict):
            return False

        tools = Tools()
        payload["head_heading_bins"] = {
            str(video_id): tools.all_yaws_per_bin_from_dataframe(dataframe)
            for video_id, dataframe in quaternion_matrices.items()
        }
        payload.setdefault("statistical_tables", {})
        payload["schema_version"] = CACHE_SCHEMA_VERSION
        return True

    def load_or_build(
        self,
        builder: Callable[[], Dict[str, Any]],
        expected_settings: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], bool]:
        """Load the pickle, or build and save it when it does not exist.

        Returns the payload and a flag indicating whether raw processing was
        required. Validation errors from an existing pickle are deliberately
        not caught: an incompatible or corrupt cache must not be mistaken for
        an absent first-run cache.
        """
        try:
            return self.load(expected_settings=expected_settings), False
        except FileNotFoundError:
            payload = builder()
            self.save(payload)
            return payload, True

    @staticmethod
    def _write_dataframe(path: Path, dataframe: pd.DataFrame) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_suffix(path.suffix + ".tmp")
        try:
            dataframe.to_csv(temporary_path, index=False)
            os.replace(temporary_path, path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def restore_analysis_inputs(self, payload: Dict[str, Any]) -> None:
        """Restore the compatibility CSVs used by existing plotting functions."""
        self.validate(payload)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for name, dataframe in payload["slider_tables"].items():
            self._write_dataframe(
                self.output_dir / f"slider_input_{name}.csv",
                dataframe,
            )
        for video_id, dataframe in payload["trigger_matrices"].items():
            self._write_dataframe(
                self.output_dir / f"participant_TriggerValueRight_{video_id}.csv",
                dataframe,
            )
        for video_id, dataframe in payload["quaternion_matrices"].items():
            self._write_dataframe(
                self.output_dir / f"participant_Yaw_{video_id}.csv",
                dataframe,
            )
        for video_id, dataframe in payload["head_heading_average"].items():
            self._write_dataframe(
                self.output_dir / f"yaw_avg_{video_id}.csv",
                dataframe,
            )
        for video_id, values in payload["head_heading_values"].items():
            path = self.output_dir / f"yaw_values_{video_id}.txt"
            temporary_path = path.with_suffix(path.suffix + ".tmp")
            try:
                np.savetxt(temporary_path, np.asarray(values, dtype=float))
                os.replace(temporary_path, path)
            finally:
                if temporary_path.exists():
                    temporary_path.unlink()

        for relative_name, dataframe in payload["statistical_tables"].items():
            relative_path = Path(relative_name)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"Unsafe statistical-table path in processed cache: {relative_name}"
                )
            self._write_dataframe(self.output_dir / relative_path, dataframe)

    def capture_statistical_tables(self, payload: Dict[str, Any]) -> int:
        """Store generated statistics CSVs in the processed-data payload."""
        statistics_dir = self.output_dir / "statistics"
        tables = {}
        if statistics_dir.is_dir():
            for path in sorted(statistics_dir.rglob("*.csv")):
                relative_name = str(path.relative_to(self.output_dir))
                tables[relative_name] = pd.read_csv(path)
        payload["statistical_tables"] = tables
        self.validate(payload)
        return len(tables)
