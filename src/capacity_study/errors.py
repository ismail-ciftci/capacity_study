"""Typed exceptions. Each carries the identifiers and the operation that failed."""


class CapacityStudyError(Exception):
    """Base class for all study errors."""


class ConfigError(CapacityStudyError):
    def __init__(self, source: str, key: str, problem: str):
        self.source, self.key, self.problem = source, key, problem
        super().__init__(f"config {source!r}: key {key!r}: {problem}")


class DataCoverageError(CapacityStudyError):
    def __init__(self, operation: str, tickers: list[str], detail: str):
        self.operation, self.tickers, self.detail = operation, tickers, detail
        super().__init__(f"{operation}: insufficient data for {tickers}: {detail}")


class DataIntegrityError(CapacityStudyError):
    def __init__(self, path: str, expected: str, actual: str):
        self.path, self.expected, self.actual = path, expected, actual
        super().__init__(f"{path}: sha256 mismatch (expected {expected}, found {actual})")


class OverwriteRefusedError(CapacityStudyError):
    def __init__(self, path: str, operation: str):
        self.path, self.operation = path, operation
        super().__init__(f"{operation}: refusing to overwrite existing {path}")


class ProvenanceError(CapacityStudyError):
    def __init__(self, operation: str, detail: str):
        self.operation, self.detail = operation, detail
        super().__init__(f"{operation}: {detail}")


class DirtyTreeError(ProvenanceError):
    def __init__(self, operation: str, changed: list[str]):
        self.changed = changed
        super().__init__(operation, f"git tree has uncommitted changes ({len(changed)} paths); "
                                    f"commit them or pass --allow-dirty")


class CheckpointMismatchError(CapacityStudyError):
    def __init__(self, job_id: str, field: str, checkpoint_value: object, current_value: object):
        self.job_id, self.field = job_id, field
        self.checkpoint_value, self.current_value = checkpoint_value, current_value
        super().__init__(f"job {job_id}: checkpoint {field}={checkpoint_value!r} "
                         f"but current {field}={current_value!r}")


class JobStateError(CapacityStudyError):
    def __init__(self, job_id: str, operation: str, detail: str):
        self.job_id, self.operation, self.detail = job_id, operation, detail
        super().__init__(f"job {job_id}: {operation}: {detail}")


class StudyStateError(CapacityStudyError):
    def __init__(self, study: str, operation: str, detail: str):
        self.study, self.operation, self.detail = study, operation, detail
        super().__init__(f"study {study}: {operation}: {detail}")
