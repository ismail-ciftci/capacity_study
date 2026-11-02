"""SQLite job queue shared by all worker processes of a study (WAL mode, BEGIN IMMEDIATE claims).

The ledger is a scheduling index. Completion truth is the job's DONE.json; `reconcile` re-derives states from the
filesystem and returns stale `running` jobs (no heartbeat within the timeout) to `pending`. Every state change by a
worker is conditioned on the worker still owning the job, so a reclaimed job cannot be overwritten by its old owner.
"""
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from capacity_study.errors import JobStateError
from capacity_study.recipes import JobSpec
from capacity_study.store import StudyPaths

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id      TEXT PRIMARY KEY,
    phase       TEXT NOT NULL,
    spec        TEXT NOT NULL,
    status      TEXT NOT NULL CHECK (status IN ('pending', 'running', 'done', 'failed')),
    depends_on  TEXT,
    cost        REAL NOT NULL,
    est_mem_mb  REAL NOT NULL,
    device      TEXT,
    worker      TEXT,
    heartbeat   REAL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    updated     REAL NOT NULL
);
"""


@dataclass(frozen=True)
class ClaimedJob:
    spec: JobSpec
    attempts: int


class Ledger:
    def __init__(self, path: Path):
        self.path = path
        self.conn = sqlite3.connect(path, timeout=60.0, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def add_jobs(self, entries: list[tuple[JobSpec, float, float, str | None]]) -> int:
        """entries: (spec, cost, est_mem_mb, depends_on). Existing job_ids are left untouched. Returns # added."""
        now = time.time()
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            before = self.conn.total_changes
            self.conn.executemany(
                "INSERT OR IGNORE INTO jobs (job_id, phase, spec, status, depends_on, cost, est_mem_mb, updated) "
                "VALUES (?, ?, ?, 'pending', ?, ?, ?, ?)",
                [(s.job_id, s.phase, json.dumps(s.to_dict()), dep, cost, mem, now) for s, cost, mem, dep in entries])
            added = self.conn.total_changes - before
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        return added

    def claim(self, worker: str, device: str, mem_budget_mb: float) -> ClaimedJob | None:
        """Highest-cost ready job that fits the device's remaining memory budget. If nothing runs on the device,
        the smallest ready job is taken even if its estimate exceeds the budget (it then fails loudly on OOM)."""
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            used = self.conn.execute("SELECT COALESCE(SUM(est_mem_mb), 0), COUNT(*) FROM jobs "
                                     "WHERE status = 'running' AND device = ?", (device,)).fetchone()
            ready = self.conn.execute(
                "SELECT j.job_id, j.spec, j.est_mem_mb, j.attempts FROM jobs j "
                "LEFT JOIN jobs d ON j.depends_on = d.job_id "
                "WHERE j.status = 'pending' AND (j.depends_on IS NULL OR d.status = 'done') "
                "ORDER BY j.cost DESC, j.job_id").fetchall()
            pick = next((r for r in ready if used[0] + r[2] <= mem_budget_mb), None)
            if pick is None and used[1] == 0 and ready:
                pick = min(ready, key=lambda r: (r[2], r[0]))
            if pick is None:
                self.conn.execute("COMMIT")
                return None
            self.conn.execute("UPDATE jobs SET status = 'running', device = ?, worker = ?, heartbeat = ?, "
                              "attempts = attempts + 1, error = NULL, updated = ? WHERE job_id = ?",
                              (device, worker, time.time(), time.time(), pick[0]))
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        return ClaimedJob(JobSpec.from_dict(json.loads(pick[1])), pick[3] + 1)

    def _owned_update(self, job_id: str, worker: str, operation: str, sql: str, args: tuple) -> None:
        cur = self.conn.execute(sql + " WHERE job_id = ? AND worker = ? AND status = 'running'",
                                (*args, job_id, worker))
        if cur.rowcount != 1:
            raise JobStateError(job_id, operation, f"worker {worker} no longer owns this job")

    def heartbeat(self, job_id: str, worker: str) -> None:
        self._owned_update(job_id, worker, "heartbeat", "UPDATE jobs SET heartbeat = ?", (time.time(),))

    def release(self, job_id: str, worker: str) -> None:
        self._owned_update(job_id, worker, "release",
                           "UPDATE jobs SET status = 'pending', worker = NULL, device = NULL, updated = ?",
                           (time.time(),))

    def complete(self, job_id: str, worker: str) -> None:
        self._owned_update(job_id, worker, "complete", "UPDATE jobs SET status = 'done', updated = ?",
                           (time.time(),))

    def fail(self, job_id: str, worker: str, error: str) -> None:
        self._owned_update(job_id, worker, "fail", "UPDATE jobs SET status = 'failed', error = ?, updated = ?",
                           (error, time.time()))

    def retry(self, job_id: str) -> None:
        cur = self.conn.execute("UPDATE jobs SET status = 'pending', worker = NULL, device = NULL, updated = ? "
                                "WHERE job_id = ? AND status = 'failed'", (time.time(), job_id))
        if cur.rowcount != 1:
            raise JobStateError(job_id, "retry", "job is not in state 'failed'")

    def reconcile(self, paths: StudyPaths, stale_seconds: float) -> dict[str, int]:
        """DONE.json on disk wins; running jobs without a recent heartbeat go back to pending."""
        now = time.time()
        fixed = {"marked_done": 0, "reclaimed_stale": 0, "done_without_marker": 0}
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            for job_id, status, hb in self.conn.execute("SELECT job_id, status, heartbeat FROM jobs").fetchall():
                has_done = paths.job(job_id).done.exists()
                if has_done and status != "done":
                    self.conn.execute("UPDATE jobs SET status = 'done', updated = ? WHERE job_id = ?",
                                      (now, job_id))
                    fixed["marked_done"] += 1
                elif not has_done and status == "done":
                    self.conn.execute("UPDATE jobs SET status = 'pending', worker = NULL, device = NULL, "
                                      "updated = ? WHERE job_id = ?", (now, job_id))
                    fixed["done_without_marker"] += 1
                elif status == "running" and (hb is None or now - hb > stale_seconds):
                    self.conn.execute("UPDATE jobs SET status = 'pending', worker = NULL, device = NULL, "
                                      "updated = ? WHERE job_id = ?", (now, job_id))
                    fixed["reclaimed_stale"] += 1
            self.conn.execute("COMMIT")
        except BaseException:
            self.conn.execute("ROLLBACK")
            raise
        return fixed

    def status_counts(self) -> list[tuple[str, str, int]]:
        return self.conn.execute("SELECT phase, status, COUNT(*) FROM jobs GROUP BY phase, status "
                                 "ORDER BY phase, status").fetchall()

    def failures(self) -> list[tuple[str, str]]:
        return self.conn.execute("SELECT job_id, error FROM jobs WHERE status = 'failed' ORDER BY job_id").fetchall()

    def running(self) -> list[tuple[str, str, str, float]]:
        return self.conn.execute("SELECT job_id, device, worker, heartbeat FROM jobs WHERE status = 'running' "
                                 "ORDER BY job_id").fetchall()
