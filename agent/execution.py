"""SQL execution helper (provided complete).

execute_sql() runs the agent's SQL against the target DB in read-only mode
and returns a structured ExecutionResult. The verify node consumes this
to decide whether the answer looks plausible.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from agent.schema import db_path


@dataclass
class ExecutionResult:
    ok: bool
    rows: list[tuple] | None = None
    columns: list[str] | None = None
    error: str | None = None
    row_count: int = 0

    def render(self, max_rows: int = 10, max_cell: int = 200) -> str:
        """Compact text rendering for prompt context.

        Phase 6 / iteration 3 (prompt & KV reduction): besides the 10-row cap,
        each cell is now truncated to ``max_cell`` chars. The row cap alone left
        cell *width* unbounded, so a single wide TEXT column (e.g. a post body or
        a card's text) could push a verify/revise prompt to thousands of tokens.
        The verifier only needs to see the *shape* of the answer (columns, a few
        representative values), not full blobs, so this is a pure token win.
        """
        if not self.ok:
            return f"ERROR: {self.error}"
        if self.row_count == 0:
            return "OK: 0 rows returned."

        def cell(c: object) -> str:
            s = str(c)
            return s if len(s) <= max_cell else s[:max_cell] + f"…(+{len(s) - max_cell} chars)"

        cols = ", ".join(self.columns or [])
        preview = "\n".join(
            " | ".join(cell(c) for c in row) for row in (self.rows or [])[:max_rows]
        )
        more = f"\n... ({self.row_count - max_rows} more rows)" if self.row_count > max_rows else ""
        return f"OK: {self.row_count} rows.\nCOLUMNS: {cols}\nFIRST ROWS:\n{preview}{more}"


def execute_sql(db_id: str, sql: str, timeout_seconds: float = 5.0) -> ExecutionResult:
    """Run SQL against db_id's sqlite, return result or error."""
    path = db_path(db_id)
    try:
        with sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            timeout=timeout_seconds,
        ) as conn:
            cur = conn.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
            return ExecutionResult(ok=True, rows=rows, columns=cols, row_count=len(rows))
    except Exception as e:  # noqa: BLE001
        return ExecutionResult(ok=False, error=f"{type(e).__name__}: {e}")
