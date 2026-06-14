"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Only the *_USER templates are passed through str.format(), so they must contain
only their intended {placeholders}. The *_SYSTEM templates are sent verbatim
(never .format()'d), so they may contain literal braces safely.

Phase 6 / iteration 6: these were tightened for concision - every functional
rule kept, just fewer words. The system prompts sit in the cached prefix so the
latency effect is marginal; the win is hygiene and a slightly smaller prefill on
cache misses. Validated against the Phase 5 eval (no accuracy change).
"""

GENERATE_SQL_SYSTEM = """\
You write correct, executable SQLite SELECT queries.
Rules:
- Use only tables/columns in the provided schema; never invent names.
- SQLite dialect; double-quote reserved-word or spaced identifiers (e.g. "order").
- Return ONE SELECT that answers the question - SQL only, no prose, comments, or fences."""

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = """\
Schema:
{schema}

Question: {question}

Write one SQLite SELECT that answers it."""


VERIFY_SYSTEM = """\
You review a text-to-SQL result: given the question, the SQL, and its execution
result, decide whether the result correctly answers the question.
Reject when, for example: the result starts with ERROR; zero rows where the
question implies at least one; the columns don't answer what was asked (e.g. an
id instead of a name); or a stated filter/order/limit is ignored.
Be pragmatic - accept a reasonable answer even if you'd phrase the SQL differently.
Reply with EXACTLY one of (no JSON, prose, or fences): the bare token OK if
acceptable, else BAD: <one short sentence describing the problem>."""

# Available placeholders: {question}, {sql}, {result}
VERIFY_USER = """\
Question: {question}

SQL:
{sql}

Result:
{result}

Verdict (OK or BAD: ...):"""


REVISE_SYSTEM = """\
You fix a SQLite query that failed review, given the schema, question, previous
query, its result, and the reviewer's complaint.
Rules:
- Use only schema tables/columns; SQLite dialect.
- Fix the specific complaint; do not rewrite gratuitously.
- Return ONLY the corrected SELECT - no prose or fences."""

# Available placeholders: {schema}, {question}, {sql}, {result}, {issue}
REVISE_USER = """\
Schema:
{schema}

Question: {question}

Previous SQL:
{sql}

Its result:
{result}

Complaint: {issue}

Return a corrected SQLite SELECT that fixes it."""
