"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by the worked-example
`generate_sql_node` in graph.py via `.format(schema=..., question=...)`, so
keep those placeholders intact. The VERIFY_* and REVISE_* prompts are yours to
design alongside their nodes - pick whatever placeholders your nodes pass in.

Filling these in is part of Phase 3.
"""

GENERATE_SQL_SYSTEM = """\
You write correct, executable SQLite SELECT queries.
Rules:
- Use only tables/columns in the provided schema; never invent names.
- SQLite dialect; double-quote reserved-word or spaced identifiers (e.g. "order").
- Text filters: compare case-insensitively, because stored capitalization rarely
  matches the question's wording and a wrong case returns zero rows. Use
  `"col" = 'value' COLLATE NOCASE`.
- SELECT exactly the column(s) the question asks for, in the order asked - nothing
  extra. "List the names" -> just the name column; "how many" -> a single COUNT;
  a which-one / yes-no question -> just that one answer column.
- Add only the filters the question states; don't invent extra conditions.
- Reach a value that lives in another table by joining on the foreign key, rather
  than guessing it exists on the current table.
- Match a specific date/time with LIKE 'prefix%' (e.g. WHERE "d" LIKE '2010-07-19%'),
  not '=', because stored timestamps may carry a trailing '.0'.
- Use SELECT DISTINCT when a join can duplicate rows but the question asks for
  distinct entities or values.

The examples below use a MADE-UP schema, only to show the style - your real schema
and question follow afterward.

  Schema (illustrative):
    CREATE TABLE "movie" ("id" INTEGER PRIMARY KEY, "title" TEXT, "studio_id" INTEGER,
      FOREIGN KEY ("studio_id") REFERENCES "studio"("id"));
    CREATE TABLE "studio" ("id" INTEGER PRIMARY KEY, "name" TEXT, "country" TEXT);

  Q: List the titles of movies made by the 'Pixar' studio.
  A: SELECT m."title" FROM "movie" m JOIN "studio" s ON m."studio_id" = s."id"
     WHERE s."name" = 'Pixar' COLLATE NOCASE;
     -- only the asked column (title); filter value matched case-insensitively;
     -- studio name reached via the FK join.

  Q: How many studios are based in the 'usa'?
  A: SELECT COUNT(*) FROM "studio" WHERE "country" = 'usa' COLLATE NOCASE;
     -- a single COUNT; case-insensitive match means 'USA'/'usa'/'Usa' all count.

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
question implies at least one (often a text filter that should be case-insensitive,
e.g. needs COLLATE NOCASE); the SELECT returns extra or wrong columns instead of
exactly what was asked (e.g. an id instead of a name, or many columns when one was
asked); or a stated filter/order/limit is ignored.
Be pragmatic - accept a reasonable answer even if you'd phrase the SQL differently.
Reply with ONLY a compact JSON object, no prose or fences: {"ok": true} if
acceptable, else {"ok": false, "issue": "<one short sentence describing the problem>"}."""

# Available placeholders: {question}, {sql}, {result}
VERIFY_USER = """\
Question: {question}

SQL:
{sql}

Result:
{result}

Verdict (JSON):"""


REVISE_SYSTEM = """\
You fix a SQLite query that failed review, given the schema, question, previous
query, its result, and the reviewer's complaint.
Rules:
- Use only schema tables/columns; SQLite dialect.
- Fix the specific complaint; do not rewrite gratuitously.
- If the result was empty, suspect a text filter and make it case-insensitive
  with COLLATE NOCASE.
- Use only columns that exist in the schema; do not invent column names.
- Return only the column(s) the question asks for, nothing extra.
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
