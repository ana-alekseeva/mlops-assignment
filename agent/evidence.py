"""Per-database evidence / data-dictionary notes.

The eval set gives only (question, db_id, gold_sql) - it strips BIRD's
"evidence" field (the external-knowledge hint that maps a vague phrase in the
question to the concrete column / value / formula it means). Without that, the
model cannot know that "crimes in 1995" is column A15, that a "normal" IgG is
900-2000, or that the carcinogenic label is stored as '+'. These are exactly the
domain facts a real text-to-SQL system keeps in a data dictionary, so we supply a
compact, hand-written one per database.

Notes are documentation of column meanings / stored value forms / domain ranges -
NOT answers to specific questions. They are injected into the generate and revise
prompts (and, being constant per DB, ride in the cached prefix).
"""
from __future__ import annotations

EVIDENCE: dict[str, str] = {
    "financial": """\
- district.A2 = district name; A3 = region.
- district.A11 = average salary; A12 / A13 = unemployment rate for 1995 / 1996.
- district.A15 = number of crimes committed in 1995; A16 = in 1996 (A14 is NOT crimes).
- Date columns are TEXT 'YYYY-MM-DD'; get the year with strftime('%Y', date).
- gender is 'M' / 'F'.""",

    "thrombosis_prediction": """\
- "Normal" lab ranges: Ig G is Laboratory.IGG normal 900-2000; UA (uric acid) is
  normal when < 6.5 for female (SEX='F') and < 8.0 for male (SEX='M'); T-BIL
  (total bilirubin) is normal when < 2.0.
- Patient.Admission = '+' means inpatient; '-' means followed up at the OUTPATIENT clinic.
- "latest / most recent examination" = the row with MAX(Date) in Laboratory.
- A patient "has symptoms" when Examination.Symptoms IS NOT NULL.
- Join Patient/Laboratory/Examination on ID.""",

    "toxicology": """\
- molecule.label = '+' means carcinogenic, '-' means non-carcinogenic.
- atom.element values are lowercase chemical symbols: Chlorine='cl', Calcium='ca',
  Hydrogen='h', Oxygen='o', Sulfur='s', etc.""",

    "superhero": """\
- "No colour" (e.g. no eye colour) is a row in the colour table whose colour =
  'No Color' - it is NOT a NULL eye_colour_id.
- A numeric attribute is "missing" when it is 0 OR NULL (e.g. weight_kg = 0 OR weight_kg IS NULL).""",

    "california_schools": """\
- "excellence rate" = satscores.NumGE1500 * 1.0 / satscores.NumTstTakr.
- NCES school identification number = schools.NCESSchool (NCESDist is the district number).
- "Enrollment (Ages 5-17)" is a column in the frpm table (join schools on CDSCode).
- An "active" school/district has schools.StatusType = 'Active'.""",

    "formula_1": """\
- "race number / race no." = results.raceId (not races.round).
- "finishers" = rows where results.time IS NOT NULL; "disqualified" = statusId = 2.
- A Grand Prix name like 'Australian Grand Prix' is races.name; circuits.name is the
  circuit's own name. Use SELECT DISTINCT for a circuit's lat/lng.
- Lap and fastestLapTime values are TEXT 'M:SS.mmm'; convert to seconds as
  minutes*60 + seconds, parsing around the ':'.""",

    "codebase_community": """\
- A post is "well-finished" when posts.ClosedDate IS NOT NULL; if ClosedDate IS NULL
  it is NOT well-finished.
- "popularity" of a post = posts.ViewCount.
- A user's posts can be reached via postHistory (postHistory.UserId -> posts via PostId).
- Date/datetime columns are TEXT with a trailing '.0' (e.g. '2010-07-19 19:39:08.0');
  match a specific timestamp with LIKE 'YYYY-MM-DD HH:MM:SS%'.""",

    "student_club": """\
- major.major_name is the major (e.g. 'Business'); major.department is the department,
  and its values end in "Department" (e.g. 'Art and Design Department').
- Amount spent is budget.spent; an event's year = SUBSTR(event.event_date, 1, 4).""",

    "card_games": """\
- legalities.status is capitalized ('Legal', 'Banned', 'Restricted'); legalities.format
  is lowercase ('gladiator', 'commander', ...). Case-insensitive matching handles this.""",
}


def get_evidence(db_id: str) -> str:
    """Return a prompt-ready notes block for db_id, or '' if we have none."""
    notes = EVIDENCE.get(db_id)
    return f"\n\nNotes for this database (use them):\n{notes}" if notes else ""
