-- Claims Ledger schema.

CREATE TABLE IF NOT EXISTS claim (
    id              TEXT PRIMARY KEY,
    category        TEXT NOT NULL CHECK (
        category IN ('experience', 'skill', 'education', 'metric')
    ),
    -- Most important field: the actual claim text. Can't be null because it's the basis of the entire ledger.
    text            TEXT NOT NULL,
    -- The job title and company this claim came from. Nullable because
    -- skills and education claims don't always map to a single role.
    source_role     TEXT,
    source_company  TEXT,
    -- SQLite has no BOOLEAN type; INTEGER 0/1 is the convention.
    has_metric      INTEGER NOT NULL DEFAULT 0 CHECK (has_metric IN (0, 1)),
    -- Verbatim copy of the original claim text as it appeared in the resume.
    raw_original    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_claim_category    ON claim(category);
CREATE INDEX IF NOT EXISTS idx_claim_source_role ON claim(source_role);
