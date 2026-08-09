# Amendment E7-01: operational definition of last clear acquisition

Attempts 1--2 were implementation failures, not passing experiments.  Attempt 1
used a thread-unsafe lazy NPZ reader.  Attempt 2 fixed that but interpreted the
cache's last minimally usable observation (only 20 clear pixels required) as the
plan's "last clear" image, leaving roughly 65/316 challenge fields.  Its report
also failed JSON serialization on a NumPy boolean.

Attempt 3 searches backward, still strictly within `GDD <= look3`, and selects
the most recent acquisition with at least 50 SCL-4/5 pixels having valid yield
support.  This is an eligibility clarification, not outcome-dependent image
selection: only support/quality masks determine the date, never yield magnitude.
Vegetation-index division uses finite masked arithmetic.  Models, samples per
training field, endpoints, qualitative-ID hashes, and all pass gates are
unchanged.

