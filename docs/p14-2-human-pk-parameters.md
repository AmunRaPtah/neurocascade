# P14.2: human PK parameter status

The human PK registry now names the three required drugs and keeps their
species/route provenance explicit. Human fentanyl is documented against the
canonical three-compartment sources (Shafer 1990; Scott & Stanski 1987;
McClain & Hug 1980; Bae 2020), but no values are executable until one
consistent parameter set is transcribed and validated.

Haloperidol and diazepam likewise remain explicit searched/source gaps in the
human registry. No rodent values are copied into human records. `None` means
missing evidence, not zero. This is intentionally an intermediate P14.2 result;
P14.4 must validate the eventual parameter sets.

## Superseded in part by P19.2

As written, this page said that "every record reports
`complete_for_model=False` until all required disposition parameters are
verified". That is no longer true. P19.2 transcribed an oral, apparent (X/F)
two-compartment haloperidol set from Pilla Reddy 2013 (PMID 24113674), and that
record reports `complete_for_model=True` for its route.

Three qualifications carry forward, and `docs/p19-2-human-haloperidol-pk.md`
has the full account:

- The completeness is route-specific. The haloperidol **intravenous** record is
  still a recorded gap, because no readable human IV micro-parameter set was
  reachable. Fentanyl and diazepam are unchanged and still incomplete.
- The set is apparent, so it names no absolute volume or clearance. F was never
  identified by the source, and the record refuses to borrow one from another
  study.
- Complete does not mean validated. No prediction from this set has yet been
  compared to a measured human observation; P19.6 is the step that can falsify
  it.
