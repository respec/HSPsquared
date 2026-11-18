# test05start30: a run that does not start on the hour, at a 30-minute step

Regression case for item S1 of issue #232 in SNOW (see `tests/test05start/README.md`
for the item). In a run that does not start on the hour the hour flag was off in the
first interval. SNOW then did not compute the compaction factor of the pack there; its
start value, 0, set the depth of the pack to 0, and the density of the pack was then
divided by that depth: a snow run with an initial pack stopped with
`ZeroDivisionError`.

## The case

HSPF test05 (`tests/test05`, 30-minute step, PERLND with SNOW and PWATER, initial
pack) with these changes, made by `tests/test05start/make_cases.py` from
`tests/test05/HSPFresults/TEST05bin.UCI`:

- Run from 1976-01-02 00:30 to 1976-01-31 24:00 (test05: the whole of 1976, from
  00:00).
- ICE-FLAG: ICEFG 0 (test05: 1), so that the ground-ice items of a sub-hourly step (F2
  and P1 of issue #232) do not enter this case.
- DISPLY dropped (HSP2 does not run it; the gfortran build of HSPF stops in its annual
  summary).
- Binary output: SNOW of the three segments at every interval (flag 2, PIVL 1). PWATER
  is not written: at a sub-hourly step its PETADJ output is item P1 of issue #232.

`HSPFresults/test05start30.wdm` is `tests/test05/HSPFresults/TEST.WDM`, unchanged. The
reference output was made as described in `tests/test05start/README.md`.

## Expected result

`tests/test_regression.py` compares 63 SNOW series (21 per segment).

- Code before the correction: the run stops in SNOW of PERLND 1 with
  `ZeroDivisionError`.
- With the correction all 63 pass; the largest difference is 0.00026 degF (PAKTMP),
  and the largest in PACK is 1.6e-5 in.
