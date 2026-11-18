# test05daily: test05 at a daily step

Regression case for items F4, F6 and E2 of issue #232, all in SNOW:

- F6: at steps longer than 6 hours SNOW read `siminfo["ICEFLG"]`, a key that is never
  set (the flag is stored as `ICEFG`), so every run with SNOW at such a step, a daily
  step included, stopped with `KeyError: 'ICEFLG'`.
- F4: HSP2 used the seasons of the northern hemisphere for the snow albedo everywhere.
  ALBED in `hpersno.f` exchanges the summer and winter equations when LAT < 0; HSP2
  did not read LAT.
- E2: when evaporation removes the whole pack while it still holds a heat deficit
  (NEGHTS > 0), the pack temperature was computed by dividing by PACKF = 0 and the run
  stopped with `ZeroDivisionError`. HSPF divides by zero at the same place, carries an
  infinite pack temperature to the end of the interval and removes the pack there
  (NOPACK); with the correction HSP2 uses 32 degF in that case, which leaves the same
  state at the end of the interval.

None of the cases of the test set runs SNOW at a step longer than 6 hours or south of
the equator. E2 needs a thin pack that one interval of evaporation removes while it is
still cold; this case reaches it with SNOEVP at its upper limit.

## The case

HSPF test05 (`tests/test05`, PERLND with SNOW and PWATER) at a daily step, with these
changes, made by `make_case.py` from `tests/test05/HSPFresults/TEST05bin.UCI` and
`TEST.WDM`:

- INDELT 24:00 (test05: 00:30).
- Input data brought to the daily step: hourly precipitation and 2-hour solar
  radiation summed, 2-hour air temperature averaged over each day; potential
  evaporation, wind, dew point and flow are daily in test05 and are used as they are.
  The data sets keep their numbers, types and forms (TSFORM). As input and run have the
  same step, the transformation DIV of EVAP, WIND and SOLR is left blank (HSPF does not
  accept DIV between equal steps).
- ICE-FLAG: ICEFG 0 (test05: 1). HSPF reports an input error and does not run when
  ICEFG is 1 at a step longer than 6 hours.
- PERLND 1 as in test05 (control).
- PERLND 2 south of the equator: LAT -43 (test05: 43), for F4.
- PERLND 3 with COVIND 0.1 in (test05: 0.5) and SNOEVP 1.0 (the upper limit of the
  parameter; test05: 0.05), so that a thin pack covers the whole segment, for E2: on
  16 December 1976 evaporation removes the whole pack of the segment (0.027 in) while
  it holds a heat deficit.
- Run over 1976, as test05. All input data sets are daily, so the defect of the WDM
  library noted in `tests/test05shade/README.md` (coarser record than the run step)
  does not arise.
- DISPLY dropped (HSP2 does not run it; the gfortran build of HSPF stops in its annual
  summary).
- Binary output: SNOW and PWATER of the three segments at every interval (flag 2,
  PIVL 1).

## Reference output

`make_case.py` writes the daily data as text files; the program `wdm_write` of the HSPF
build below wrote them to `HSPFresults/test05daily.wdm` with the WDM library of HSPF.

`HSPFresults/test05daily.hbn` was written by HSPF built from the source distributed
with PyHSPF 0.2.4 (Lampert and Wu, 2015, Environmental Modelling & Software 68,
166-174), compiled with gfortran 13.3 and run as a stand-alone program. In that source
the routine HSPF_INI is an empty stub that leaves two outputs undefined; the build sets
both to 0. The build scripts, `wdm_write` included, are in the archive
https://doi.org/10.5281/zenodo.24292492 (`fortran_engine/`).

The sections of that source write the binary output with plain unformatted WRITE
statements, so the gfortran build gives unformatted sequential records framed by
4-byte length markers rather than the Lahey layout of the Windows builds. `HBNOutput`
reads both layouts, and the file is kept as HSPF wrote it.

## Expected result

`tests/test_regression.py` compares 150 series (SNOW and PWATER of the three
segments).

- Code before the corrections: the run stops in SNOW of PERLND 1 with
  `KeyError: 'ICEFLG'` (F6).
- F6 corrected, alone or with F4: the run stops in SNOW of PERLND 3 with
  `ZeroDivisionError` (E2).
- F6 and E2 corrected: 33 series of PERLND 2 fail, 14 of SNOW and 19 of PWATER (PACK
  up to 1.59 in, PERO up to 0.024 in); PERLND 1 and 3 pass (F4).
- All three corrected: all 150 pass; the largest difference is 0.00016 degF (PAKTMP),
  and the largest in PACK is 2e-6 in.
