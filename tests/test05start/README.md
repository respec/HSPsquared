# test05start: a run from 06:00 to a December end, at a 1-hour step

Regression case for items M1, S1, F3 and F5 of issue #232.

- M1: `hoursval`, `monthval` and `dayval` in `utilities.py` built their calendar
  series only up to 00:00 on 31 December (hourly) or 1 December (monthly and daily) of
  the last year of the run. For a run that ends in December before 24:00 on
  31 December the series were shorter than the run and the loops read beyond their
  end; this run stopped with `ZeroDivisionError`. The series now run to the end of the
  following year; the values inside the run are unchanged.
- S1: HSPF sets its hour and day flags in the first interval of every run (`hper.f`:
  "hour and day flags are always set on in first interval"). `hoursval` with
  `dofirst=True` set the flag on the first value of its calendar series, 00:00 on
  1 January of the first year, so in a run that does not start at midnight the day
  flag was off in the first interval (and in a run that does not start on the hour,
  the hour flag too). The daily quantities were then computed only from the next
  midnight: in PWATER the interflow recession constants stay undefined (NaN) until
  then. With the correction the flag is on in the first interval of the run. The
  6 a.m. flag of SNOW (`hour6flag`) no longer passes `dofirst`: ICING of HSPF follows
  the clock, and its value at the start is item F3.
- F3: PSNOW of HSPF sets the initial 6 a.m. flag of the ground-ice routine (HR6FG) on
  when the run starts before hour 7 (06:00 is in hour 6 in the convention of HSPF) and
  when it starts without a pack (NOPACK). HSP2 took it from the first value of its
  6 a.m. flag series, which is off at 06:00, so HSPF updates the freezing capacity
  (XLNMLT) in the first interval of the run and HSP2 did so only the next morning.
- F5: NOPACK of HSPF sets the freezing capacity XLNMLT to 0 when the run starts without
  a pack; HSP2 kept the value of SNOW-INIT2.

`tests/test05start30` shows S1 in SNOW.

## The case

HSPF test05 (`tests/test05`, PERLND with SNOW and PWATER, ICEFG 1) with these changes,
made by `make_cases.py` from `tests/test05/HSPFresults/TEST05bin.UCI`:

- INDELT 01:00 (test05: 00:30); at this step the ground-ice items of a sub-hourly step
  (F2 and P1 of issue #232) do not arise.
- Run from 1976-11-15 06:00 to 1976-12-15 24:00 (test05: the whole of 1976).
- SNOW-INIT1 and SNOW-INIT2: on PERLND 1 a thin pack (0.1 in, so that it covers a fifth
  of the segment), for F3; on PERLND 2 no pack and XLNMLT 0.5 in, for F5; on PERLND 3
  the pack of test05.
- PWAT-STATE1: IFWS 0.1 in on PERLND 3 (test05: 0.0), so that interflow acts in the
  first interval, for S1.
- DISPLY dropped (HSP2 does not run it; the gfortran build of HSPF stops in its annual
  summary).
- Binary output: SNOW and PWATER of the three segments at every interval (flag 2,
  PIVL 1).

`HSPFresults/test05start.wdm` is `tests/test05/HSPFresults/TEST.WDM`, unchanged.

## Reference output

`HSPFresults/test05start.hbn` was written by HSPF built from the source distributed
with PyHSPF 0.2.4 (Lampert and Wu, 2015, Environmental Modelling & Software 68,
166-174), compiled with gfortran 13.3 and run as a stand-alone program. In that source
the routine HSPF_INI is an empty stub that leaves two outputs undefined; the build sets
both to 0. The build scripts are in the archive
https://doi.org/10.5281/zenodo.24292492 (`fortran_engine/`).

The sections of that source write the binary output with plain unformatted WRITE
statements, so the gfortran build gives unformatted sequential records framed by
4-byte length markers rather than the Lahey layout of the Windows builds. `HBNOutput`
reads both layouts, and the file is kept as HSPF wrote it.

## Expected result

`tests/test_regression.py` compares 150 series (SNOW and PWATER of the three
segments).

- Code before the corrections: the run stops with `ZeroDivisionError` (M1).
- M1 corrected: 17 series fail. PERLND 3: interflow, upper zone and total storage
  (S1), with the HSP2 messages "UZRAA exceeds UZRA array bounds" and "INTGB exceeds
  INTGRL array bounds". PERLND 1: 8 SNOW and 5 PWATER series (F3; XLNMLT 0.14 in,
  PACK 0.021 in). PERLND 2: XLNMLT (F5; 0.5 in).
- M1 and S1 corrected: the 14 series of F3 and F5 fail.
- M1, S1 and F3 corrected: XLNMLT of PERLND 2 fails (F5).
- All four corrected: all 150 pass; the largest difference is 3.1e-5 (DULL), and the
  largest in PACK is 1.2e-6 in.
