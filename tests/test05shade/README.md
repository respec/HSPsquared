# test05shade: shaded land at a 30-minute step

Regression case for item F1 of issue #232. In SNOW, HSP2 multiplied only the open part
of the longwave term by the interval length (`delt60`); HSPF multiplies the whole term,
shaded part included (HEXCHR in `hpersno.f`). The two codes therefore agree at a
1-hour step or without shading, and differ when SHADE > 0 at any other step. The cases
of the test set run hourly or without shading, so none of them showed it.

## The case

HSPF test05 (`tests/test05`, 30-minute step, PERLND with SNOW and PWATER) with these
changes to `tests/test05/HSPFresults/TEST05bin.UCI`:

- SNOW-PARM1: SHADE 0.0 on PERLND 1 (control), 0.3 on PERLND 2 and 0.72 on PERLND 3
  (test05 has 0.0 on all three).
- ICE-FLAG: ICEFG 0 (test05: 1), so that the ground-ice items of a sub-hourly step
  (F2 and P1 of issue #232) do not enter this case.
- Run from 1976-01-02 00:00 to 1976-04-30 24:00, which covers the snow season of the
  case (test05: the whole of 1976). Starting on 2 January also keeps out a defect of
  the WDM library of HSPF: when the instant one interval before the start has no data
  and the record is coarser than the run step, HSPF receives the hourly input one
  interval early.
- DISPLY dropped (HSP2 does not run it; the gfortran build of HSPF stops in its annual
  summary).
- Binary output: SNOW of the three segments at every interval (flag 2, PIVL 1);
  PWATER is not written, so the case compares SNOW only.

`HSPFresults/test05shade.wdm` is `tests/test05/HSPFresults/TEST.WDM`, unchanged.

## Reference output

`HSPFresults/test05shade.hbn` was written by HSPF built from the source distributed
with PyHSPF 0.2.4 (Lampert and Wu, 2015, Environmental Modelling & Software 68,
166-174), compiled with gfortran 13.3 and run as a stand-alone program. In that source
the routine HSPF_INI is an empty stub that leaves two outputs undefined; the build sets
both to 0. The build scripts are in the archive
https://doi.org/10.5281/zenodo.24292492 (`fortran_engine/`).

The sections of that source write the binary output with plain unformatted WRITE
statements, so the gfortran build gives unformatted sequential records framed by
4-byte length markers rather than the Lahey layout of the Windows builds. `HBNOutput`
reads both layouts, and the file is kept as HSPF wrote it. Check: the same engine run on
test05 as shipped writes the same 84 records as `tests/test05/HSPFresults/test05.hbn`
(same headers, names and dates), with values within 2e-5.

## Expected result

`tests/test_regression.py` compares the 63 SNOW series (21 per segment). With the code
before the correction, 29 series of segments 2 and 3 fail (PACK up to 0.23 in on
segment 2 and 0.68 in on segment 3); segment 1 (SHADE 0) passes. With the correction
all 63 pass; the largest difference is 0.0004 degF (PAKTMP) and the largest in PACK is
3e-5 in.
