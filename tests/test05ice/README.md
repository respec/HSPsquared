# test05ice: ground ice at a 30-minute step

Regression case for items F2 and P1 of issue #232, which act with ICEFG = 1 at a step
below one hour.

- F2 (SNOW): ICING in HSPF (`hpersno.f`) sets the 6 a.m. flag HR6FG while HR < 7, HR
  being the hour (1 to 24) in which the interval ends, and updates the freezing
  capacity of the pack once, in the first interval of the morning with HR >= 7. HSP2
  set the flag only in the first interval of each of the hours 0 to 5, so at a step
  below one hour it updated the capacity up to six times a night.
- P1 (PWATER): PWATRX in HSPF (`hperwat.f`) recomputes PETADJ and the frozen-ground
  factor INFFAC on the hour and keeps them in between. HSP2 recomputed INFFAC in every
  interval from the PACKI of that interval, applied the soil-temperature option
  (IFFCFG = 2) in every interval, and wrote PETADJ only on the hour (0 in the other
  intervals).

## The case

HSPF test05 (`tests/test05`, 30-minute step, PERLND with SNOW and PWATER, ICEFG 1, SHADE
0) with these changes to `tests/test05/HSPFresults/TEST05bin.UCI`:

- Run from 1976-01-02 00:00 to 1976-04-30 24:00, which covers the snow season of the
  case (test05: the whole of 1976). Starting on 2 January also keeps out a defect of
  the WDM library of HSPF: when the instant one interval before the start has no data
  and the record is coarser than the run step, HSPF receives the hourly input one
  interval early.
- DISPLY dropped (HSP2 does not run it; the gfortran build of HSPF stops in its annual
  summary).
- Binary output: SNOW and PWATER of the three segments at every interval (flag 2,
  PIVL 1).

`HSPFresults/test05ice.wdm` is `tests/test05/HSPFresults/TEST.WDM`, unchanged.

## Reference output

`HSPFresults/test05ice.hbn` was written by HSPF built from the source distributed with
PyHSPF 0.2.4 (Lampert and Wu, 2015, Environmental Modelling & Software 68, 166-174),
compiled with gfortran 13.3 and run as a stand-alone program. In that source the
routine HSPF_INI is an empty stub that leaves two outputs undefined; the build sets
both to 0. The build scripts are in the archive https://doi.org/10.5281/zenodo.24292492
(`fortran_engine/`).

The sections of that source write the binary output with plain unformatted WRITE
statements, so the gfortran build gives unformatted sequential records framed by 4-byte
length markers rather than the Lahey layout that the canonical HSPF
(https://github.com/respec/FORTRAN, and the executable installed with BASINS) writes
through its compiler independent routines. `HBNOutput` reads both layouts, and the file
is kept as HSPF wrote it. Check: the same engine run on test05 as shipped writes the
same 84 records as `tests/test05/HSPFresults/test05.hbn` (same headers, names and
dates), with values within 2e-5.

## Expected result

`tests/test_regression.py` compares the 150 series of SNOW and PWATER (50 per
segment). With the code before the corrections, 74 series fail on all three segments
(PACK up to 0.13 in, INFFAC up to 0.26, PETADJ up to 1). With the corrections all 150
pass; the largest difference is 0.0003 degF (PAKTMP) and the largest in PACK is
2e-5 in.
