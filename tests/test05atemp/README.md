# test05atemp: air temperature at a 30-minute step

Regression case for items A1 and E1 of issue #232.

- A1: ATEMP of HSPF (`hperair.f`) reads the gage temperature and updates the air
  temperature only in the first interval of each hour (HRFG), and holds both for the
  rest of the hour. HSP2 updated them in every interval (and with ELDAT 0 passed the
  gage temperature through), so at a step shorter than one hour it differed whenever
  the gage temperature, or with ELDAT not 0 the occurrence of precipitation, changed
  within the hour.
- E1: SNOW sets the density of new snow (RDNSN) at the first snowfall of an hour or of
  a precipitation event, and HSP2 started it at 0. When the air temperature changes
  within the hour (A1), the first snowfall of a run can begin in an interval that is
  neither, after rain earlier in the hour; the snowfall was then divided by 0 and the
  run stopped with `ZeroDivisionError`. With A1 corrected this no longer happens when
  ATEMP is active; the start value RDCSN keeps a run that reads a sub-hourly air
  temperature as input (ATEMP inactive) from stopping. HSPF leaves RDNSN undefined
  until it is first set (the state is filled with the integer -999, a NaN as a real),
  so this start value is a choice, not a reproduction.

## The case

HSPF test05 (`tests/test05`, 30-minute step, PERLND) with these changes to
`tests/test05/HSPFresults/TEST05bin.UCI` and `TEST.WDM`:

- ATEMP active (test05: inactive, with the air temperature read as AIRTMP); the gage
  temperature goes to EXTNL GATMP. ATEMP-DAT: ELDAT 0 on PERLND 1, 1111 ft on PERLND 2
  and -1111 ft on PERLND 3. With a round difference such as 1000 ft the air temperature
  falls exactly on PETMIN in some intervals, where the single precision of HSPF and the
  double precision of HSP2 can decide differently.
- Input data at the run step, written to `test05atemp.wdm`: the air temperature
  interpolated linearly in time from the 2-hour means of test05 (each mean placed at
  the middle of its interval), so that it changes within the hour; hourly
  precipitation, 2-hour solar radiation and daily evaporation and wind divided evenly
  over the 30-minute intervals; daily dew point and flow repeated. As input and run
  have the same step, the transformation DIV of EVAP, WIND and SOLR is left blank (HSPF
  does not accept DIV between equal steps).
- ICE-FLAG: ICEFG 0 (test05: 1), so that the ground-ice items of a sub-hourly step (F2
  and P1 of issue #232) do not enter this case.
- Run from 1976-02-20 00:00 to 1976-04-30 24:00. With this start the first snowfall of
  the run on PERLND 1 and 2 begins within an hour, after rain in its first half, which
  is where the code before the corrections meets E1.
- DISPLY dropped (HSP2 does not run it; the gfortran build of HSPF stops in its annual
  summary).
- Binary output: ATEMP and SNOW of the three segments at every interval (flag 2,
  PIVL 1). PWATER is not written: at a sub-hourly step its PETADJ output is item P1 of
  issue #232.

## Reference output

The program `wdm_write` of the HSPF build below wrote the input data to
`HSPFresults/test05atemp.wdm` with the WDM library of HSPF.

`HSPFresults/test05atemp.hbn` was written by HSPF built from the source distributed
with PyHSPF 0.2.4 (Lampert and Wu, 2015, Environmental Modelling & Software 68,
166-174), compiled with gfortran 13.3 and run as a stand-alone program. In that source
the routine HSPF_INI is an empty stub that leaves two outputs undefined; the build sets
both to 0. The build scripts, `wdm_write` included, are in the archive
https://doi.org/10.5281/zenodo.24292492 (`fortran_engine/`).

The sections of that source write the binary output with plain unformatted WRITE
statements, so the gfortran build gives unformatted sequential records framed by 4-byte
length markers rather than the Lahey layout that the canonical HSPF
(https://github.com/respec/FORTRAN, and the executable installed with BASINS) writes
through its compiler independent routines. `HBNReader` reads both layouts, and the file
is kept as HSPF wrote it.

## Expected result

`tests/test_regression.py` compares 69 series (ATEMP and SNOW of the three segments).

- Code before the corrections: the run stops in SNOW of PERLND 1 with
  `ZeroDivisionError` (E1).
- E1 corrected alone: 34 series fail (A1): the gage and air temperatures of the three
  segments (up to 3.7 degF) and 15 and 13 SNOW series of PERLND 1 and 2 (PACK up to
  0.18 in).
- A1 corrected, with or without E1: all 69 pass; the largest difference is 0.00012 in
  (PDEPTH), and the largest in PACK is 2e-5 in.
