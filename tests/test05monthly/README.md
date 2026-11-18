# test05monthly: monthly tables left out while their flags are on

Regression case for item P2 of issue #232. When a monthly variation flag of PWATER is
on and its monthly table is not in the input, HSPF uses the defaults of the table for
every month: 0.0 for MON-INTERCEP (VCSFG), 0.1 for MON-MANNING (VNNFG) and 0.0 for
MON-LZETPARM (VLEFG). The importer of HSP2 already fills a missing row of a monthly
table with these defaults when the table is given for other land segments; when the
table is absent altogether it wrote no monthly table, and PWATER fell back without a
message to the constant value of PWAT-PARM4.

The correction lists the three tables in `src/hsp2/data/DefaultableTables.csv`, so
that the importer writes them with their defaults when they are absent, as it does for
the other tables listed there. When the flag is off PWATER does not read them.
MON-UZSN, MON-INTERFLW and MON-IRC have no defaults in `Parameters.csv` and are left
as they are.

## The case

HSPF test05 (`tests/test05`, PERLND with SNOW and PWATER; VCSFG, VUZFG, VNNFG and VLEFG
on) with these changes, made by `make_case.py` from
`tests/test05/HSPFresults/TEST05bin.UCI`:

- INDELT 01:00 (test05: 00:30); at a sub-hourly step the PETADJ output of PWATER is
  item P1 of issue #232.
- Run from 1976-04-01 00:00 to 1976-07-31 24:00 (test05: the whole of 1976).
- MON-INTERCEP, MON-MANNING and MON-LZETPARM left out (MON-UZSN kept).
- PWAT-PARM4: NSUR 0.3 on PERLND 1, CEPSC 0.1 on PERLND 2 and LZETP 0.5 on PERLND 3
  (test05: NSUR 0.1, CEPSC and LZETP blank), constant values that differ from the
  defaults of the tables.
- DISPLY dropped (HSP2 does not run it; the gfortran build of HSPF stops in its annual
  summary).
- Binary output: PWATER of the three segments at every interval (flag 2, PIVL 1).

`HSPFresults/test05monthly.wdm` is `tests/test05/HSPFresults/TEST.WDM`, unchanged.

## Reference output

`HSPFresults/test05monthly.hbn` was written by HSPF built from the source distributed
with PyHSPF 0.2.4 (Lampert and Wu, 2015, Environmental Modelling & Software 68,
166-174), compiled with gfortran 13.3 and run as a stand-alone program. In that source
the routine HSPF_INI is an empty stub that leaves two outputs undefined; the build sets
both to 0. The build scripts are in the archive
https://doi.org/10.5281/zenodo.24292492 (`fortran_engine/`). HSPF reported no error
for the absent tables.

The sections of that source write the binary output with plain unformatted WRITE
statements, so the gfortran build gives unformatted sequential records framed by
4-byte length markers rather than the Lahey layout of the Windows builds. `HBNOutput`
reads both layouts, and the file is kept as HSPF wrote it.

## Expected result

`tests/test_regression.py` compares 87 PWATER series (29 per segment).

- Code before the correction: 26 series fail, 14 of PERLND 2 (interception storage up
  to 0.1 in, total storage up to 0.96 in) and 12 of PERLND 3 (lower zone storage up to
  8.1 in). On PERLND 1 the larger Manning's n changes no series by more than the
  tolerance in this period.
- With the correction all 87 pass; the largest difference is 1.7e-5 in (PERS).
