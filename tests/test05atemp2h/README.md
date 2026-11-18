# test05atemp2h: air temperature at a 2-hour step

Regression case for item A2 of issue #232. ATEMP of HSPF (`hperair.f`) uses the dry
lapse rate of the hour in which the interval ends, LAPSE(HR) with HR from 1 to 24, at
any step. HSP2 averaged the hourly lapse rates over the interval: the same value at
steps of one hour or less, a different one at longer steps, where the air temperature
differs whenever ELDAT is not 0.

## The case

As `tests/test05atemp` and made by the same script, `tests/test05atemp/make_cases.py`
(ATEMP active with ELDAT 0, 1111 and -1111 ft; ICEFG 0; input data at the run step;
DISPLY dropped), with these differences:

- INDELT 02:00. Input data at 2 hours: hourly precipitation summed in pairs; the 2-hour
  air temperature and solar radiation of test05 as they are; daily evaporation and wind
  divided evenly, daily dew point and flow repeated.
- Run from 1976-01-01 00:00 to 1976-04-30 24:00.
- Binary output: ATEMP, SNOW and PWATER of the three segments at every interval
  (flag 2, PIVL 1).

The reference output was made as described in `tests/test05atemp/README.md`.

## Expected result

`tests/test_regression.py` compares 156 series (ATEMP, SNOW and PWATER of the three
segments).

- Code before the correction: 29 series of PERLND 2 and 3 fail: the air temperature
  (0.17 degF), 19 SNOW and 8 PWATER series (PACK up to 0.033 in). PERLND 1 (ELDAT 0)
  passes.
- With the correction all 156 pass; the largest difference is 1.7e-5 in (PERS), and
  the largest in PACK is 2e-6 in.
