''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''


ERRMSG = []


def GQUAL():
	''' Simulate the behavior of a generalized quality constituent'''


	# initialize month-data input
	GQAFXM = ???
	GQACNM = ???

	# initialize atmospheric deposition fluxes;    I= 5*MXGQAL
	GQCF10 = ???
	GQCF11 = ???

	# initialize table-type subscripts
	SUB1 = 0
	SUB2 = 0
	SUB3 = 0
	SUB4 = 0
	SUB5 = 0
	SUB51= 0
	SUB6 = 0
	SUB7 = 0
	SUBK = 0

	# table-type gq-gendata
	NGQUAL = ui['NGQUAL']
	TEMPFG = ui['TEMPFG']
	PHFLAG = ui['PHFLAG']
	ROXFG  = ui['ROXFG']
	CLDFG  = ui['CLDFG']
	SDFG   = ui['SDFG']
	PHYTFG = ui['PHYTFG']
	LAT    = ui['LAT']

	# table-type gq-ad-flags; Atmospheric Deposition Flags
	# zero value means data from ts as GQADFX or missing, >0 means index number of MONTH-DATA block
	GQADFG = ui['GQADFG']   # dimension 7

	NGQ3 = NGQUAL * 3

	# quality constituent loop
	# data for each constituent is seperate!!!!
	DO 120 I=1, NGQUAL
	
		GQADFG = ui['GQADFG']
		if GQAFXM in ui
		
		
		# read in month-data tables where necessary
	IF GQADFG(N):   # monthly flux must be read
		GQAFXM(1,J) = ???
		# convert units to internal - not done by MDATBL
		IF UUNITS == 1:   # convert from qty/ac.day to qty/ft2.ivl
			GQAFXM(I,J) = GQAFXM(I,J) * DELT60 / (24.0 * 43560.0)    # DO 10 I= 1, 12
		ELIF UUNITS == 2:      # convert from qty/ha.day to qty/m2.ivl
			GQAFXM(I,J) = GQAFXM(I,J) * DELT60 / (24.0*10000.0)      # DO 20 I= 1, 12
		END IF
	END IF
	IF GQADFG(N+1):   # monthly ppn conc must be read
		GQACNM(1,J = ???
	ENDIF 
	
	
	
		# table-type gq-qaldata
		GQID(K,I)  = ui['???']     # DO 40 K=1,5
		DQAL(I)    = RVAL(6)
		CONCID(I)  = RVAL(7)
		CONV(I)    = RVAL(8)
		QTYID(1,I) = RVAL(9)
		QTYID(2,I) = RVAL(10)
		RDQAL(I)   = DQAL(I) * VOL
		CINV(I)    = 1.0 / CONV(I)  # get reciprocal of unit conversion factor

		# process flags for this constituent
		QALFG(1,I)    # table-type gq-qalfg
		GQPM2(1,I)    # table-type gq-flg2

		# process parameters for this constituent
		IF  QALFG(1,I) == 1:    # qual undergoes hydrolysis
			SUB1 = SUB1 + 1
			HYDPM(1,I)  # table-type gq-hydpm
			HYDPM(K,I) = HYDPM(K,I) * DELTS   # convert rates from /sec to /ivl; DO 50 K=1,3
		END IF

		IF QALFG(2,I) == 1:    # qual undergoes oxidation by free radical processes
			SUB2 = SUB2 + 1
			ROXPM(1,I)                        # table-type gq-roxpm
			ROXPM(1,I) = ROXPM(1,I) * DELTS   # convert rate from /sec to /ivl
		END IF

		IF QALFG(3,I) == 1:	# qual undergoes photolysis
			# table-type gq-photpm
			SUB3 = SUB3 + 1
			PHOTPM(1,I)
		END IF

		IF QALFG(4,I) == 1:   # qual undergoes volatilization
			SUB4 = SUB4 + 1
			CFGAS(I)     # table-type gq-cfgas
		END IF

		IF QALFG(5,I) == 1:    # qual undergoes biodegradation
			SUB5 = SUB5 + 1
			# table-type gq-biopm
			BIOPM(1,I) = RVAL(1)
			BIOPM(2,I) = RVAL(2)
			# convert rate from /day to /ivl
			BIOPM(1,I) = BIOPM(1,I) * DELT60 / 24.0
			# specifies source of biomass data using GQPM2(7,I)
			BIOM = # from ts, monthly, constant
			SUB51 = SUB51 + 1 # only if BIOM comes from monthly
		END IF

		IF QALFG(6,I) == 1:   #  qual undergoes "general" decay
			SUB6 = SUB6 + 1
			# table-type gq-gendecay
			GENPM(1,I))
			# convert rate from /day to /ivl
			GENPM(1,I) = GENPM(1,I) * DELT60 / 24.0
		END IF

		IF QALFG(7,I) == 1:  # constituent is sediment-associated
			# get all required additional input
			SUB7 = SUB7 + 1
			ADDCPM               # table-type gq-seddecay
			# convert rates from /day to /ivl
			ADDCPM(1,I)	= ADDCPM(1,I) *	DELT60 / 24.0
			ADDCPM(3,I)	= ADDCPM(3,I) *	DELT60 / 24.0
	
			ADPM(1,1,I)     # table-type gq-kd
			ADPM(1,2,I)     # gq-adrate
			ADPM(K,2,I) = ADPM(K,2,I) * DELT60 / 24.0   # convert rates from /day to /ivl

			
			ADPM(1,3,I)     # table-type gq-adtheta
			SQAL(1,I)       # table-type gq-sedconc

			# find the total quantity of material on various forms of sediment
			RSQAL(4,I)	= 0.0
			RSQAL(8,I)	= 0.0
			RSQAL(12,I)	= 0.0
			DO 110 J=1,3
				RSQAL(J,  I) = SQAL(J,I)   * RSED(J)
				RSQAL(J+4,I) = SQAL(J+3,I) * RSED(J+3)
				RSQAL(J+8,I) = RSQAL(J,I)  + RSQAL(J+4,I)
				RSQAL(4,  I) = RSQAL(4,I)  + RSQAL(J,  I)
				RSQAL(8,  I) = RSQAL(8,I)  + RSQAL(J+4,I)
				RSQAL(12, I) = RSQAL(12,I) + RSQAL(J+8,I)
			# 110      CONTINUE
		ELSE:
			# qual not sediment-associated
			RSQAL(12,I )= 0.0
		END IF

		# find total quantity of qual in the rchres
		RRQAL(I) = RDQAL(I) + RSQAL(12,I)
		GQST(I,1)= RRQAL(I)
	# 120  CONTINUE

	# find values for global flags

	# gqalfg indicates whether any qual undergoes each of the decay processes or is sediment-associated

	DO 140 K=1,7
		GQALFG(K) = 0
		DO 130 I=1,NGQUAL
			IF (QALFG(K,I).EQ.1)
				GQALFG(K)= 1
		# 130    CONTINUE
	# 140  CONTINUE

	# qalgfg indicates whether a qual undergoes any of the 6 decay processes
	DO 160 I=1,NGQUAL
		QALGFG(I)=0
		DO 150 K=1,6
			IF (QALFG(K,I).EQ.1)
				QALGFG(I)=1
		# 150    CONTINUE
	# 160  CONTINUE

	# gdaufg indicates whether any constituent is a "daughter" compound through each of the 6 possible decay processes
	DO 180 K=1,6
		GDAUFG(K)= 0
		DO 170 I=1,NGQUAL
			IF (GQPM2(K,I).EQ.1) THEN
				GDAUFG(K)= 1
			END IF
		# 170    CONTINUE
	# 180  CONTINUE

	# daugfg indicates whether or not a given qual is a daughter compound
	DO 200 I=1,NGQUAL
		DAUGFG(I)=0
		DO 190 K=1,6
			IF (GQPM2(K,I).EQ.1) THEN
				DAUGFG(I)=1
			END IF
		# 190    CONTINUE
	# 200  CONTINUE

	# get initial value for all inputs which can be constant,
	# vary monthly, or be a time series-some might be over-ridden by
	# monthly values or time series

	# table-type gq-values
	TWAT  = RVAL(1)
	PHVAL = RVAL(2)
	ROC   = RVAL (3)
	CLD   = RVAL(4)
	SDCNC = RVAL(5)
	PHY   = RVAL (6)

	IF (TEMPFG.EQ.3) THEN
		TEMPM   # table-type mon-watemp
	END IF

	IF (GQALFG(1).EQ.1.AND.PHFLAG.EQ.3) THEN
		PHVALM      # table-type mon-phval
	END IF

	IF (GQALFG(2).EQ.1.AND.ROXFG.EQ.3) THEN
		ROCM   # table-type mon-roxygen
	END IF

	IF (GQALFG(3).EQ.1) THEN   # one or more quals undergoes photolysis-get required input
		ALPH    #  table-type gq-alpha
		GAMM    #  table-type gq-gamma
		DEL     #  table-type gq-delta
		KCLD    #  table-type gq-cldfact

		# get any monthly values needed for photolysis
		CLDM    # table-type mon-cloud;  IF CLDFG == 3
		END IF

		SDCNCM   # table-type mon-sedconc;  IF SDFG == 3
		PHYM     # table-type mon-phyto;    IF PHYTFG == 3
		CFSAEX   # table-type surf-exposed; IF HTFG == 0

		# fact1 is a pre-calculated value used in photolysis simulation
		FACT1 = CFSAEX * DELT60 / 24.0

		# decide which set of light data to use
		JCITMP = LAT
		LIGHT  = (ABS(INT(JCITMP)) + 5) // 10
		IF LIGHT == 0  # no table for equation, so use 10 deg table
			LIGHT= 1
		END IF

		# read the light data- 9 values to a line,
		SGRP  = 50 + LIGHT
		INITFG= 1
		DO 210 L=1,4
			LIT(K,L)  # FIRST 9; index K
			LIT(K,L)  # SECOND 9; index K
		# 210    CONTINUE

		# determine which season (set) of data to start with
		LITFG = 0

		# look one time-step ahead to see which "month" to use,
		# because we might be on a month boundary, in which case
		# datim will contain the earlier month
		IDELT = DELT
		DO 220 I=1,5
			NEWDAT(I) =DATIM(I)
		# 220    CONTINUE

		CALL ADDTIM()

		NEWMO = NEWDAT(2)
		LSET = NEWMO/3
		IF (LSET.EQ.0) THEN
			LSET= 4
		END IF

		# southern hemisphere is 2 seasons out of phase

		IF (LAT.LT.0) THEN
			LSET= LSET + 2
			IF (LSET.GT.4) THEN
				LSET= LSET - 4
			END IF
		END IF
	END IF

	IF (GQALFG(4).EQ.1) THEN   # one or more constituents undergoes volatilization process- input required to compute reaeration coefficient
		OXPM1  # flags - table-type ox-flags
		IF (HTFG.EQ.0) THEN  # get elevation and compute pressure correction factor
			ELEV   # table-type elev
			CFPRES= ((288.0-0.001981*ELEV)/288.0)**5.256
		END IF

		IF (LKFG.EQ.1) THEN   # rchres is a lake
			CFOREA    # table-type ox-cforea
		ELSE:         # rchres is a free-flowing stream
			if REAMFG == 1:    # tsivoglou method - table-type ox-tsivoglou
				REAKT
				TCGINV
				IF (HYDRFG.EQ.0) THEN    # read in len, delth - table-type ox-len-delth
					LEN
					DELTH
				ELSE:
					# len, delth are available from hydr
				END IF
			elif REAMFG == 2:     # owen/churchill/o'connor-dobbins - 
				TCGINV     # table-type ox-tcginv
			elif REAMFG == 3:     #   user formula - table-type ox-reaparm
				OXPM4
			END IF
		END IF
	END IF

	# process tables specifying relationship between "parent" and "daughter" compounds
	SUBK = 0

	DO 270 K=1,6
		IF (GDAUFG(K).EQ.1) THEN
			# read ngqual rows of mxgqal each - in case using old 3-row table
			C(1,1,K)   # table-type gq-daughter
			SUBK = SUBK + 1
		END IF
	# 270  CONTINUE

	IF (GQALFG(7).EQ.1) THEN     #  one or more quals are sediment-associated
		IF (SEDFG.EQ.0) THEN	# section sedtrn not active
			# ERRMSG
		END IF
	END IF

	IF (HYDRFG.EQ.1) THEN   # check that required options in section hydr have been selected
		IF (GQALFG(3).EQ.1.AND.AUX1FG.EQ.0) THEN
			# ERRMSG: error-simulation of photolysis requires aux1fg to be on to calculate average depth
		END IF

		IF (GQALFG(4).EQ.1) THEN
			IF (LKFG.EQ.0) THEN
				IF (AUX2FG.EQ.0) THEN
					# ERRMSG:  error-simulation of volatilization in a free flowing stream requires aux3fg on
				END IF
			ELSE:
				IF (AUX1FG.EQ.0) THEN
					# ERRMG: error-simulation of volatilization in a lake requires aux1fg on to calculate average depth
				END IF
			END IF
		END IF
	END IF


#####################  end PGQUAL

	VOLSP= VOL  # single precision version of vol

	# get any time series normally supplied by other active module sections
	AVDEP = ts['AVDEP']

	IF UUNITS == 1:   # english to metric
		AVDEPM = AVDEP * 0.3048
		AVDEPE = AVDEP
	ELSE:    # metric to english
		AVDEPM = AVDEP
		AVDEPE = AVDEP * 3.28
	END IF
	
	IF GQALFG(1) == 1:   # doing hydrolysis- we need ph data
		if PHFLAG == 1:   # case 1
			# time series
			IF PHFG == 1:    # use value computed in last time step
				PHVAL= PH
			ELSE:
				PHVAL from input time series
			END IF
		elif PHFLAG == 2:
			PHVAL =  user-supplied value, read in by run interpreter
		elif PHFLAG == 3:  # user-supplied monthly values
			IF (DAYFG .EQ. 1) THEN
				# interpolate a new value
				PHVAL= DAYVAL()
			END IF
		END IF

		# get water temp data
		if TEMPFG == 1:   # get from time series
			TW = ts['TW']
			TWAT = TW
		elif TEMPFG == 2:    
			TWAT = # user-supplied value

		elif TEMPFG == 3:  # user-supplied monthly values
			IF DAYFG == 1:
				# interpolate a new value
				TWAT= 
			END IF

		# this number is used to adjust reaction rates for temperature
		TW20 = TWAT - 20.0

		# tw20 may be required for bed decay of qual even if tw is undefined (due to vol=0.0)
		IF TWAT <= -10.0:
			TW20 = 0.0
		END IF

		# correct unrealistically high values of tw calculated in htrch
		IF TWAT >= 50.0:
			TW20 = 30.0
		END IF

		IF (GQALFG(2) .EQ. 1) THEN   # one or more constituents undergo oxidation by free radical processes
			ROC = ??? #  constant, monthly, ts['ROC']; roxfg

		IF (GQALFG(3) .EQ. 1) THEN   # one or more constituents undergoes photolysis decay
			IF (LITFG .EQ. 1) THEN
				# we need the next set of light data
				LSET= LSET + 1
				IF (LSET .GT. 4) THEN
					LSET= 1
				END IF
			END IF

		IF (EMONFG .EQ. 1) THEN
			# this is the last time step in the present month- check if
			# we will start a new season on next time step
			IF (NXTMON/3 .NE. MON/3) THEN
				LITFG = 1
			ELSE
				LITFG = 0
			END IF
		ELSE
			LITFG = 0
		END IF

		SDCNC = ???  # constant, monthly, ts; SDFG, note: interpolate to daily value only
		PHYTO = ???  # constant, monthly, ts; PHYTFG, note: interpolate to daily value only
		CLD = ???    # constant, monthly, ts['CLOUD']

		IF AVDEPE > 0.17:
			# depth of water in rchres is greater than two inches -
			# consider photolysis; this criteria will also be applied to other decay processes

			DO 250 L=1,18
				# evaluate the light extinction exponent- 2.76*klamda*d
				KL   = ALPH(L) + GAMM(L) * SDCNC + DEL(L) * PHY
				EXPNT= 2.76 * KL * AVDEPM * 100.0

				# evaluate the cloud factor
				CLDL= (10.0 - CLD * KCLD(L)) / 10.0

				IF EXPNT <= -20.0:
					EXPNT= -20.
				END IF
				IF EXPNT >= 20.0:
					EXPNT= 20.
				END IF
				# evaluate the precalculated factors fact2
				FACT2(L) = CLDL * LIT(L,LSET) * (1.0 - EXP(-EXPNT)) / EXPNT

			# 250      CONTINUE
		ELSE: 
			# depth of water in rchres is less than two inches -photolysis is not considered
		END IF
	END IF

	IF GQALFG(4) == 1:  # prepare to simulate volatilization by finding the oxygen reaeration coefficient
		IF (LKFG .EQ. 1)    
			WIND =  ts['WIND']
			AVDEP = ts['AVDEP']
		ELSE:         # water body is not a lake
			AVDEP = ts['AVDEP']
			AVVEL = ts['AVVEL']
		END IF

		IF UUNITS == 1:
			AVDEPE = AVDEP
			AVVELE = AVVEL
		ELSE:
			AVDEPE = AVDEP * 3.28
			AVVELE = AVVEL * 3.28
		END IF

		IF AVDEPE > 0.17:   # rchres depth is sufficient to consider volatilization
			# compute oxygen reaeration rate-korea
			KOREA = OXREA(LKFG,WIND,CFOREA,AVVELE,AVDEPE,TCGINV,REAMFG,REAK,REAKT,EXPRED,EXPREV,LEN, DELTH,TWAT,DELTS,DELT60,UUNITS,KOREA)
		ELSE:
			# rchres depth is not sufficient to consider volatilization
		END IF
	END IF

	PREC = ts['PREC']
	SAREA = ts['SAREA']

	# main loop-simulate each quality constituent
	DO 420 I=1,NGQUAL
		# get data on inflow of dissolved material
		FPT= GQIF1X(I)
		IF (FPT .GT. 0) THEN
			A = PAD(FPT + IVL1)
			IDQAL(I)= A * CONV(I)   # convert to internal "concentration" units
		ELSE:
			IDQAL(I)= 0.0
		END IF

		# compute atmospheric deposition influx ; N = 2*(I-1)+ 1
		#  dry deposition
		IF GQADFG(N) <= -1:
			GQADDR(I) = SAREA * CONV(I) * GQADFX
		ELIF GQADFG(N) >= 1:
			GQADDR(I)= SAREA * CONV(I) * DAYVAL(GQAFXM(MON,I),GQAFXM(NXTMON,I),DAY,NDAYS)
		ELSE:
			GQADDR(I)= 0.0
		END IF
		# wet deposition
		IF GQADFG(N+1) <= -1:
			GQADWT(I) = PREC * SAREA * GQADCN
		ELIF GQADFG(N+1) >= 1:
			GQADWT(I)= PREC * SAREA * DAYVAL(GQACNM(MON,I),GQACNM(NXTMON,I),DAY,NDAYS)
		ELSE:
			GQADWT(I)= 0.0
		END IF

		GQADEP(I) = GQADDR(I) + GQADWT(I)
		INDQAL    = IDQAL(I)  + GQADEP(I)

		# simulate advection of dissolved material
		DPDQAL(I) = DQAL(I)
		DPDQAL(I), RODQAL(I), ODQAL(1,I)) =  ADVECT(INDQAL, DPDQAL(I), RODQAL(I),ODQAL(1,I))
		DQAL(I) = DPDQAL(I)

		# get biomass input, if required (for degradation)
		IF QALFG(5,I) == 1:
			IDMK = GQPM2(7,I)
			if IDMK == 1:   # input time series-read from scratch pad
				BIO(I) = 
				TSSUB(1) = 1
			elif IDMK == 2:
				# single user-supplied value-read by run interpreter
			elif IDMK == 3:   # monthly user-supplied values
				IF (DAYFG .EQ. 1) THEN
					BIO(I)= DAYVAL(BIOM(MON,I),BIOM(NXTMON,I),DAY,NDAYS)
				END IF
		END IF

		IF AVDEPE > 0.17:   #  simulate decay of dissolved material
			DDQAL(1,I)) = DDECAY(QALFG(1,I),TW20,HYDPM(1,I),PHVAL,ROXPM(1,I),ROC,FACT2(1),FACT1,PHOTPM(1,I),KOREA,CFGAS(I),
			BIOPM(1,I),BIO(I),GENPM(1,I),VOLSP,DQAL(I),HR,DELT60,DDQAL(1,I))

			PDQAL(I)= 0.0
			DO 310 K=1,6
				IF GQPM2(K,I) == 1:    # this compound is a "daughter"-compute the contribution to it from its "parent(s)"

					ITOBE = I - 1
					DO 300 J= 1,ITOBE
						PDQAL(I)= PDQAL(I) + DDQAL(K,J)*C(I,J,K)
					# 300          CONTINUE
				END IF
			# 310      CONTINUE

			# update the concentration to account for decay and for input
			# from decay of "parents"- units are conc/l
			DQAL(I)= DQAL(I) + (PDQAL(I) - DDQAL(7,I))/VOLSP

		ELSE
			# rchres depth is less than two inches - dissolved decay is not considered
			DO 320 L=1,7
				DDQAL(L,I) = 0.0
			# 320      CONTINUE
			PDQAL(I)= 0.0
		END IF

		IF QALFG(7,I) == 1:   # this constituent is associated with sediment
			# zero the accumulators
			ISQAL(4,I)  = 0.0
			DSQAL(4,I)  = 0.0
			ROSQAL(4,I) = 0.0
			
			IF NEXITS > 1:
				DO 330 N= 1, NEXITS
					TOSQAL(N,I)= 0.0
				# 330        CONTINUE
			END IF

			# repeat for each sediment size fraction
			DO 350 J= 1,3
				# get data on inflow of sediment-associated material
				FPT= GQIF2X(J,I)
				IF (FPT .GT. 0) THEN
					ISQAL(J,I)= PAD(FPT + IVL1)*CONV(I)
				ELSE
					ISQAL(J,I)= 0.0
				END IF

				# advect this material, including calculation of deposition and scour
				GQECNT(1),SQAL(J,I),SQAL(J + 3,I),DSQAL(J,I), ROSQAL(J,I),OSQAL(1,J,I)) = ADVQAL (ISQAL(J,I),RSED(J),RSED(J + 3),\
				DEPSCR(J),ROSED(J),OSED(1,J),NEXITS,RCHNO, MESSU,MSGFL,DATIM, GQID(1,I),J,RSQAL(J,I),RSQAL(J + 4,I),GQECNT(1),
				SQAL(J,I),SQAL(J + 3,I),DSQAL(J,I),ROSQAL(J,I),OSQAL(1,J,I))

				ISQAL(4,I) = ISQAL(4,I) + ISQAL(J,I)
				DSQAL(4,I) = DSQAL(4,I) + DSQAL(J,I)
				ROSQAL(4,I) = ROSQAL(4,I) + ROSQAL(J,I)
				IF NEXITS > 1:
					DO 340 N= 1, NEXITS
						TOSQAL(N,I)= TOSQAL(N,I) + OSQAL(N,J,I)
					# 340          CONTINUE
				END IF
	
			# 350      CONTINUE

			TIQAL(I)  = IDQAL(I)  + ISQAL(4,I)
			TROQAL(I) = RODQAL(I) + ROSQAL(4,I)
			IF NEXITS > 1:
				DO 360 N= 1, NEXITS
					TOQAL(N,I) = ODQAL(N,I) + TOSQAL(N,I)
				# 360        CONTINUE
			END IF

			IF (AVDEPE .GT. 0.17) THEN     # simulate decay on suspended sediment
				SQAL((1),I), SQDEC((1),I)) =  ADECAY(ADDCPM(1,I),TW20,RSED(1),SQAL((1),I),SQDEC((1),I))

			ELSE:
				# rchres depth is less than two inches - decay of qual
				# associated with suspended sediment is not considered
				DO 370 L=1,3
					SQDEC(L,I)= 0.0
				# 370        CONTINUE
			END IF

			# simulate decay on bed sediment
			SQAL((4),I), SQDEC((4),I)) = ADECAY(ADDCPM(3,I),TW20,RSED(4),SQAL((4),I),SQDEC((4),I))

			# get total decay
			SQDEC(7,I)= 0.0
			DO 380 K=1,6
				SQDEC(7,I)= SQDEC(7,I) + SQDEC(K,I)
			# 380      CONTINUE

			IF AVDEPE > 0.17:  # simulate exchange due to adsorption and desorption
				DQAL(I), SQAL(1,I), ADQAL(1,I) = ADSDES(VOLSP,RSED(1),ADPM(1,1,I),TW20,DQAL(I),SQAL(1,I),ADQAL(1,I))
			ELSE:
				# rchres depth is less than two inches - adsorption and
				# desorption of qual is not considered
				DO 390 L=1,7
					ADQAL(L,I) = 0.0
				# 390        CONTINUE
			END IF

			# find total quantity of material on various forms of sediment
			RSQAL(4, I) = 0.0
			RSQAL(8, I) = 0.0
			RSQAL(12,I) = 0.0
			DO 400 J=1,3
				RSQAL(J,  I) = SQAL(J,I)   * RSED(J)
				RSQAL(J+4,I) = SQAL(J+3,I) * RSED(J+3)
				RSQAL(J+8,I) = RSQAL(J,I)  + RSQAL(J+4,I)
				RSQAL(4,  I) = RSQAL(4, I) + RSQAL (J,I)
				RSQAL(8,  I) = RSQAL(8, I) + RSQAL (J+4,I)
				RSQAL(12, I) = RSQAL(12,I) + RSQAL (J+8,I)
			# 400      CONTINUE

		ELSE:
			# qual constituent not associated with sediment-total just
			# above should have been set to zero by run interpreter
			TIQAL(I) = IDQAL(I)
			TROQAL(I) = RODQAL(I)
			IF NEXITS > 1:
				DO 410 N= 1, NEXITS
					TOQAL(N,I)= ODQAL(N,I)
				# 410        CONTINUE
			END IF
		END IF

		# find total quantity of qual in rchres
		RDQAL(I) = DQAL(I) * VOLSP
		IF QALFG(7,I) == 1:
			RRQAL(I)= RDQAL(I) + RSQAL(12,I)
		ELSE:
			RRQAL(I)= RDQAL(I)
		END IF
		GQST(I,1) = RRQAL(I)
	# 420  CONTINUE

	RETURN
	END


def adecay(addcpm, tw20, rsed, sqal, sqdec):   	# real  addcpm(2),rsed(3),sqal(3),sqdec(3),tw20
	''' simulate decay of material in adsorbed state'''

	if addcpm(1) > 0.0:     # calculate temp-adjusted decay rate
		dk  = addcpm(1) * addcpm(2)**tw20
		fact = 1.0 - exp(-dk)
		do 10 i=1,3    # particle size loop
			if sqal(i) > 1.0e-30:
				dconc    = sqal(i) * fact
				sqal(i)  = sqal(i) - dconc
				sqdec(i) = dconc * rsed(i)
			else:
				sqdec(i) = 0.0
		# 10     continue
	else
		do 20 i=1,3
			sqdec(i) = 0.0
		#20     continue

	return  sqal, sqdec


def adsdes(vol,rsed,adpm,tw20,dqal,sqal,adqal):  #  adpm(6,3),adqal(7),dqal,rsed(6),sqal(6),tw20,vol
	''' simulate exchange of a constituent between the dissolved
	state and adsorbed state-note that 6 adsorption site classes are
	considered: 1- suspended sand  2- susp. silt  3- susp. clay
	4- bed sand  5- bed silt  6- bed clay'''

	if vol > 0.0:     # adsorption/desorption can take place
		# first find the new dissolved conc.
		num   = vol	* dqal
		denom = vol
		do 10 j=1,6
			if rsed(j) > 0.0:  # this sediment class is present-evaluate terms due to it
				# transfer rate, corrected for water temp
				akj  = adpm(j,2) * adpm(j,3)**tw20
				temp = 1.0 / (1.0 + akj)

				# calculate 1/a and c/a
				ainv(j)  = akj * adpm(j,1) * temp
				cainv(j) = sqal(j) * temp

				# accumulate terms for numerator and denominator in dqal equation
				num   = num + (sqal(j) - cainv(j)) * rsed(j)
				denom = denom + rsed(j) * ainv(j)
		# 10     continue

		# calculate new dissolved concentration-units are conc/l
		dqal= num / denom

		# calculate new conc on each sed class and the corresponding adsorption/desorption flux
		adqal(7) = 0.0
		do 20 j=1,6
			if rsed(j) > 0.0:	# this sediment class is present-calculate data pertaining to it
				# new concentration
				temp = cainv(j) + dqal * ainv(j)

				# quantity of material transferred
				adqal(j) = (temp - sqal(j)) * rsed(j)
				sqal(j)  = temp

				# accumulate total adsorption/desorption flux
				adqal(7) = adqal(7) + adqal(j)
			else:     # this sediment class is absent
				adqal(j) = 0.0
				# sqal(j) is unchanged-"undefined"
		# 20     continue
	else:    # no water, no adsorption/desorption
		do 30 j=1,7
			adqal(j) = 0.0
			# sqal(1 thru 3) and dqal should already have been set to undefined values
		# 30     continue

	return dqal, sqal, adqal)


def advqal(isqal,rsed,bsed,depscr,rosed,osed,nexits,rchno,messu,msgfl,datim,
	gqid,j,rsqals,rbqals,ecnt,sqal,bqal,dsqal,rosqal,osqal)

	''' simulate the advective processes, including deposition and
	scour for the quality constituent attached to one sediment size fraction'''

	if depscr < 0.0:      # there was scour during the interval
		if bsed <= 0.0:   #  bed was scoured "clean"
			bqal  = -1.0e30
			dsqal = -1.0 * rbqals  # cbrb changed sign of dsqal; it should be negative for scour; fixed 4/2007 
		else:              # there is still bed material left
			bqal = rbqals / (bsed - depscr)
			dsqal= bqal * depscr

		# calculate concentration in suspension-under these conditions,
		# denominator should never be zero
		sqal   = (isqal + rsqals - dsqal) / (rsed + rosed)
		rosqal = rosed * sqal
	else:           # there was deposition or no scour/deposition during the interval
		denom = rsed + depscr + rosed
		if denom <= 0.0:     # there was no sediment in suspension during the interval
			sqal   = -1.0e30
			rosqal = 0.0
			dsqal  = 0.0
			if abs(isqal) > 0.0 or abs(rsqals) > 0.0:
				pass # errmsg: error-under these conditions these values should be zero
		else:   # there was some suspended sediment during the interval
			# calculate conc on suspended sed
			sqal   = (isqal + rsqals) / denom
			rosqal = rosed * sqal
			dsqal  = depscr * sqal
			if rsed <= 0.0:
					# rchres ended up without any suspended sediment-revise
					# value for sqal, but values obtained for rsqal,
					# rosqal, and dsqal are still ok
					sqal = -1.0e30

		# calculate conditions on the bed
		if bsed <= 0.0:     # no bed sediments at end of interval
			bqal = -1.0e30
			if abs(dsqal) > 0.0 or abs(rbqals > 0.0:
				# errmsg:  zrerror-under this condition these values should be zero
			end if
		else:     # there is bed sediment at the end of the interval
			rbqal= dsqal + rbqals
			bqal = rbqal / bsed

	if nexits > 1:   # we need to compute outflow through each individual exit
		if rosed <= 0.0:    # all zero
			do 10 l=1,5
				osqal(l)=0.0
			# 10       continue
		else:
			do 20 n= 1, nexits
				osqal(n) = rosqal * osed(n) / rosed
			# 20       continue
	return ecnt, sqal, bqal, dsqal, rosqal, osqal


def ddecay (qalfg,tw20,hydpm,phval,roxpm,roc,fact2,fact1,photpm,korea,cfgas,biopm,
	bio,genpm,volsp,dqal,hr,delt60,ddqal)
	''' estimate decay of dissolved constituent'''

    # bio,biopm(2),cfgas,ddqal(7),delt60,dqal,fact1,fact2(18),genpm(2),hydpm(4),korea,photpm(20),phval, roc,roxpm(2),tw20,volsp

	if dqal > 1.0e-25:     # simulate decay
		k1 = 0.0
		if qalfg(1) == 1:  # simulate hydrolysis
			khyd = hydpm(1) * 10.0**(-phval) + hydpm(2) * 10.0**(phval - 14.0) + hydpm(3)
			k1 = khyd * hydpm(4)**tw20  # adjust for temperature

		k2 = 0.0
		if qalfg(2) == 1:   # simulate oxidation by free radical processes
			krox = roxpm(1) * roc
			k2 = krox * roxpm(2)**tw20  # adjust for temperature

		k3 = 0.0
		if qalfg(3) == 1:     # simulate photolysis
			# go through summation over 18 wave-length intervals
			fact3 = 0.0
			do 10 l=1,18
				fact3 = fact3 + fact2(l) * photpm(l)
			# 10       continue
			k3 = fact1 * photpm(19) * fact3 * photpm(20)**tw20
		if delt60 < 24.0:
			if 18 > hr >= 6:  # it is a daylight hour; photolysis rate is doubled for this interval
				k3 = 2.0 * k3
			else:     # it is not a daylight hour; photolysis does not occur
				k3 = 0.0
		# else:
			# simulation interval is greater than 24 hours;
			# no correction is made to photolysis rate to
			# represent diurnal fluctuation

		# simulate volatilization
		k4 = korea * cfgas  if qalfg(4) == 1 else 0.0

		# simulate biodegradation
		k5 = biopm(1) * bio * biopm(2)**tw20  if qalfg(5) == 1 else 0.0

		# simulate simple first-order decay
		k6 = genpm(1) * genpm(2)**tw20  if qalfg(6) == 1 else 0.0

		# get total decay rate
		k7 = k1 + k2 + k3 + k4 + k5 + k6

		# calculate the total change in material due to decay-units are conc*vol/l.ivl
		ddqal(7) = dqal * (1.0 - exp(-k7)) * volsp

		# prorate among the individual decay processes- the method used
		# for proration is linear, which is not strictly correct, but
		# should be a good approximation under most conditions
		do 30 i=1,6
			if k7 > 0.0:
				ddqal (i) = k(i) / k7  * ddqal(7)
			else:
				ddqal(i) = 0.0
			end if
		# 30     continue

	else:
		# too little dissolved material to simulate decay
		do 40 i=1,7
			ddqal(i) = 0.0
		# 40     continue
	return ddqal
