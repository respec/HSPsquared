''' Copyright (c) 2020 by RESPEC, INC.
Author: Robert Heaphy, Ph.D.
License: LGPL2
'''


ERRMSG:  [warning, fatal error]

def ACIDPH (FSTCAL):
	VOL = ts['VOL']
	
	
	''' Simulate acid-base chemistry of mine drainage-affected reaches'''
	ACFLAG = ui['ACFLAG']   # table-type acid-flags;  returns 14 values  AC01 to AC14
	ACPARM = ui['ACPARM']   # table-type acid-parms;  returns 7 values ACPARM1 to ACPARM7
	ACCONC = ui['ACCONC']   # table-type acid-init; returns 7 values ACCONC1 to ACCON7

	ACSTOR[:,1] = ACCONC[:] * VOL
	ACPH = -LOG10(ACCONC(5))

	# initialize all computed fluxes
	ACFLX1[:,1]   = 0.0    # dimension 7
	ACFLX3[:,1]   = 0.0    # dimension 7
	ACFLX2[J,I,1] = 0.0    # I dimension 7, J dimension 5
	ACFLXC[:,1]   = 0.0    # dimension 10
	ACFLXG[:,1]   = 0.0   # dimension 3


	#$ACINIT()
	''' Initialize common block versions of number of chemicals, number
	of cons, number of gquals, chemical names, and molec wts'''


	'''
C     IMETHOD - method 
C     NCHEMS - number of chemical species
C     NCONS  - number of CONServatives
C     NGQLS  - number of GQUALs
C     INAMES - chemical names
C     MOLWT  - molecular weights of chemicals
	'''

	IMETHOD = ACFLAG[2]
	IF IMETHOD == 1 OR IMETHOD == 2 OR IMETHOD == 3:
		NCHEMS    = 7
		NCONS     = 2
		NGQLS     = 0
		CNAMES(1) = '  TOTAL AL+3'
		CNAMES(2) = '   FREE AL+3'
		CNAMES(3) = '  TOTAL FE+3'
		CNAMES(4) = '   FREE FE+3'
		CNAMES(5) = '          H+'
		CNAMES(6) = ' TOT INORG C'
		CNAMES(7) = '  ALKALINITY'
		MOLWT(1) = 26.98
		MOLWT(2) = 26.98
		MOLWT(3) = 55.85
		MOLWT(4) = 55.85
		MOLWT(5) = 1.008
		MOLWT(6) = 12.01
		MOLWT(7) = 50.04
	ELIF IMETHOD == 4:
		NCHEMS    = 5
		NCONS     = 1
		NGQLS     = 0
		CNAMES(1) = '        Al+3'
		CNAMES(2) = '        Fe+3'
		CNAMES(3) = '        Mn+2'
		CNAMES(4) = '     ACIDITY'
		CNAMES(5) = '          H+'
		MOLWT(1) = 26.98
		MOLWT(2) = 55.85
		MOLWT(3) = 54.94
		MOLWT(4) = 50.04
		MOLWT(5) = 1.008
	END IF

	DO 10 I= 1, NCHEMS
		READ (CNAMES(I),1000) (INAMES(I,J),J=1,3)
	# 10   CONTINUE


	# check number of chemicals, cons, and gquals
	NCNTMP = NUMCON
	IF NUMCON == 0:
		NCNTMP = -999
	END IF
	NGQTMP = NUMGQL
	IF NUMGQL == 0:
		NGQTMP = -999
	END IF

	IF NCNTMP > NCONS OR NGQTMP > NGQUAL:
		# ERRMSG: error - must simulate at least as many species in cons and gqual  as are used in acidph section
	END IF

################## END  PACID

	IF FSTCAL == 1:
		# check ratio of co2 to atmospheric co2
		IF (ACPARM(1) <= 0.0 OR ACPARM(1) > 2.0:
			ACPARM(1) = 1.0
			# ERRMSG: warn that ratio has been changed
		END IF
	END IF

	# get average depth
	AVDEP = ts['AVDEP']

	# convert depth to english units
	IF (UUNITS == 1:   # english
		AVDEPE = AVDEP
	ELSE:   # metric to english
		AVDEPE = AVDEP * 3.28
	END IF

	# get inflows from pad - must convert to molar units
	DO 10 I= 1, NUMCHM
		ACINFL(I) = ts['ACINFL' + str(I)]
	10   CONTINUE

	# advect all chemicals
	DO 20 I= 1, NUMCHM
		DACCNC(I) = ACCONC(I)
		DACCNC(I), ACFLX1(I,1), ACFLX2(1,I,1) =  ADVECT(ACINFL(I,1), DACCNC(I), ACFLX1(I,1), ACFLX2(1,I,1)
		ACCONC(I) = DACCNC(I)
	# 20   CONTINUE

	IF VOL > 0.0:   # water in reach, attempt to simulate processes
		# get water temperature
		TW = ??? # from constant, user param(ACPARM(3) if ACPARM <= 0), etc.
		TWKELV  = TW + 273.16

		# check for sufficient water (at least 2 inches) in reach
		IF AVDEPE > 0.17:   # enough water to warrant simulation of chemical processes
			# get constants used by all routines
			DACPM1 = ACPARM(1)
			DACPM2 = ACPARM(2)
			#$ACDATA()
			'''initialize constants and assign temperature-dependent values to constants used by acidph routines'''

			# al - oh and al - so4 complexation constants
			rkal1  = 1.03d-5
			rkal2  = 7.14d-6
			rkal3  = 1.0d-14
			rkal4  = 94.2
			rkals1 = 1.63d3
			rkals2 = 1.29d5

			# al - f complexation constants
			rkalf1 = 1.05d7
			rkalf2 = 5.77d12
			rkalf3 = 1.07d17
			rkalf4 = 5.37d19
			rkalf5 = 8.33d20
			rkalf6 = 7.49d20

			# fe - f complexation constants
			rkfef1 = 1.0d6
			rkfef2 = 1.585d9
			rkfef3 = 5.012d11

			# fe - oh and fe - so4 complexation constants
			rkfe1  = 6.457d-3
			rkfe2  = 3.162d-4
			rkfe3  = 3.981d-8
			rkfe4  = 3.162d-9
			rkfes1 = 1.413d4
			rkfes2 = 2.399d5

			# atmospheric co2 (atm)
			atmco2 = 3.16d-4

			# first dissociation constant of h2co3
			rkco21 = 10.0**(545.56 + 0.12675 * twkelv - 215.21 * log10(twkelv) - 17052.0 / twkelv)

			# second dissociation constant of h2co3
			rkco22 = 10.0**(-2902.39 / twkelv - 0.02379 * twkelv + 6.498)

			# ion product of water (stumm and morgan, 1981, p. 127)
			rkh2o = 10.0**(-4470.99 / twkelv + 6.0875 - 0.01706 * twkelv)

			# -log(henry's law constant of co2) in units of mole/liter/atm
			ta = -2385.73 / twkelv + 14.0184 - 0.0152642 * twkelv

			# use ideal gas law to convert to dimensionless form
			rkhco2 = 10.0**(-ta) * 0.082057 * twkelv

			# compute co2 concentration as fraction of atmospheric value
			co2 = fraco2 * atmco2 * 10.0**(-ta)

			# ksp of solid MIN_eral
			if OPTFLG == 1:
				# gibbsite
				if   KSPFLG == 2:   rksp = 2.308d-33   # microcrystalline
				elif KSPFLG == 3:   rksp = 6.068d-34   # natural
				elif KSPFLG == 4:   rksp = 1.329d-34   # synthetic
				elif KSPFLG == 5:   rksp = rkspin      # user value
				else:               rksp = 6.501d-32   # amorphous

				rksp *= exp(32.0632 - 9560.0 / twkelv)
			elif OPTFLG == 2:
				rksp = 2.7      # fe(oh)3
			# end ACDATA()

			# assign double-precision versions of concs and save previous time step values
			DO 30 I= 1, NUMCHM
				DACCNS(I) = ACCONC(I)
				DACCNC(I) = ACCONC(I)
				DMOLWT(I) = ACCONV(I)
			# 30       CONTINUE

			# call the simulation routines, depending on option flag
			IF ACFLAG(2) == 1:        # option 1: al(oh)3 (gibbsite) saturation and fe
				CALL ACCAL1()
			ELIF ACFLAG(2) == 2:
				CALL ACCAL2()
			ELIF ACFLAG(2) == 3:   # option 3: no solid saturated and al and fe
				CALL ACCAL3()
			ELSE IF (ACFLAG(2) == 4:  # option 4: WV method (Al, Fe, Mn, acidity, conductivity)
				DACPM4 = ACPARM(4)
				DACPM5 = ACPARM(5)
				CALL ACCAL4()
			ELSE:
				# ERRMSG: error no. 2, invalid value of acflag(2) (this is a fatal error)
			END IF

			# assign fluxes, storages, and real*4 concs
			DO 40 I= 1, NUMCHM
				ACFLX3(I,1) = (DACCNC(I) - DACCNS(I))*VOL
				ACSTOR(I,1) = DACCNC(I) * VOL
				ACCONC(I)   = DACCNC(I)
			# 40       CONTINUE
			ACPH = DPH
		ELSE:   # not enough water to warrant simulation of chemical processes fluxes set to zero
			DO 50 I= 1, NUMCHM
				ACSTOR(I,1) = DACCNC(I) * VOL
				ACCONC(I)   = DACCNC(I)
				ACFLX3(I,1) = 0.0
			# 50       CONTINUE
			ACPH = -DLOG10(DACCNC(5))
		END IF
	ELSE:   # reach is dry, set concentrations, storages and fluxes appropriately
		DO 60 I= 1, NUMCHM
			ACCONC(I)   = -1.0E+30
			ACSTOR(I,1) = 0.0
			ACFLX3(I,1) = 0.0
		# 60     CONTINUE
		ACPH = -1.0E+30
	END IF

	RETURN
	END



def ACCAL1 (TOTAL,TOTFE,ST,FT,CO2, RKAL1,RKAL2,RKAL3,RKAL4,RKALS1,RKALS2,
	RKALF1,RKALF2,RKALF3,RKALF4,RKALF5,RKALF6,	 RKFE1,RKFE2,RKFE3,RKFE4,RKFES1,RKFES2,
	RKFEF1,RKFEF2,RKFEF3,RKC1,RKC2,RKW,RKSP,TWKELV,RKHCO2,H,
	PH,ALFREE,ALOH,ALS,ALF,FEFREE,FEOH,FES,FEF,ALK,ALKW,ALKC,ALKAL,ALKFE,TIC,F1,S1,INUM)

	''' Estimate pH value based on total Al, Fe conc. and Gibbsite control'''

	# initialize parameters used in pH deterMINing algorithm
	ITER  = 0
	MIN_   = 0
	MAX_   = 0
	hMIN  = 0.0
	hMAX  = 0.0
	erMIN = 0.0
	erMAX = 0.0
	INUM = 0

	# set initial estimate of [H+]
	IF 1.0E-1 >  H > 1.0E-14:
		# use previous value
	ELSE:  # use ph=5
		H = 10.0**(-5)
	END IF

	# compute free aluMIN_um conc. based on Gibbsite MIN_eral solubility
	# 100  CONTINUE
	INUM   = INUM + 1
	ALFREE = RKSP * (H / RKW)**3

	# Al(+3) __ aloh(+2) __ al(oh)2(+1) __ al(oh)3(0) __ al(oh)4(-)
	AAL0 = 1.0 / (1.0 + RKAL1 / H * (1.0 + RKAL2 / H * (1.0 + RKAL3 / H * (1.0 + RKAL4 / H))))
	AAL1 = RKAL1 / H * AAL0
	AAL2 = RKAL2 / H * AAL1
	AAL3 = RKAL3 / H * AAL2
	AAL4 = RKAL4 / H * AAL3

	# Fe(+3) __ feoh(+2) __ fe(oh)2(+1) __ fe(oh)3(0) __ fe(oh)4(-)
	AFE0 = 1.0 / (1.0 + RKFE1 / H * (1.0 + RKFE2 / H * (1.0 + RKFE3 / H * (1.0 + RKFE4/H)))) 
	AFE1 = RKFE1 / H * AFE0
	AFE2 = RKFE2 / H * AFE1
	AFE3 = RKFE3 / H * AFE2
	AFE4 = RKFE4 / H * AFE3

	FETX(1) = TOTFE

	200  CONTINUE
	DO 300 I=1,2
		FRTIO1 = 0.0
		FRTIO2 = 0.0
		SRTIO1 = 0.0
		SRTIO2 = 0.0
		FET    = FETX(I)

		# compute sulfate complexation
		IF ST > 0.0:
			A      = RKALS2 * ALFREE + RKFES2 * FET * AFE0
			B      = 1.0 + RKALS1 * ALFREE + RKFES1 * FET * AFE0
			C      = -ST
			S1     = (-B + SQRT(B * B - 4.0 * A * C)) / (2.0 * A)
			SRTIO1 = AAL0 * (RKALS1 * S1 + RKALS2 * S1 * S1)
			SRTIO2 = AFE0 * (RKFES1 * S1 + RKFES2 * S1 * S1)
		END IF

		# compute fluoride complexation
		IF FT > 0.0:
			A      = -FT
			B      = 1.0 + RKALF1 * ALFREE + RKFEF1 * FET * AFE0
			C      = RKALF2 * ALFREE + RKFEF2 * FET * AFE0
			D      = RKALF3 * ALFREE + RKFEF3 * FET * AFE0
			E      = RKALF4 * ALFREE
			F      = RKALF5 * ALFREE
			G      = RKALF6 * ALFREE
			NXX    = 6
			F1     = NEWTON(NXX,FT,A,B,C,D,E,F,G)
			FRTIO1 = AAL0 * (RKALF1 * F1 + RKALF2 * F1**2 + RKALF3 *F1**3 + RKALF4 * F1**4 + RKALF5 * F1**5 + RKALF6 * F1**6)
			FRTIO2 = AFE0 * (RKFEF1 * F1 + RKFEF2 * F1**2 + RKFEF3 *F1**3)
		END IF
        FETX(I+1) = TOTFE / (1.0 + FRTIO2 + SRTIO2)
        IF ABS((FETX(I+1) - FETX(I)) / FETX(I)) < 0.001:
			GO TO 400
	300  CONTINUE

	SLOPE1  = (FETX(3) - FETX(2)) / (FETX(2) - FETX(1))
	FETX(1) = FETX(3) + SLOPE1 / (1.0 - SLOPE1) * (FETX(3) - FETX(2))

	GO TO 200

	400  CONTINUE

	FEFREE = FET * AFE0
	FEOH   = FET - FEFREE
	FEF    = FET * FRTIO2
	FES    = FET * SRTIO2
	ALOH   = (ALFREE / AAL0) - ALFREE
	ALF    = (ALFREE + ALOH) * FRTIO1
	ALS    = (ALFREE + ALOH) * SRTIO1

	# compute total aluMIN_um concentration
	ALtMAX = ALOH + ALF + ALS + ALFREE

	ERROR  = ALtMAX - TOTAL
	IF ABS(ERROR) <= 1.0e-8:
		GO TO 600
	IF (MIN_*MAX_ .NE. 1) THEN
		IF (ERROR .LT. 0.0) THEN
			hMAX  = H
			erMAX = ERROR
			MAX_   = 1
			IF (MIN_ .EQ. 1)
				GO TO 500
			H = H * 2.0
		ELSE:
			hMIN  = H
			erMIN = ERROR
			MIN_   = 1
			IF (MAX_ .EQ. 1)
				GO TO 500
			H = H * 0.5
		END IF
		GO TO 100
	END IF

	500  CONTINUE
	ITER = ITER + 1
	IF ITER <= 1:
		SLOPE  = (erMAX - erMIN) / (hMAX - hMIN)
		H      = hMAX - erMAX / SLOPE
	ELSE:
		CONV(hMIN,hMAX,erMIN,erMAX,ERROR, H)
	END IF

	GO TO 100

	600  CONTINUE

	PH = -LOG10(H)

	# H2c03(*) __ hco3(-) __ co3(-2)
	AC0  = 1.0/(1.0 + RKC1/H*(1.0 + RKC2/H))
	AC1  = RKC1 / H * AC0
	AC2  = RKC2 / H * AC1
	TIC  = CO2 * (273.15 / TWKELV) * RKHCO2 / AC0

	# compute components of alkalinity CO3--AL--H2O
	ALKC  = (AC1 + 2.0 * AC2) * TIC
	ALKAL = (4.0 * AAL4 + 3.0 * AAL3 + 2.0 * AAL2 + AAL1) * (ALOH + ALFREE)
	ALKFE = (4.0 * AFE4 + 3.0 * AFE3 + 2.0 * AFE2 + AFE1) * FET
	ALKW  = RKW / H - H
	ALK   = ALKC + ALKW + ALKAL + ALKFE

	RETURN  H,PH,ALFREE,ALOH,ALS,ALF,FEFREE,FEOH,FES,FEF,ALK,ALKW,ALKC,ALKAL,ALKFE,TIC,F1,S1,INUM)
	END


def ACCAL2(TOTAL,TOTFE,ST,FT,CO2,RKAL1,RKAL2,RKAL3,RKAL4,RKALS1,RKALS2,
	RKALF1,RKALF2,RKALF3,RKALF4,RKALF5,RKALF6,RKFE1,RKFE2,RKFE3,RKFE4,RKFES1,RKFES2,
	RKFEF1,RKFEF2,RKFEF3,RKC1,RKC2,RKW,RKSP,TWKELV,RKHCO2,H,
	PH,ALFREE,ALOH,ALS,ALF,FEFREE,FEOH,FES,FEF,	ALK,ALKW,ALKC,ALKAL,ALKFE,TIC,F1,S1,INUM):

	''' Estimate pH value based on total Al, Fe conc. and Fe(OH)3 control '''

	ITER = 0
	MIN_  = 0
	MAX_  = 0
	hMIN = 0.0
	hMAX = 0.0
	erMIN= 0.0
	erMAX= 0.0
	INUM = 0

	# set initial estimate of [H+]
	IF 1.0E-1 > H > 1.0E-14:   # use previous h
		H = H
	ELSE:   # use ph = 5
		H = 1E-5
	END IF

	# compute free AluMIN_um conc. based on Gibbsite MIN_eral solubility
	# 100  CONTINUE
		INUM   = INUM + 1
		FEFREE = 10.0**RKSP * H**3

		# Al(+3) __ aloh(+2) __ al(oh)2(+1) __ al(oh)3(0) __ al(oh)4(-)
		AAL0 = 1.0 / (1.0 + RKAL1 / H * (1.0 + RKAL2 / H * (1.0 + RKAL3 / H * (1.0 + RKAL4 / H))))
		AAL1 = RKAL1 /H * AAL0
		AAL2 = RKAL2 /H * AAL1
		AAL3 = RKAL3 /H * AAL2
		AAL4 = RKAL4 /H * AAL3

		# Fe(+3) __ feoh(+2) __ fe(oh)2(+1) __ fe(oh)3(0) __ fe(oh)4(-)
		AFE0 = 1.0 / (1.0 + RKFE1 / H * (1.0 + RKFE2 / H * (1.0 + RKFE3 / H * (1.0 + RKFE4 / H))))
		AFE1 = RKFE1 / H * AFE0
		AFE2 = RKFE2 / H * AFE1
		AFE3 = RKFE3 / H * AFE2
		AFE4 = RKFE4 / H * AFE3
		ALTX(1) = TOTAL
		200  CONTINUE
			DO 300 I= 1, 2
				FRTIO1 = 0.0
				FRTIO2 = 0.0
				SRTIO1 = 0.0
				SRTIO2 = 0.0
				ALT    = ALTX(I)

				# compute sulfate complexation
				IF ST > 0.0:
					A      = RKALS2 * ALT * AAL0 + RKFES2 * FEFREE
					B      = 1.0 + RKALS1 * ALT * AAL0 + RKFES1 * FEFREE
					C      = -ST
					S1     = (-B + SQRT(B*B - 4.0 * A * C)) / (2.0 * A)
					SRTIO1 = AAL0 * (RKALS1 * S1 + RKALS2 * S1 * S1)
					SRTIO2 = AFE0 * (RKFES1 * S1 + RKFES2 * S1 * S1)
				END IF

				# compute fluoride complexation
				IF FT > 0.0:
					A      = -FT
					B      = 1.0 + RKALF1 * ALT * AAL0 + RKFEF1 * FEFREE
					C      = RKALF2 * ALT * AAL0 + RKFEF2 * FEFREE
					D      = RKALF3 * ALT * AAL0 + RKFEF3 * FEFREE
					E      = RKALF4 * ALT * AAL0
					F      = RKALF5 * ALT * AAL0
					G      = RKALF6 * ALT * AAL0
					NXX    = 6
					F1     = NEWTON(NXX,FT,A,B,C,D,E,F,G)
					FRTIO1 = AAL0 * (RKALF1 * F1 + RKALF2 *F1**2 + RKALF3 * F1**3 + RKALF4 * F1**4 + RKALF5 * F1**5 + RKALF6 * F1**6)
					FRTIO2 = AFE0 * (RKFEF1 * F1 + RKFEF2 *F1**2 + RKFEF3 * F1**3)
				END IF

				ALTX(I+1) = TOTAL / (1.0 + FRTIO1 + SRTIO1)

				IF ABS((ALTX(I+1) - ALTX(I)) / ALTX(I)) < 0.001:
					GO TO 400

			# 300  CONTINUE

			SLOPE1  = (ALTX(3) - ALTX(2)) / (ALTX(2) - ALTX(1))
			ALTX(1) = ALTX(3) + SLOPE1 / (1.0 - SLOPE1) * (ALTX(3) - ALTX(2))
		# GO TO 200

	# 400  CONTINUE

	FEOH   = (FEFREE / AFE0) - FEFREE
	FEF    = (FEFREE + FEOH) * FRTIO2
	FES    = (FEFREE + FEOH) * SRTIO2
	ALFREE = ALT * AAL0
	ALOH   = ALT - ALFREE
	ALF    = ALT * FRTIO1
	ALS    = ALT * SRTIO1

	# compute total aluMIN_um concentration
	FEtMAX = FEOH + FEF + FES + FEFREE

		ERROR  = FEtMAX - TOTFE
		IF ABS(ERROR) <= 1.0E-8:
			GO TO 600
		IF MIN_ * MAX_ != 1:
			IF ERROR < 0.0:
				hMAX  = H
				erMAX = ERROR
				MAX_   = 1
				IF MIN_ == 1:
					GO TO 500
				H = H * 2.0
			ELSE:
				hMIN  = H
				erMIN = ERROR
				MIN_   = 1
				IF MAX_ == 1:
					GO TO 500
				H = H * 0.5
			END IF
			# GO TO 100
		END IF

		# 500  CONTINUE

		ITER = ITER + 1
		IF ITER <= 1:
			SLOPE  = (erMAX - erMIN) / (hMAX - hMIN)
			H      = hMAX - erMAX / SLOPE
		ELSE:
			H = CONV(hMIN,hMAX,erMIN,erMAX,ERROR,H)
		ENDIF

		# GO TO 100

	# 600  CONTINUE

	PH = -LOG10(H)

	# H2c03(*) __ hco3(-) __ co3(-2)
	AC0  = 1.0 / (1.0 + RKC1 / H * (1.0 + RKC2 / H))
	AC1  = RKC1 / H * AC0
	AC2  = RKC2 / H * AC1
	TIC  = CO2 * (273.15 / TWKELV) * RKHCO2 / AC0

	# compute components of alkalinity CO3--AL--H2O
	ALKC  = (AC1 + 2.0 * AC2) * TIC
	ALKAL = (4.0 * AAL4 + 3.0 * AAL3 + 2.0 * AAL2 + AAL1) * (ALOH + ALFREE)
	ALKFE = (4.0 * AFE4 + 3.0 * AFE3 + 2.0 * AFE2 + AFE1) * (FEOH + FEFREE)
	ALKW  = RKW / H - H
	ALK   = ALKC + ALKW + ALKAL + ALKFE

	RETURN  H,PH,ALFREE,ALOH,ALS,ALF,FEFREE,FEOH,FES,FEF,ALK,ALKW,ALKC,ALKAL,ALKFE,TIC,F1,S1,INUM)
	END



DEF ACCAL3 (TOTAL,TOTFE,ST,FT,ALK,CO2,RKAL1,RKAL2,RKAL3,RKAL4,RKALS1,RKALS2,
	RKALF1,RKALF2,RKALF3,RKALF4,RKALF5,RKALF6,RKFE1,RKFE2,RKFE3,RKFE4,RKFES1,RKFES2,
	RKFEF1,RKFEF2,RKFEF3,RKC1,RKC2,RKW,TWKELV,RKHCO2,H,
	PH,ALFREE,ALOH,ALS,ALF,FEFREE,FEOH,FES,FEF,ALKW,ALKC,ALKAL,ALKFE,TIC,F1,S1,INUM)

	''' Estimate pH value based on total Al and Fe concs. and alkalinity'''

	# initialize parameters used in pH deterMINing algorithm
	ITER   = 0
	MIN_    = 0
	MAX_    = 0
	hMIN   = 0.0
	hMAX   = 0.0
	erMIN  = 0.0
	erMAX  = 0.0
	INUM   = 0

	# set initial estimate of [H+]
	IF 1.0E-1 > H > 1.0E-14:
		# use previous ph
	ELSE:   # use ph = 5
        H = 1E-5
	END IF

	# 100  CONTINUE

	INUM = INUM + 1

	# Al(+3) __ aloh(+2) __ al(oh)2(+1) __ al(oh)3(0) __ al(oh)4(-)
	AAL0 = 1.E0/(1.E0 + RKAL1/H*(1.E0 + RKAL2/H*(1.E0 + RKAL3/H*(1.E0 + RKAL4/H))))
	AAL1 = RKAL1 / H * AAL0
	AAL2 = RKAL2 / H * AAL1
	AAL3 = RKAL3 / H * AAL2
	AAL4 = RKAL4 / H * AAL3

	# Fe(+3) __ feoh(+2) __ fe(oh)2(+1) __ fe(oh)3(0) __ fe(oh)4(-)
	AFE0 = 1.D0/(1.D0 + RKFE1/H*(1.D0 + RKFE2/H*(1.D0 + RKFE3/H* (1.D0 + RKFE4/H))))
	AFE1 = RKFE1 / H * AFE0
	AFE2 = RKFE2 / H * AFE1
	AFE3 = RKFE3 / H * AFE2
	AFE4 = RKFE4 / H * AFE3

	ALTX(1) = TOTAL
	FETX(1) = TOTFE

	# 200  CONTINUE
		DO 300 I= 1, 2
			FRTIO1 = 0.0
			FRTIO2 = 0.0
			SRTIO1 = 0.0
			SRTIO2 = 0.0
			ALT1   = ALTX(I)
			FET1   = FETX(I)

			# compute sulfate complexation
			IF ST > 0.0:
				A      = RKALS2 * ALT1 * AAL0 + RKFES2 * FET1 * AFE0
				B      = 1.0 + RKALS1 * ALT1 * AAL0 + RKFES1 * FET1 * AFE0
				C      = -ST
				S1     = (-B + SQRT(B*B - 4.0 * A * C)) / (2.0 * A)
				SRTIO1 = AAL0 * (RKALS1 * S1 + RKALS2 * S1 * S1)
				SRTIO2 = AFE0 * (RKFES1 * S1 + RKFES2 * S1 * S1)
			END IF

			# compute fluoride complexation
			IF FT > 0.0:
				A      = -FT
				B      = 1.0 + RKALF1 * ALT1 * AAL0 + RKFEF1 * FET1 * AFE0
				C      = RKALF2 * ALT1 * AAL0 + RKFEF2 *FET1 * AFE0
				D      = RKALF3 * ALT1 * AAL0 + RKFEF3 *FET1 * AFE0
				E      = RKALF4 * ALT1 * AAL0
				F      = RKALF5 * ALT1 * AAL0
				G      = RKALF6 * ALT1 * AAL0
				NXX    = 6
				F1     = NEWTON(NXX,FT,A,B,C,D,E,F,G)
				FRTIO1 = AAL0 * (RKALF1 * F1 + RKALF2 * F1**2 + RKALF3 * F1**3 + RKALF4 * F1**4 + RKALF5 * F1**5 + RKALF6 * F1**6)
				FRTIO2 = AFE0 * (RKFEF1 * F1 + RKFEF2 * F1**2 + RKFEF3 * F1**3)
			ENDIF

			ALTX(I+1) = TOTAL / (1.0 + FRTIO1 + SRTIO1)
			FETX(I+1) = TOTFE / (1.0 + FRTIO2 + SRTIO2)

			IF ABS((ALTX(I+1) - ALTX(I))  /ALTX(I)) < 0.001:
				IF TOTFE <= 0.0:
					GO TO 400
				IF ABS((FETX(I+1) - FETX(I)) / FETX(I) < 0.001:
					GO TO 400
			END IF
		# 300  CONTINUE

		SLOPE1 = (ALTX(3) - ALTX(2)) / (ALTX(2) - ALTX(1))

		IF TOTFE > 0.0:
			SLOPE2 = (FETX(3) - FETX(2)) / (FETX(2) - FETX(1))
		END IF

		ALTX(1) = ALTX(3) + SLOPE1 / (1.0 - SLOPE1) * (ALTX(3) - ALTX(2))
		IF (TOTFE > 0.0:
			FETX(1) = FETX(3) + SLOPE2 / (1.0 - SLOPE2) * (FETX(3) - FETX(2))
		END IF
	# GO TO 200

	400  CONTINUE

	# H2c03(*) __ hco3(-) __ co3(-2)

	AC0  = 1.D / (1.0 + RKC1 / H * (1.0 + RKC2 / H))
	AC1  = RKC1 / H * AC0
	AC2  = RKC2 / H * AC1
	TIC  = CO2 * (273.15 / TWKELV) * RKHCO2 / AC0

	# compute components of alkalinity CO3--AL--H2O
	ALKC  = (AC1 + 2.0 * AC2) * TIC
	ALKAL = (4.0 * AAL4 + 3.0 * AAL3 + 2.0 * AAL2 + AAL1) * ALT1
	ALKFE = (4.0 * AFE4 + 3.0 * AFE3 + 2.0 * AFE2 + AFE1) * FET1
	ALKW  = RKW / H - H
	ALKT  = ALKC + ALKAL + ALKW + ALKFE

	ERROR = ALKT - ALK
	IF ABS(ERROR) < 1.0E-8
		GO TO 600
	IF MIN_ * MAX_ != 1:
		IF ERROR < 0.0:     # under-estimate, set upper limit, set MAX_ = 1
			hMAX  = H
			erMAX = ERROR
			MAX_   = 1
			IF MIN_ == 1:
				GO TO 500
			H = H * 0.5
        ELSE:                 # over-estimate, set lower limit, set MIN_ = 1
			hMIN  = H
			erMIN = ERROR
			MIN_   = 1
			IF MAX_ == 1:
				GO TO 500
			H = H * 2.0
        END IF
        # GO TO 100
	ENDIF

	# 500  CONTINUE
	ITER  = ITER + 1
	IF ITER <= 1:
		SLOPE = (erMAX - erMIN)/(hMAX - hMIN)
		H     = hMAX - erMAX / SLOPE
	ELSE:
		h = CONV(hMIN,hMAX,erMIN,erMAX,ERROR, H)
	ENDIF

	# GO TO 100

	# 600  CONTINUE

	PH     = -LOG10(H)
	FEFREE = FET1 * AFE0
	FEOH   = FET1 - FEFREE
	FEF    = FET1 * FRTIO2
	FES    = FET1 * SRTIO2
	ALFREE = ALT1 * AAL0
	ALOH   = ALT1 - ALFREE
	ALF    = ALT1 * FRTIO1
	ALS    = ALT1 * SRTIO1

	RETURN H,PH,ALFREE,ALOH,ALS,ALF,FEFREE,FEOH,FES,FEF,ALKW,ALKC,ALKAL,ALKFE,TIC,F1,S1,INUM
	END


def accal4 (total,totfe,totmn,totac,cond, rmlwal,rmlwfe,rmlwmn,rcoefm,rcoefh,
	h, ph,newal,newfe,newmn,newac)
	'''estimates ph value based on total al, final ferric iron conc. (fe 3+), manganese (mn 2+)'''

	# initialize constants used in ph deterMINing algorithm
	ivalfe  = 3
	ivalal  = 3
	ivalmn  = 2
	rfacta  = 0.5029
	rfactd  = 10.0**(-3.0)
	rfactb  = 0.014 * rfactd
	rfactc  = 0.24
	rstdpot = -0.615
	rpartps = 0.21
	rksp1   = 10**(-33.00)
	rksp2   = 10**(-38.46)

	# set initial estimate of [h+]
	if h < 10.0e-2:      # use previous value
		rih = h
	else:                 # use ph=6.5
		rih = 10.0**(-6.5)
	riph = -log10(h)   # calculate initial ph 

	# compute free aluMIN_um conc. 
	rion = cond * rfactb
	rkpow  = -rfacta * (ivalal**2.0) * ((sqrt(rion) / (1.0 + sqrt(rion))) - rfactc * rion)
	rgamma = 10.0**rkpow
	racth  = h
	ractoh = 10.0**(-14) / racth

	ral = (rksp1 / (rgamma * ractoh**3.0)) * total

	if rih < 10.0-7:
		tmpal = total * (1.0 - rcoefm)
		rcal = tmpal
	else
		tmpalr = total * rgamma
		if tmpalr >= ral:
			tmpal = ral
			rcal = total - tmpalr
		else:
			rcal = 0.0
	newal = total - rcal

	# compute ferric iron concentration
	rkpow  = -rfacta * (ivalfe**2.0) * ((sqrt(rion) / (1.0 + sqrt(rion))) - rfactc * rion)
	rgamma = 10.0**rkpow
	racth  = h
	ractoh = 10.0**(-14) / racth
	rfe = (rksp2 / (rgamma * ractoh**3.0)) * totfe
	if rih < 10.0e-7:
		tmpfe = totfe * (1.0 - rcoefm)
		rcfe  = tmpfe
	else:
		tmpfer = totfe * rgamma
		if totfe >= rfe:
			tmpfe = rfe
			rcfe = totfe - tmpfer
		else:
			rcfe = 0.
	newfe = totfe - rcfe

	# compute manganese concentration
	rkpow = -rfacta * (ivalmn**2.0) * ((sqrt(rion) / (1.0 + sqrt(rion))) - rfactc * rion)
	rgamma = 10.0**rkpow
	racth  = h 
	rk     = 10.0**(2.e0 * rstdpot / rfacta)
	rmn    = (rk * (racth**2.0)) / (rgamma * (rpartps**(0.5))) 
	if rih < 10.0e-7:
		tmpmn = totmn * (1.0 - rcoefm)
		rcmn = tmpmn
	else:
		tmpmnr = totmn * rgamma
		if (totmn .ge. rmn) then
			tmpmn = rmn
			rcmn = totmn - tmpmnr
		else
			rcmn = 0.
	newmn = totmn - rcmn

	# recalculate ph
	rhpow = log10(rih)
	racth = 10.0**rhpow
	rkpow = -rfacta * ((sqrt(rion) / (1.0 + sqrt(rion))) - rfactc * rion)
	# use the change in conc. values
	rgamma = 10.0**rkpow
	tmpmc = ((3.0 * rcfe) + (3.0 * rcal) + (2.0 * rcmn)) *  rgamma * (1.0 - ((riph + rcoefh) / 10.0))

	ph = - log10(racth + tmpmc)
	h = 10.0**(-ph)

	# compute acidity

	# use the initial conc. of the metals
	rac = (3.0 * totfe) + (3.0 * total) +  (2.0 * totmn) +  10.0**(-riph)

	# use the change in conc. of the metals
	rcac = (3.0 * rcfe) + (3.0 * rcal) + (2.0 * rcmn) + 10.0**(-ph)

	if ph > 7.0:
		newac = 0.0
	else: 
		if ph < 2.0:
			# use the new acidity calculated from the recalculated conc. of metals
			newac = rcac
		else
			newac = (rcac * totac) / rac
	return h, ph,newal,newfe,newmn,newac)


def conv(hMIN, hMAX, erMIN, erMAX, error, h)
	''' convergence algorithm for acid ph module'''
	MIN_ = 0
	MAX_ = 0
	if (error < 0.0) 
		MIN_ = 1
	elif (error > 0.0)
		MAX_ = 1
		
	sMIN = (erMIN - error) / (hMIN - h)
	sMAX = (erMAX - error) / (hMAX - h)
	xMIN = hMIN - erMIN / sMIN
	xMAX = hMAX - erMAX / sMAX
	tMAX = abs(h - hMIN)
	tMIN = abs(h - hMAX)

	if MIN_ == 1:
		erMIN = error
		hMIN  = h
	if MAX__ == 1:
		erAX = error
		hMAX  = h
	if (xMAX - hMAX) * (xMAX - hMIN) >= 0.0:
		xMAX = hMIN
		tMAX = 1.0
		tMIN = 2.0
	elif (xMIN - hMAX) * (xMIN - hMIN) >= 0.0:
		xMIN = hMAX
		tMIN = 1.0
		tMAX = 2.0
	h = (tMAX * xMAX + tMIN * xMIN) / (tMAX + tMIN)
	return h


def newton(m, xMAX, a, b, c, d, e, f, g)
	'''performs newton-raphson solution for acid ph module'''
	xton = zeros(6)
	x = xMAX / 2.0
	while abs((x - x0) / x0) > .001:
		x0      = x
		xton[0] = x0
		for i in range(2,m)   # do 30 i= 2, m
			xton[i] = xton[i-1] * x0
		# 30     continue

		y0    = a +       b * xton[0] +       c * xton[1] +       d * xton[2] +       e * xton[3] +       f * xton[4] + g * xton[5]
		slope = b + 2.0 * c * xton[0] + 3.0 * d * xton[1] + 4.0 * e * xton[2] + 5.0 * f * xton[3] + 6.0 * g * xton[4]
		x     = x0 - y0 / slope
	return x

