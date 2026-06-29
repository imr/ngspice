.SUBCKT MCP6041 1 2 3 4 5
*               | | | | |
*               | | | | Output
*               | | | Negative Supply
*               | | Positive Supply
*               | Inverting Input
*               Non-inverting Input
*
********************************************************************************
* Software License Agreement                                                   *
*                                                                              *
* The software supplied herewith by Microchip Technology Incorporated (the     *
* 'Company') is intended and supplied to you, the Company's customer, for use  *
* soley and exclusively on Microchip products.                                 *
*                                                                              *
* The software is owned by the Company and/or its supplier, and is protected   *
* under applicable copyright laws. All rights are reserved. Any use in         *
* violation of the foregoing restrictions may subject the user to criminal     *
* sanctions under applicable laws, as well as to civil liability for the       *
* breach of the terms and conditions of this license.                          *
*                                                                              *
* THIS SOFTWARE IS PROVIDED IN AN 'AS IS' CONDITION. NO WARRANTIES, WHETHER    *
* EXPRESS, IMPLIED OR STATUTORY, INCLUDING, BUT NOT LIMITED TO, IMPLIED        *
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE APPLY TO  *
* THIS SOFTWARE. THE COMPANY SHALL NOT, IN ANY CIRCUMSTANCES, BE LIABLE FOR    *
* SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES, FOR ANY REASON WHATSOEVER.     *
********************************************************************************
*
* The following op-amps are covered by this model:
*      MCP6041,MCP6042,MCP6043,MCP6044
*
* Revision History:
*      REV A: 07-Sep-01, Created model
*      REV B: 27-Aug-06, Added over temperature, improved output stage, 
*                        fixed overdrive recovery time
*      REV C: 09-Apr-07, Adjusted quiescent current to match spec
*      REV D: 27-Jul-07, Modified output impedance at expense of comparator operation
*                        to correct transient response with capacitive load
*
* Recommendations:
*      Use PSPICE (other simulators may require translation)
*      For a quick, effective design, use a combination of: data sheet
*            specs, bench testing, and simulations with this macromodel
*      For high impedance circuits, set GMIN=100F in the .OPTIONS statement
*
* Supported:
*      Typical performance for temperature range (-40 to 125) degrees Celsius
*      DC, AC, Transient, and Noise analyses.
*      Most specs, including: offsets, DC PSRR, DC CMRR, input impedance,
*            open loop gain, voltage ranges, supply current, ... , etc.
*      Temperature effects for Ibias, Iquiescent, Iout short circuit 
*            current, Vsat on both rails, Slew Rate vs. Temp and P.S.
*
* Not Supported:
*      Chip select (MCP6043)
*      Some Variation in specs vs. Power Supply Voltage
*      Monte Carlo (Vos, Ib), Process variation
*      Distortion (detailed non-linear behavior)
*      Behavior outside normal operating region
*
* Input Stage
V10  3 10 -500M
R10 10 11 69k
R11 10 12 69k
C12  1  0 6P
C11 11 12 95P
E12 71 14 POLY(6) 20 0 21 0 22 0 23 0 26 0 27 0 2.00M 10 10 29 29 1 1
G12 1 0 62 0 1m
M12 11 14 15 15 NMI
G13 1 2 62 0 20u 
M14 12 2 15 15 NMI 
G14 2 0 62 0 1m
C14  2  0 6P
I15 15 4 4U
V16 16 4 -300M
GD16 16 1 TABLE {V(16,1)} ((-100,-1p)(0,0)(1m,1u)(2m,1m)) 
V13 3 13 -300M
GD13 2 13 TABLE {V(2,13)} ((-100,-1p)(0,0)(1m,1u)(2m,1m)) 
R71  1  0 20.0E12
R72  2  0 20.0E12
R73  1  2 20.0E12
I80  1  2 500E-15
*
* Noise, PSRR, and CMRR
I20 21 20 423U
D20 20  0 DN1
D21  0 21 DN1
I22 22 23 1N
R22 22 0  1k
R23  0 23 1k
G26  0 26 POLY(2) 3 0 4 0   0.00 -79.4U -39.8U
R26 26  0 1
G27  0 27 POLY(2) 1 0 2 0 0 26u 26u
R27 27  0 1
*
* Open Loop Gain, Slew Rate
G30  0 30 12 11 3.2
R30 30  0 1.00K
I31 0 31 DC 338
R31 31  0 1 TC=2.25M,-15U
GD31 30 0 TABLE {V(30,31)} ((-100,-1n)(0,0)(1m,0.1)(2m,2))
I32 32 0 DC 535
R32 32  0 1 TC=2.02M,-11U
GD32 0 30 TABLE {V(30,32)} ((-2m,2)(-1m,0.1)(0,0)(100,-1n))
G33  0 33 30 0 1m
R33  33 0 3K
G34  0 34 33 0 1
R34  34 0 1K
C34  34 0 100M
G37  0 341 34 0 1m
R341  341 0 1k
C341  341 0 1.3N
G371  0 37 341 0 1m
R37  37 0 1K
C37  37 0 3N
G38  0 38 37 0 1m
R38  39 0 1K
L38  38 39 13M
E38  35 0 38 0 1
G35 33 0 TABLE {V(35,3)} ((-1,-1n)(0,0)(3.4k,1n))(3.5k,1))
G36 33 0 TABLE {V(35,4)} ((-3.5k,-1)((-3.4k,-1n)(0,0)(1,1n))
*
* Output Stage
R80 50 0 100MEG
G50 0 50 57 96 2
R58 57  96 0.50
R57 57  0 101k
C58  5  0 2.00P
G57  0 57 POLY(3) 3 0 4 0 35 0 0 10U 1.49U 9.1U
GD55 55 57 TABLE {V(55,57)} ((-2m,-1)(-1m,-1m)(0,0)(10,1n))
GD56 57 56 TABLE {V(57,56)} ((-2m,-1)(-1m,-1m)(0,0)(10,1n))
E55 55  0 POLY(2) 3 0 51 0 -0.7M 1 -40M
E56 56  0 POLY(2) 4 0 52 0 0.6M 1 -55M
R51 51 0 1k
R52 52 0 1k
GD51 50 51 TABLE {V(50,51)} ((-10,-1n)(0,0)(1m,1m)(2m,1))
GD52 50 52 TABLE {V(50,52)}  ((-2m,-1)(-1m,-1m)(0,0)(10,1n))
G53  3  0 POLY(1) 51 0 -4U 1M
G54  0  4 POLY(1) 52 0 -4U -1M
*
* Current Limit
G99 96 5 99 0 1
R98 0 98 1 TC=-6.9M
G97 0 98 TABLE { V(96,5) } ((-11.0,-3.9M)(-1.00M,-3.87M)(0,0)(1.00M,3.23M)(11.0,3.26M))
E97 99 0 VALUE { V(98)*((V(3)-V(4))*1.39 + -1.5)}
D98 4 5 DESD
D99 5 3 DESD
*
* Temperature / Voltage Sensitive IQuiscent
R61 0 61 1 TC=2.52M,-4.31U
G61 3 4 61 0 1
G60 0 61 TABLE {V(3, 4)} 
+ ((0,0)(700M,5.3N)(770M,10.0N)(1.00,480N)
+ (1.5,500N)(3.5,530N)(7.00,580N))
*
* Temperature Sensistive offset voltage
I73 0 70 DC 1uA
R74 0 70 1 TC=1.5
E75 1 71 70 0 1 
*
* Temp Sensistive IBias
I62 0 62 DC 1uA
R62 0 62 REXP  210U
*
* Models
.MODEL NMI NMOS(L=2.00U W=42.0U KP=20.0U LEVEL=1 )
.MODEL DESD  D   N=1 IS=1.00E-15
.MODEL DN1 D   IS=1P KF=0.2F AF=1
.MODEL REXP RES TCE= 9
.ENDS MCP6041
