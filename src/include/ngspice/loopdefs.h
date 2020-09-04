/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jaijeet S Roychowdhury
Modified: 2000 AlansFixes

Copyright 2020 Anamosic Ballenegger Design.  All rights reserved.
Author: 2020 Florian Ballenegger

**********/

#ifndef ngspice_LOOPDEF_H
#define ngspice_LOOPDEF_H

#ifdef D_DBG_ALLTIMES
#define D_DBG_BLOCKTIMES
#define D_DBG_SMALLTIMES
#endif

#include "ngspice/jobdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/cktdefs.h"

/* function prototypes that usually go into cktdefs.h */
extern int LOOPsetParm(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value);
extern int LOOPaskQuest(CKTcircuit *ckt, JOB *anal, int which, IFvalue *value);
extern int LOOPpreset(CKTcircuit *ckt, JOB *anal);
extern int LOOPan(CKTcircuit *ckt, int restart);


    /* structure used to describe an LOOP analysis to be performed */
    /* cktnewan.c is patched to initialiaze this data structure to all zero */
typedef struct {
    int JOBtype;
    JOB *JOBnextJob;		/* pointer to next thing to do */
    char *JOBname;		/* name of this job */
    double LOOPstartFreq;		/* the start value of
                                   frequency for loop stability analysis */
    double LOOPstopFreq;		/* the stop value ove above */
    double LOOPfreqDelta;		/* multiplier for decade/octave
                                   stepping, step for linear steps. */
    double LOOPsaveFreq;		/* frequency at which we left off last time*/
    int    LOOPstepType;		/* values described below */
    int    LOOPnumSteps;
    double LOOPomega;		/* current omega1 */
    IFuid  LOOPprobeSrc;	        /* probe source */
    char*  LOOPname;
    char*  LOOPportname;
    int    LOOPportnum;
    int	   LOOPdirection;  /* 0: not set, 1: forward, 2:reverse */
    IFuid  LOOPinSrc;
    IFuid  LOOPoutSrc;
    CKTnode *LOOPoutPos;
    CKTnode *LOOPoutNeg;

    double** LoopGain;
    double** iLoopGain;
    
    unsigned LOOPnameGiven : 1;
    unsigned LOOPportnumGiven : 1;
    unsigned LOOPportnameGiven : 1;
    unsigned LOOPinSrcGiven : 1;
    unsigned LOOPoutSrcGiven : 1;
    unsigned LOOPoutPosGiven : 1;
    unsigned LOOPoutNegGiven : 1;
} LOOPAN;

/* available step types: */

#define DECADE 1
#define OCTAVE 2
#define LINEAR 3

/* defns. used in DsetParm */

#define LOOP_DEC 1
#define LOOP_OCT 2
#define LOOP_LIN 3
#define LOOP_START 4
#define LOOP_STOP 5
#define LOOP_STEPS 6

#define LOOP_PROBESRC 10
#define LOOP_PORTNAME 11
#define LOOP_PORTNUM 12
#define LOOP_DIR 13
#define LOOP_PROBESRCNEG 15
#define LOOP_PORTNAMENEG 16
#define LOOP_PORTNUMNEG 17
#define LOOP_DIRNEG 18

#define LOOP_INSRC 20
#define LOOP_OUTSRC 21
#define LOOP_OUTPOS 22
#define LOOP_OUTNEG 23

#define LOOP_NAME 30


#endif
