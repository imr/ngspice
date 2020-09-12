/*******************************************************************************
 * Copyright 2020 Florian Ballenegger, Anamosic Ballenegger Design
 *******************************************************************************
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

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
    CKTnode *LOOPrefNode;

    double** LoopGain;
    double** iLoopGain;
    
    unsigned LOOPinIV : 2;
    unsigned LOOPoutIV : 2;
    
    unsigned LOOPnameGiven : 1;
    unsigned LOOPportnumGiven : 1;
    unsigned LOOPportnameGiven : 1;
    unsigned LOOPinSrcGiven : 1;
    unsigned LOOPoutSrcGiven : 1;
    unsigned LOOPoutPosGiven : 1;
    unsigned LOOPoutNegGiven : 1;
    unsigned LOOPrefNodeGiven : 1;
} LOOPAN;

/* available step types: */

#define DECADE 1
#define OCTAVE 2
#define LINEAR 3

/* I or V flags */
#define LOOP_IV_UNSET 0
#define LOOP_IV_VOLTAGE 1
#define LOOP_IV_CURRENT 2

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
#define LOOP_REFNODE 14
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
