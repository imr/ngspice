/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 1999 Paolo Nenzi
**********/
/*
 */
#ifndef DCTR
#define DCTR


#include "jobdefs.h"
#include "tskdefs.h"
#include "gendefs.h"
    /*
     * structures used to describe D.C. transfer curve analyses to
     * be performed.
     */

#define TRCVNESTLEVEL 2 /* depth of nesting of curves - 2 for spice2 */

/* PN: The following define is for temp sweep */
/* Courtesy of: Serban M. Popescu */
#ifndef TEMP_CODE
#define TEMP_CODE 1023
#endif

typedef struct {
    int JOBtype;
    JOB *JOBnextJob;
    char *JOBname;
    double TRCVvStart[TRCVNESTLEVEL];   /* starting voltage/current */
    double TRCVvStop[TRCVNESTLEVEL];    /* ending voltage/current */
    double TRCVvStep[TRCVNESTLEVEL];    /* voltage/current step */
    double TRCVvSave[TRCVNESTLEVEL];    /* voltage of this source BEFORE 
                                         * analysis-to restore when done */
    int TRCVgSave[TRCVNESTLEVEL];    /* dcGiven flag; as with vSave */
    IFuid TRCVvName[TRCVNESTLEVEL];     /* source being varied */
    GENinstance *TRCVvElt[TRCVNESTLEVEL];   /* pointer to source */
    int TRCVvType[TRCVNESTLEVEL];   /* type of element being varied */
    int TRCVset[TRCVNESTLEVEL];     /* flag to indicate this nest level used */
    int TRCVnestLevel;      /* number of levels of nesting called for */
    int TRCVnestState;      /* iteration state during pause */
} TRCV;

#define DCT_START1 1
#define DCT_STOP1 2
#define DCT_STEP1 3
#define DCT_NAME1 4
#define DCT_TYPE1 5
#define DCT_START2 6
#define DCT_STOP2 7
#define DCT_STEP2 8
#define DCT_NAME2 9
#define DCT_TYPE2 10

#endif /*DCTR*/
