/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 1999 Paolo Nenzi
**********/
/*
 */
#ifndef ngspice_TRCVDEFS_H
#define ngspice_TRCVDEFS_H


#include "ngspice/jobdefs.h"
#include "ngspice/tskdefs.h"
#include "ngspice/gendefs.h"
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

enum {
    DCT_START1 = 1,
    DCT_STOP1,
    DCT_STEP1,
    DCT_NAME1,
    DCT_TYPE1,
    DCT_START2,
    DCT_STOP2,
    DCT_STEP2,
    DCT_NAME2,
    DCT_TYPE2,
};

#endif
