/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/
/*
 * SENdefs.h - structures for sensitivity package
 */

#ifndef ngspice_SEN2DEFS_H
#define ngspice_SEN2DEFS_H


#include "ngspice/smpdefs.h"
#include "ngspice/jobdefs.h"
#include "ngspice/typedefs.h"

struct SENstruct {
    int JOBtype;
    JOB *JOBnextJob;    /* pointer to next thing to do */
    char *JOBname;      /* name of this job */

    int SENnumVal;         /* length of the next two arrays */
    char **SENdevices;  /* names of the devices to do sensetivity analysis of */
    char **SENparmNames;/* parameters of the above devices to do analysis wrt */

    unsigned int SENinitflag :1 ;   /* indicates whether sensitivity structure*/
                                    /* is to be initialized */ 
    unsigned int SENicflag :1 ; /* indicates whether initial conditions
                                   are specified for transient analysis */

    unsigned int SENstatus :1;     /* indicates whether perturbation
                          is in progress*/ 
    unsigned int SENacpertflag :1; /* indictes whether the perturbation
                          is to be carried out in ac analysis
                          (is done only for first frequency )*/
    int SENmode;     /* indicates the type of sensitivity analysis
                        reqired: DC, Transient, or AC */
    int SENparms;    /* # of  design parameters  */
    double SENpertfac;    /* perturbation factor (for active
                             devices )*/ 
    double  **SEN_Sap;  /* sensitivity matrix (DC and transient )*/
    double  **SEN_RHS;  /* RHS matrix (real part)
                           contains the sensitivity values after SMPsolve*/
    double  **SEN_iRHS; /* RHS matrix (imag part )
                           contains the sensitivity values after SMPsolve*/
    int SENsize;      /* stores the number of rows of each of the above
            three matrices */
    SMPmatrix  *SEN_Jacmat; /* sensitivity Jacobian matrix, */
    double  *SEN_parmVal;   /* table containing values of design parameters */
    char    **SEN_parmName; /* table containing names of design parameters */

};

/* SENmode */
#define DCSEN  0x1
#define TRANSEN  0x2
#define ACSEN  0x4

#define NORMAL 0
#define PERTURBATION  1
#define OFF 0
#define ON 1


enum {
    SEN_AC = 1,
    SEN_DC,
    SEN_TRAN,
    SEN_DEV,
    SEN_PARM,
};

#endif
