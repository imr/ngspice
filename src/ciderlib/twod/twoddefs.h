/*
 * 2001 Paolo Nenzi
 */
 
#ifndef ngspice_TWODDEFS_H
#define ngspice_TWODDEFS_H

/* Debug statements */

extern BOOLEAN TWOacDebug;
extern BOOLEAN TWOdcDebug;
extern BOOLEAN TWOtranDebug;
extern BOOLEAN TWOjacDebug;

/* Now some defines for the two dimensional simulator
 * library.
 * Theese defines were gathered from all the code in
 * oned directory.
 */


/* Temporary hack to remove NUMOS gate special case */
#ifdef NORMAL_GATE
#define GateTypeAdmittance oxideAdmittance
#else
#define GateTypeAdmittance contactAdmittance
#endif				/* NORMAL_GATE */ 

/* Temporary hack to remove NUMOS gate special case */
#ifdef NORMAL_GATE
#define GateTypeConductance oxideConductance
#define GateTypeCurrent oxideCurrent
#else
#define GateTypeConductance contactConductance
#define GateTypeCurrent contactCurrent
#endif /* NORMAL_GATE */ 
 
 /* This structure was moved up from twoadmit.c */
struct mosAdmittances {
    SPcomplex yIdVdb;
    SPcomplex yIdVsb;
    SPcomplex yIdVgb;
    SPcomplex yIsVdb;
    SPcomplex yIsVsb;
    SPcomplex yIsVgb;
    SPcomplex yIgVdb;
    SPcomplex yIgVsb;
    SPcomplex yIgVgb;
};

#define MAXTERMINALS 5  /* One more than max number of terminals */
#define ELCT_ID poiEqn

#define MIN_DELV 1e-3
#define NORM_RED_MAXITERS 10

#endif
