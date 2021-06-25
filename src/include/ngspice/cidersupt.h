/*
 * cidersupt.h
 * 
 * CIDER support library header
 */
 
#ifndef ngspice_CIDERSUPT_H
#define ngspice_CIDERSUPT_H

#include <stdio.h>

#include "ngspice/bool.h"
#include "ngspice/gendev.h"
#include "ngspice/material.h"
#include "ngspice/numglobs.h"
#include "ngspice/profile.h"

/* externals for database.c */
extern struct plot* DBread( char *);
extern double *DBgetData( struct plot *, char *, int );
extern void DBfree( struct plot *);

/* externals for devprint.c */
extern void printVoltages(FILE *, char *, char *, int, 
                          int, double, double, double, 
	                      double, double, double );

/* externals for geominfo.c */
extern void printCoordInfo(CoordInfo *);
extern void killCoordInfo(CoordInfo *);
extern void ONEprintDomainInfo(DomainInfo *);
extern void TWOprintDomainInfo(DomainInfo *);
extern void killDomainInfo(DomainInfo *);
extern void ONEprintBoundaryInfo(BoundaryInfo *);
extern void TWOprintBoundaryInfo(BoundaryInfo *);
extern void killBoundaryInfo(BoundaryInfo *);
extern void TWOprintElectrodeInfo(ElectrodeInfo *);
extern void killElectrodeInfo(ElectrodeInfo *);

/* externals for globals.c */
extern void GLOBcomputeGlobals(GLOBvalues *, double);
extern void GLOBputGlobals(GLOBvalues *);
extern void GLOBgetGlobals(GLOBvalues *);
extern void GLOBprnGlobals(FILE *, GLOBvalues *);

/* externals for integset.c */
extern void computeIntegCoeff(int, int, double *, double *);
extern void computePredCoeff(int, int, double *, double *);

/* externals for integuse.c */
extern double integrate(double **, TranInfo *, int );
extern double predict(double **, TranInfo *, int );
extern double computeLTECoeff( TranInfo * );
extern double integrateLin(double **, TranInfo *, int );

/* externals for logfile.c */
extern void LOGmakeEntry(char *, char * );

/* externals for mater.c */
extern void  MATLdefaults(MaterialInfo * );
extern void  MATLtempDep(MaterialInfo *, double );
extern void  printMaterialInfo(MaterialInfo * );

/* externals for mobil.c */
extern void MOBdefaults(MaterialInfo *, int, int, int, int );
extern void MOBtempDep(MaterialInfo *, double );
extern void MOBconcDep(MaterialInfo *, double, double *, double *);
extern void MOBfieldDep(MaterialInfo *, int, double, double *, double * );

/* externals for recomb.c */
extern void recomb(double, double, double, double, double, double, 
                    double, double *, double *, double * );

/* externals for suprem.c */
extern int readAsciiData(const char *, int, DOPtable **);
extern int readSupremData(const char *, int, int, DOPtable ** );

/* externals for suprmitf.c */
extern int SUPbinRead(const char *, float *, float *, int *, int *);
extern int SUPascRead(const char *, float *, float *, int *, int *);




/* externals for misc.c */
extern double guessNewConc(double , double );
extern double lookup(double **, double );
extern BOOLEAN hasSORConverged(double *, double *, int);
extern BOOLEAN foundError(int );
extern BOOLEAN compareFiletypeVar(char *);



#endif
