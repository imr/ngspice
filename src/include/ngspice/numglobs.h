/**********
Copyright 1991 Regents of the University of California. All rights reserved.
Authors: 1987 Karti Mayaram, 1991 David Gates
**********/

/* Define various flags, constants, and indices */
/* These variables are defined in support/globals.c */

#ifndef ngspice_NUMGLOBS_H
#define ngspice_NUMGLOBS_H

extern int BandGapNarrowing;
extern int TempDepMobility;
extern int ConcDepMobility;
extern int FieldDepMobility;
extern int TransDepMobility;
extern int SurfaceMobility;
extern int MatchingMobility;
extern int MobDeriv;
extern int CCScattering;
extern int Srh;
extern int Auger;
extern int ConcDepLifetime;
extern int AvalancheGen;
extern int FreezeOut;
extern int OneCarrier;
extern int MaxIterations;
extern int AcAnalysisMethod;

extern double Temp;
extern double RelTemp;
extern double Vt;
extern double RefPsi;
extern double EpsNorm;
extern double VNorm;
extern double NNorm;
extern double LNorm;
extern double TNorm;
extern double JNorm;
extern double GNorm;
extern double ENorm;

typedef struct sGLOBvalues
{
  double Temp;
  double RelTemp;
  double Vt;
  double RefPsi;
  double EpsNorm;
  double VNorm;
  double NNorm;
  double LNorm;
  double TNorm;
  double JNorm;
  double GNorm;
  double ENorm;
} GLOBvalues;

#endif
