/**********
Copyright 1991 Regents of the University of California.  All rights reserved.
Author:	1987 Kartikeya Mayaram, U. C. Berkeley CAD Group
Author:	1991 David A. Gates, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/numglobs.h"
#include "ngspice/numconst.h"
#include "ngspice/numenum.h"
#include "ngspice/cidersupt.h"

/* Global Variable Declarations 
char *LogFileName = "cider.log";

int BandGapNarrowing;
int TempDepMobility, ConcDepMobility, FieldDepMobility, TransDepMobility;
int SurfaceMobility, MatchingMobility, MobDeriv;
int CCScattering;
int Srh, Auger, ConcDepLifetime, AvalancheGen;
int FreezeOut = FALSE;
int OneCarrier;

int MaxIterations = 100;
int AcAnalysisMethod = DIRECT;

double Temp, RelTemp, Vt, RefPsi;
double EpsNorm, VNorm, NNorm, LNorm, TNorm, JNorm, GNorm, ENorm;
 RefPsi is the potential at Infinity */

/*
 * Compute global values for this device.
 */
void GLOBcomputeGlobals(GLOBvalues *pGlobals, double temp)
/* GLOBvalues *pGlobals Global Parameter Data Structure */
/* double temp Instance Temperature */
{
    double tmp1;
    double mnSi, mpSi;			/* electron and hole conduction mass */
    double eg0;				/* band gap */
    double nc0, nv0;			/* conduction/valence band states */

    /* compute temp. dependent global parameters */
    Temp = temp;
    RelTemp = Temp / 300.0;
    tmp1 = pow( RelTemp, 1.5 );

    Vt = BOLTZMANN_CONSTANT * Temp / CHARGE;
    eg0 = EGAP300_SI + DGAPDT_SI * ( (300.0 * 300.0) / (300.0 + TREF_EG_SI)
	- (Temp * Temp) / (Temp + TREF_EG_SI) );
    mnSi = 1.039 + 5.477e-4 * Temp - 2.326e-7 * Temp * Temp;
    mpSi = 0.262 * log( 0.259 * Temp );
    nc0 = NCV_NOM * pow( mnSi, 1.5 ) * tmp1;
    nv0 = NCV_NOM * pow( mpSi, 1.5 ) * tmp1;
    RefPsi = 0.0;

    /* set up the normalization factors */
    EpsNorm = EPS_SI;
    VNorm = Vt;
    NNorm = sqrt( nc0 ) * sqrt( nv0 );        /* this way no overflow */
    LNorm = sqrt( ( VNorm * EpsNorm ) / ( CHARGE * NNorm ) );
    TNorm = LNorm * LNorm / VNorm;
    JNorm = CHARGE * NNorm * VNorm / LNorm;
    GNorm = JNorm / VNorm;
    ENorm = VNorm / LNorm;

    RefPsi /= VNorm;

    /* Save Globals */
    GLOBputGlobals( pGlobals );
   /*
    * GLOBprnGlobals( stdout, pGlobals );
    */

}

void GLOBputGlobals(GLOBvalues *values)
{
  if ( values == NULL ) {
    fprintf( stderr, "Error: tried to get from NIL GLOBvalues\n");
    exit(-1);
  }

  /* Temperature-related globals */
  values->Temp = Temp;
  values->RelTemp = RelTemp;
  values->Vt = Vt;
  values->RefPsi = RefPsi;

  /* Normalization Factors */
  values->EpsNorm = EpsNorm;
  values->VNorm = VNorm;
  values->NNorm = NNorm;
  values->LNorm = LNorm;
  values->TNorm = TNorm;
  values->JNorm = JNorm;
  values->GNorm = GNorm;
  values->ENorm = ENorm;

  return;
}

/*
 * Reload all globals needed during DEV loading routines
 * and DEV output routines
 */
void GLOBgetGlobals(GLOBvalues *values)
{
  if ( values == NULL ) {
    fprintf( stderr, "Error: tried to get from NIL GLOBvalues\n");
    exit(-1);
  }

  /* Temperature-related globals */
  Temp = values->Temp;
  RelTemp = values->RelTemp;
  Vt = values->Vt;
  RefPsi = values->RefPsi;

  /* Normalization Factors */
  EpsNorm = values->EpsNorm;
  VNorm = values->VNorm;
  NNorm = values->NNorm;
  LNorm = values->LNorm;
  TNorm = values->TNorm;
  JNorm = values->JNorm;
  GNorm = values->GNorm;
  ENorm = values->ENorm;

  return;
}

void GLOBprnGlobals(FILE *file, GLOBvalues *values)
{
  static const char tabformat[] = "%12s: % .4e %-12s\t";
  static const char newformat[] = "%12s: % .4e %-12s\n";

  if ( values == NULL ) {
    fprintf( stderr, "Error: tried to print NIL GLOBvalues\n");
    exit(-1);
  }
  fprintf( file, "*** GLOBAL PARAMETERS AT %g deg K\n", values->Temp );
  fprintf( file, "****** Temperature-Dependent Voltages\n" );
  fprintf( file, tabformat, "Vt", values->Vt, "V" );
  fprintf( file, newformat, "RefPsi", values->RefPsi * values->VNorm, "V" );
  fprintf( file, "****** Normalization Factors\n" );
  fprintf( file, newformat, "EpsNorm", values->EpsNorm, "F/cm" );
  fprintf( file, newformat, "VNorm", values->VNorm, "V" );
  fprintf( file, newformat, "NNorm", values->NNorm, "/cm^3" );
  fprintf( file, newformat, "LNorm", values->LNorm, "cm" );
  fprintf( file, newformat, "TNorm", values->TNorm, "s" );
  fprintf( file, newformat, "JNorm", values->JNorm, "A/cm^2" );
  fprintf( file, newformat, "GNorm", values->GNorm, "A/V" );
  fprintf( file, newformat, "ENorm", values->ENorm, "V/cm" );

  return;
}

static int cider_is_loaded = 0;
void CiderLoaded(int val)
{
   cider_is_loaded += val;
}

int IsCiderLoaded(void)
{
  return cider_is_loaded;
}
