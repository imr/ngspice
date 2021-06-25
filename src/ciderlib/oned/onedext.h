/*
 * 2001 Paolo Nenzi
 */
 
 /* External symbols for ONE Dimensional simulator */

#ifndef ngspice_ONEDEXT_H
#define ngspice_ONEDEXT_H

#include "ngspice/profile.h"
#include "ngspice/onemesh.h"
#include "ngspice/onedev.h"
#include "ngspice/carddefs.h"
#include "ngspice/bool.h"
#include "ngspice/complex.h"
 
/* oneadmit.c */
extern int NUMDadmittance(ONEdevice *, double, SPcomplex *);
extern int NBJTadmittance(ONEdevice *, double, SPcomplex *,
                          SPcomplex *, SPcomplex *, 
			  SPcomplex *);
extern BOOLEAN ONEsorSolve(ONEdevice *, double *, double  *, double omega);
extern void NUMDys(ONEdevice *, SPcomplex *, SPcomplex *);
extern void NBJTys(ONEdevice *,SPcomplex *, SPcomplex *, SPcomplex *, 
                   SPcomplex *, SPcomplex *);
		   
extern SPcomplex *computeAdmittance(ONEnode *, BOOLEAN, double *xReal, 
                                    double *,SPcomplex *);
				    
/* oneaval.c */
extern double ONEavalanche(BOOLEAN, ONEdevice *, ONEnode *);

/* onecond.c */
extern void NUMDconductance(ONEdevice *, BOOLEAN, double *, double *);
extern void NBJTconductance(ONEdevice *, BOOLEAN, double *, double *, 
                            double *, double *, double *);
extern void NUMDcurrent(ONEdevice *, BOOLEAN, double *, double *);
extern void NBJTcurrent(ONEdevice *, BOOLEAN, double *, double *, 
                        double *);
			

/* onecont */
extern void ONE_jacBuild(ONEdevice *);
extern void ONE_sysLoad(ONEdevice *, BOOLEAN, ONEtranInfo *);
extern void ONE_jacLoad(ONEdevice *);
extern void ONE_rhsLoad(ONEdevice *, BOOLEAN, ONEtranInfo *);
extern void ONE_commonTerms(ONEdevice *, BOOLEAN, BOOLEAN, ONEtranInfo *);


/* onedest */
extern void ONEdestroy(ONEdevice *);
 
/* onedopng.c */
extern double ONEdopingValue(DOPprofile *, DOPtable *, double );
extern void ONEsetDoping(ONEdevice *, DOPprofile *, DOPtable *);

/* onefreez.c */
extern void ONEQfreezeOut(ONEnode *, double *, double *, double *, 
                          double *);
extern void ONE_freezeOut(ONEnode *, double, double, double *, 
                          double*, double *, double *);
			  

/* onemesh.c */
extern void ONEbuildMesh(ONEdevice *, ONEcoord *, ONEdomain *, 
                         ONEmaterial *);
extern void ONEgetStatePointers(ONEdevice *, int *);
extern void adjustBaseContact(ONEdevice *, int, int);
extern void NBJTjunctions(ONEdevice *, int *, int *);
extern void ONEprnMesh(ONEdevice *);

/* onepoiss.c */
extern void ONEQjacBuild(ONEdevice *);
extern void ONEQsysLoad(ONEdevice *);
extern void ONEQrhsLoad(ONEdevice *);
extern void ONEQcommonTerms(ONEdevice *);

/*oneprint.c */
extern void ONEprnSolution(FILE *, ONEdevice *, OUTPcard *, BOOLEAN, char *);
extern void ONEmemStats(FILE *, ONEdevice *);
extern void ONEcpuStats(FILE *f, ONEdevice *);


/* oneproj.c */
extern void NUMDproject(ONEdevice *, double);
extern void NBJTproject(ONEdevice *, double, double, double);
extern void NUMDupdate(ONEdevice *,double, BOOLEAN);
extern void NBJTupdate(ONEdevice *, double, double, double, BOOLEAN);
extern void NUMDsetBCs(ONEdevice *, double);
extern void NBJTsetBCs(ONEdevice *, double, double);

/* oneread.c */
extern int ONEreadState(ONEdevice *, char *, int, double *, double *);

/*onesetup.c */
extern void ONEsetup(ONEdevice *);
extern void ONEsetBCparams(ONEdevice *, BDRYcard *, CONTcard *);
extern void ONEnormalize(ONEdevice *);

/* onesolve.c */
extern void ONEdcSolve(ONEdevice *, int, BOOLEAN, BOOLEAN, ONEtranInfo *);
extern BOOLEAN ONEpsiDeltaConverged(ONEdevice *, int *);
extern BOOLEAN ONEdeltaConverged(ONEdevice *);
extern void ONEresetJacobian(ONEdevice *);
extern void ONEstoreNeutralGuess(ONEdevice *);
extern void ONEequilSolve(ONEdevice *);
extern void ONEbiasSolve(ONEdevice *, int, BOOLEAN, ONEtranInfo *);
extern void ONEstoreEquilibGuess(ONEdevice *);
extern void ONEstoreInitialGuess(ONEdevice *);
extern int ONEnewDelta(ONEdevice *, BOOLEAN, ONEtranInfo *);
extern double ONEtrunc(ONEdevice *, ONEtranInfo *, double);
extern void ONEsaveState(ONEdevice *);
extern double ONEnuNorm(ONEdevice *);
extern void ONEjacCheck(ONEdevice *, BOOLEAN, ONEtranInfo *);
extern void ONEpredict(ONEdevice *, ONEtranInfo *);
extern BOOLEAN ONEdeviceConverged(ONEdevice *);


#endif
