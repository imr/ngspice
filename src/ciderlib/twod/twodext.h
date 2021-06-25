/*
 * 2001 Paolo Nenzi
 */
 
 /* External symbols for Two Dimensional simulator */
 
#ifndef ngspice_TWODEXT_H
#define ngspice_TWODEXT_H

#include "ngspice/profile.h"
#include "ngspice/twomesh.h"
#include "ngspice/twodev.h"
#include "ngspice/carddefs.h"
#include "ngspice/bool.h"
#include "ngspice/complex.h"
 
/* twoadmit.c */
extern int NUMD2admittance(TWOdevice *, double, SPcomplex *);
extern int NBJT2admittance(TWOdevice *, double, SPcomplex *,
                          SPcomplex *, SPcomplex *, SPcomplex *);
extern int NUMOSadmittance(TWOdevice *, double, struct mosAdmittances *);
extern BOOLEAN TWOsorSolve(TWOdevice *, double *, double  *, double);
extern SPcomplex *contactAdmittance(TWOdevice *, TWOcontact *, BOOLEAN,
                                    double *, double *, SPcomplex *);
extern SPcomplex *oxideAdmittance(TWOdevice *, TWOcontact *,BOOLEAN, 
                                  double *, double *, SPcomplex *);				    
				    
extern void NUMD2ys(TWOdevice *, SPcomplex *, SPcomplex *);
extern void NBJT2ys(TWOdevice *,SPcomplex *, SPcomplex *, SPcomplex *, 
                    SPcomplex *, SPcomplex *);
extern void NUMOSys(TWOdevice *, SPcomplex *, struct mosAdmittances *);		   
				    
/* twoaval.c */
extern double TWOavalanche(TWOelem *, TWOnode *);

/* twocond.c */
extern void NUMD2conductance(TWOdevice *, BOOLEAN, double *, double *);
extern void NBJT2conductance(TWOdevice *, BOOLEAN, double *, double *, 
                             double *, double *, double *);
extern void NUMOSconductance(TWOdevice *, BOOLEAN, double *, 
                             struct mosConductances *);
extern double contactCurrent(TWOdevice *, TWOcontact *);
extern double oxideCurrent(TWOdevice *, TWOcontact *, BOOLEAN);	
extern double contactConductance(TWOdevice *, TWOcontact *, BOOLEAN,
			                     double *, BOOLEAN, double *);
extern double oxideConductance(TWOdevice *, TWOcontact *, BOOLEAN, 
			double *, BOOLEAN, double *);		     			     
extern void NUMD2current(TWOdevice *, BOOLEAN, double *, double *);
extern void NBJT2current(TWOdevice *, BOOLEAN, double *, double *, 
                        double *);
extern void NUMOScurrent(TWOdevice *, BOOLEAN , double *, double *, double *, 
                         double *);		

/* twocont */
extern void TWO_jacBuild(TWOdevice *);
extern void TWO_sysLoad(TWOdevice *, BOOLEAN, TWOtranInfo *);
extern void TWO_jacLoad(TWOdevice *);
extern void TWO_rhsLoad(TWOdevice *, BOOLEAN, TWOtranInfo *);
extern void TWO_commonTerms(TWOdevice *, BOOLEAN, BOOLEAN, TWOtranInfo *);

/* twocurr.c */
extern void nodeCurrents(TWOelem *, TWOnode *, double *, double *,
    double *,  double *, double *, double *, double *, double *);
    
/* twodest */
extern void TWOdestroy(TWOdevice *);
 
/* twodopng.c */
extern double TWOdopingValue(DOPprofile *, DOPtable *, double, double);
extern void TWOsetDoping(TWOdevice *, DOPprofile *, DOPtable *);

/* twoelect.c */
extern int TWOcmpElectrode(TWOelectrode *, TWOelectrode *);
extern void checkElectrodes(TWOelectrode *, int);
extern void setupContacts(TWOdevice *, TWOelectrode *, TWOnode ***);

/* twofield.c */
extern void nodeFields(TWOelem *, TWOnode *, double *, double *);

/* twomesh.c */
extern void TWObuildMesh(TWOdevice *, TWOdomain *, TWOelectrode *,
                         TWOmaterial *);
extern void TWOprnMesh(TWOdevice *);			 
extern void TWOgetStatePointers(TWOdevice *, int *);

/* twomobdv.c */
extern void TWO_mobDeriv(TWOelem *, int, double);
extern void TWONmobDeriv(TWOelem *, int, double);
extern void TWOPmobDeriv(TWOelem *, int, double);

/* twomobfn.c */
extern void MOBsurfElec(TWOmaterial *, TWOelem *, double, double,
                        double, double, double, double);
extern void MOBsurfHole(TWOmaterial *, TWOelem *, double, double,
                        double, double, double, double);

/* twomobil.c */
extern void TWO_mobility(TWOelem *, double);
extern void TWONmobility(TWOelem *, double);
extern void TWOPmobility(TWOelem *, double);

/* twoncont.c */
extern void TWONjacBuild(TWOdevice *);
extern void TWONsysLoad(TWOdevice *, BOOLEAN, TWOtranInfo *);
extern void TWONjacLoad(TWOdevice *);
extern void TWONrhsLoad(TWOdevice *, BOOLEAN, TWOtranInfo *);
extern void TWONcommonTerms(TWOdevice *, BOOLEAN, BOOLEAN, TWOtranInfo *);

/* twopcont.c */
extern void TWOPjacBuild(TWOdevice *);
extern void TWOPsysLoad(TWOdevice *, BOOLEAN, TWOtranInfo *);
extern void TWOPjacLoad(TWOdevice *);
extern void TWOPrhsLoad(TWOdevice *, BOOLEAN, TWOtranInfo *);
extern void TWOPcommonTerms(TWOdevice *, BOOLEAN, BOOLEAN, TWOtranInfo *);

/* twopoiss.c */
extern void TWOQjacBuild(TWOdevice *);
extern void TWOQsysLoad(TWOdevice *);
extern void TWOQrhsLoad(TWOdevice *);
extern void TWOQcommonTerms(TWOdevice *);

/*twoprint.c */
extern void TWOprnSolution(FILE *, TWOdevice *, OUTPcard *, BOOLEAN, char *);
extern void TWOmemStats(FILE *, TWOdevice *);
extern void TWOcpuStats(FILE *, TWOdevice *);


/* twoproj.c */
extern void NUMD2project(TWOdevice *, double);
extern void NBJT2project(TWOdevice *, double, double);
extern void NUMOSproject(TWOdevice *, double, double, double);
extern void NUMD2update(TWOdevice *,double, BOOLEAN);
extern void NBJT2update(TWOdevice *, double, double, BOOLEAN);
extern void NUMOSupdate(TWOdevice *, double, double, double, BOOLEAN);
extern void storeNewRhs(TWOdevice *, TWOcontact *);

/* tworead.c */
extern int TWOreadState(TWOdevice *, char *, int, double *, double *, double *);

/*twosetbc.c */
extern void NUMD2setBCs(TWOdevice *, double);
extern void NBJT2setBCs(TWOdevice *, double, double);
extern void NUMOSsetBCs(TWOdevice *, double, double, double);

/*twosetup.c */
extern void TWOsetup(TWOdevice *);
extern void TWOsetBCparams(TWOdevice *, BDRYcard *);
extern void TWOnormalize(TWOdevice *);

/* twosolve.c */
extern void TWOdcSolve(TWOdevice *, int, BOOLEAN, BOOLEAN, TWOtranInfo *);
extern BOOLEAN TWOdeltaConverged(TWOdevice *);
extern BOOLEAN TWOdeviceConverged(TWOdevice *);
extern void TWOresetJacobian(TWOdevice *);
extern void TWOstoreNeutralGuess(TWOdevice *);
extern int TWOequilSolve(TWOdevice *);
extern void TWObiasSolve(TWOdevice *, int, BOOLEAN, TWOtranInfo *);
extern void TWOstoreEquilibGuess(TWOdevice *);
extern void TWOstoreInitialGuess(TWOdevice *);
extern void oldTWOnewDelta(TWOdevice *, BOOLEAN, TWOtranInfo *);
extern int TWOnewDelta(TWOdevice *, BOOLEAN, TWOtranInfo *);
extern void TWOpredict(TWOdevice *, TWOtranInfo *);
extern double TWOtrunc(TWOdevice *, TWOtranInfo *, double);
extern void TWOsaveState(TWOdevice *);
extern BOOLEAN TWOpsiDeltaConverged(TWOdevice *);
extern double TWOnuNorm(TWOdevice *);
extern void TWOjacCheck(TWOdevice *, BOOLEAN, TWOtranInfo *);

#endif
