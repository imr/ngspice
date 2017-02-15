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
extern bool TWOsorSolve(TWOdevice *, double *, double  *, double);
extern SPcomplex *contactAdmittance(TWOdevice *, TWOcontact *, bool,
                                    double *, double *, SPcomplex *);
extern SPcomplex *oxideAdmittance(TWOdevice *, TWOcontact *,bool, 
                                  double *, double *, SPcomplex *);				    
				    
extern void NUMD2ys(TWOdevice *, SPcomplex *, SPcomplex *);
extern void NBJT2ys(TWOdevice *,SPcomplex *, SPcomplex *, SPcomplex *, 
                    SPcomplex *, SPcomplex *);
extern void NUMOSys(TWOdevice *, SPcomplex *, struct mosAdmittances *);		   
				    
/* twoaval.c */
extern double TWOavalanche(TWOelem *, TWOnode *);

/* twocond.c */
extern void NUMD2conductance(TWOdevice *, bool, double *, double *);
extern void NBJT2conductance(TWOdevice *, bool, double *, double *, 
                             double *, double *, double *);
extern void NUMOSconductance(TWOdevice *, bool, double *, 
                             struct mosConductances *);
extern double contactCurrent(TWOdevice *, TWOcontact *);
extern double oxideCurrent(TWOdevice *, TWOcontact *, bool);	
extern double contactConductance(TWOdevice *, TWOcontact *, bool,
			                     double *, bool, double *);
extern double oxideConductance(TWOdevice *, TWOcontact *, bool, 
			double *, bool, double *);		     			     
extern void NUMD2current(TWOdevice *, bool, double *, double *);
extern void NBJT2current(TWOdevice *, bool, double *, double *, 
                        double *);
extern void NUMOScurrent(TWOdevice *, bool , double *, double *, double *, 
                         double *);		

/* twocont */
extern void TWO_jacBuild(TWOdevice *);
extern void TWO_sysLoad(TWOdevice *, bool, TWOtranInfo *);
extern void TWO_jacLoad(TWOdevice *);
extern void TWO_rhsLoad(TWOdevice *, bool, TWOtranInfo *);
extern void TWO_commonTerms(TWOdevice *, bool, bool, TWOtranInfo *);

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
extern void TWONsysLoad(TWOdevice *, bool, TWOtranInfo *);
extern void TWONjacLoad(TWOdevice *);
extern void TWONrhsLoad(TWOdevice *, bool, TWOtranInfo *);
extern void TWONcommonTerms(TWOdevice *, bool, bool, TWOtranInfo *);

/* twopcont.c */
extern void TWOPjacBuild(TWOdevice *);
extern void TWOPsysLoad(TWOdevice *, bool, TWOtranInfo *);
extern void TWOPjacLoad(TWOdevice *);
extern void TWOPrhsLoad(TWOdevice *, bool, TWOtranInfo *);
extern void TWOPcommonTerms(TWOdevice *, bool, bool, TWOtranInfo *);

/* twopoiss.c */
extern void TWOQjacBuild(TWOdevice *);
extern void TWOQsysLoad(TWOdevice *);
extern void TWOQrhsLoad(TWOdevice *);
extern void TWOQcommonTerms(TWOdevice *);

/*twoprint.c */
extern void TWOprnSolution(FILE *, TWOdevice *, OUTPcard *);
extern void TWOmemStats(FILE *, TWOdevice *);
extern void TWOcpuStats(FILE *, TWOdevice *);


/* twoproj.c */
extern void NUMD2project(TWOdevice *, double);
extern void NBJT2project(TWOdevice *, double, double);
extern void NUMOSproject(TWOdevice *, double, double, double);
extern void NUMD2update(TWOdevice *,double, bool);
extern void NBJT2update(TWOdevice *, double, double, bool);
extern void NUMOSupdate(TWOdevice *, double, double, double, bool);
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
extern void TWOdcSolve(TWOdevice *, int, bool, bool, TWOtranInfo *);
extern bool TWOdeltaConverged(TWOdevice *);
extern bool TWOdeviceConverged(TWOdevice *);
extern void TWOresetJacobian(TWOdevice *);
extern void TWOstoreNeutralGuess(TWOdevice *);
extern void TWOequilSolve(TWOdevice *);
extern void TWObiasSolve(TWOdevice *, int, bool, TWOtranInfo *);
extern void TWOstoreEquilibGuess(TWOdevice *);
extern void TWOstoreInitialGuess(TWOdevice *);
extern void oldTWOnewDelta(TWOdevice *, bool, TWOtranInfo *);
extern int TWOnewDelta(TWOdevice *, bool, TWOtranInfo *);
extern void TWOpredict(TWOdevice *, TWOtranInfo *);
extern double TWOtrunc(TWOdevice *, TWOtranInfo *, double);
extern void TWOsaveState(TWOdevice *);
extern bool TWOpsiDeltaConverged(TWOdevice *);
extern double TWOnuNorm(TWOdevice *);
extern void TWOjacCheck(TWOdevice *, bool, TWOtranInfo *);

#endif
