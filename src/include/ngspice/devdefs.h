/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef ngspice_DEVDEFS_H
#define ngspice_DEVDEFS_H

#include "ngspice/optdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/noisedef.h"
#include "ngspice/complex.h"

double DEVlimvds(double,double);
double DEVpnjlim(double,double,double,double,int*);
double DEVfetlim(double,double,double);
double DEVlimitlog(double, double, double, int*);
void DEVcmeyer(double,double,double,double,double,double,double,double,double,
        double,double,double*,double*,double*,double,double,double,double);
void DEVqmeyer(double,double,double,double,double,double*,double*,double*,
        double,double);
void DevCapVDMOS(double, double, double, double, double,
                 double*, double*);
double DEVpred(CKTcircuit*,int);

/* Cider integration */
double limitResistorVoltage( double, double, int * );
double limitJunctionVoltage( double, double, int * );
double limitVbe( double, double, int * );
double limitVce( double, double, int * );
double limitVgb( double, double, int * );

/* Area Calculation Method (ACM) for MOS models (devsup.c) */
int
ACM_SourceDrainResistances(int, double, double, double, double, double,
                           double, double, int, double, double, double,
                           int, double, double, double, double *, double *);
int
ACM_saturationCurrents(int, int, int, double, double, double, double, double,
                       double, int, double, int, double, int, double, int,
                       double, double *, double *);
int
ACM_junctionCapacitances(int, int, int, double, double, double, double, int,
                         double, int, double, int, double, int, double,
                         double, double, double, double *, double *,
                         double *, double *, double *, double *);

typedef struct SPICEdev {
    IFdevice DEVpublic;

    int (*DEVparam)(int,IFvalue*,GENinstance*,IFvalue *);  
        /* routine to input a parameter to a device instance */
    int (*DEVmodParam)(int,IFvalue*,GENmodel*);   
        /* routine to input a paramater to a model */
    int (*DEVload)(GENmodel*,CKTcircuit*);   
        /* routine to load the device into the matrix */
    int (*DEVsetup)(SMPmatrix *, GENmodel *, CKTcircuit *, int *);
        /* setup routine to preprocess devices once before soloution begins */
    int (*DEVunsetup)(GENmodel*,CKTcircuit*);
	/* clean up before running again */
    int (*DEVpzSetup)(SMPmatrix *, GENmodel *, CKTcircuit *, int *);
        /* setup routine to process devices specially for pz analysis */
    int (*DEVtemperature)(GENmodel*,CKTcircuit*);
        /* subroutine to do temperature dependent setup processing */
    int (*DEVtrunc)(GENmodel*,CKTcircuit*,double*);
        /* subroutine to perform truncation error calc. */
    int (*DEVfindBranch)(CKTcircuit*,GENmodel*,IFuid); 
        /* subroutine to search for device branch eq.s */
    int (*DEVacLoad)(GENmodel*,CKTcircuit*); 
        /* ac analysis loading function */
    int (*DEVaccept)(CKTcircuit*,GENmodel*); 
        /* subroutine to call on acceptance of a timepoint */
    void (*DEVdestroy)(void);
        /* subroutine to delete device specfic extra data */
    int (*DEVmodDelete)(GENmodel*);
        /* subroutine to delete a model */
    int (*DEVdelete)(GENinstance*);
        /* subroutine to delete an instance */
    int (*DEVsetic)(GENmodel*,CKTcircuit*);  
        /* routine to pick up device init conds from rhs */
    int (*DEVask)(CKTcircuit*,GENinstance*,int,IFvalue*,IFvalue*);    
        /* routine to ask about device details*/
    int (*DEVmodAsk)(CKTcircuit*,GENmodel*,int,IFvalue*); 
        /* routine to ask about model details*/
    int (*DEVpzLoad)(GENmodel*,CKTcircuit*,SPcomplex*); 
        /* routine to load for pole-zero analysis */
    int (*DEVconvTest)(GENmodel*,CKTcircuit*);  
        /* convergence test function */

    int (*DEVsenSetup)(SENstruct*,GENmodel*);
        /* routine to setup the device sensitivity info */
    int (*DEVsenLoad)(GENmodel*,CKTcircuit*);   
        /* routine to load the device sensitivity info */
    int (*DEVsenUpdate)(GENmodel*,CKTcircuit*); 
        /* routine to update the device sensitivity info */
    int (*DEVsenAcLoad)(GENmodel*,CKTcircuit*); 
        /* routine to load  the device ac sensitivity info */
    void (*DEVsenPrint)(GENmodel*,CKTcircuit*);   
        /* subroutine to print out sensitivity info */
    int (*DEVsenTrunc)(GENmodel*,CKTcircuit*,double*);   
        /* subroutine to print out sensitivity info */
    int (*DEVdisto)(int,GENmodel*,CKTcircuit*);  
    	/* procedure to do distortion operations */
    int (*DEVnoise)(int, int, GENmodel*,CKTcircuit*, Ndata *, double *);
	/* noise routine */
    int (*DEVsoaCheck)(CKTcircuit*,GENmodel*);
	/* subroutine to call on soa check */
#ifdef CIDER 	
    void (*DEVdump)(GENmodel *, CKTcircuit *);
    void (*DEVacct)(GENmodel *, CKTcircuit *, FILE *);
        /* routines to report device internals
         * now used only by cider numerical devices
         */
#endif
    int *DEVinstSize;    /* size of an instance */
    int *DEVmodSize;     /* size of a model */

} SPICEdev;  /* instance of structure for each possible type of device */


extern SPICEdev **DEVices;
extern int        DEVmaxnum;    /* size of DEVices array */


/* IOP( )		Input/output parameter
 * IOPP( )	IO parameter which the principle value of a device (used
 *			for naming output variables in sensetivity)
 * IOPA( )	IO parameter significant for time-varying (non-dc) analyses
 * IOPAP( )	Principle value is significant for time-varying analyses
 * IOPAA( )	IO parameter significant for ac analyses only
 * IOPAAP( )	IO parameter principle value for ac analyses only
 * IOPN( )	IO parameter significant for noise analyses only
 * IOPR( )	Redundant IO parameter name (e.g. "vto" vs. "vt0")
 * IOPX( )	IO parameter which is not used by sensetivity in any case
 *
 * IOPQ( )	This (Q) parameter must be non-zero for sensetivity of
 *			following Z parameter (model params done first)
 * IOPZ( )	Prev. 'Q' parameter must be non-zero for sensetivity
 * IOPQO( )	Like Q, but or-ed with previous Q value
 * ....U( )	uninteresting for default "show" command output
 */


# define IOP(a,b,c,d)   { a, b, c|IF_SET|IF_ASK,			d }
# define IOPU(a,b,c,d)   { a, b, c|IF_SET|IF_ASK|IF_UNINTERESTING,	d }
# define IOPUR(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_UNINTERESTING|IF_REDUNDANT, d }
# define IOPP(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_PRINCIPAL,		d }
# define IOPPR(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_PRINCIPAL|IF_REDUNDANT, d }
# define IOPA(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_AC,			d }
# define IOPAR(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_AC|IF_REDUNDANT,    d }
# define IOPAU(a,b,c,d) { a, b, c|IF_SET|IF_ASK|IF_AC|IF_UNINTERESTING,d }
# define IOPAP(a,b,c,d) { a, b, c|IF_SET|IF_ASK|IF_AC|IF_PRINCIPAL,	d }
# define IOPAPR(a,b,c,d) { a, b, c|IF_SET|IF_ASK|IF_AC|IF_PRINCIPAL|IF_REDUNDANT, d }
# define IOPAA(a,b,c,d) { a, b, c|IF_SET|IF_ASK|IF_AC_ONLY,		d }
# define IOPAAU(a,b,c,d) { a, b, c|IF_SET|IF_ASK|IF_AC_ONLY|IF_UNINTERESTING,d}
# define IOPPA(a,b,c,d) { a, b, c|IF_SET|IF_ASK|IF_AC_ONLY|IF_PRINCIPAL, d }
# define IOPN(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_NOISE,		d }
# define IOPR(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_REDUNDANT,		NULL }
# define IOPX(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_NONSENSE,		d }
# define IOPXR(a,b,c,d) { a, b, c|IF_SET|IF_ASK|IF_NONSENSE|IF_REDUNDANT, d }
# define IOPXU(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_NONSENSE|IF_UNINTERESTING,\
									d }
# define IOPQ(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_SETQUERY,		d }
# define IOPQR(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_SETQUERY|IF_REDUNDANT, d }
# define IOPQU(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_SETQUERY|IF_UNINTERESTING,\
									d }
# define IOPZ(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_CHKQUERY,		d }
# define IOPZR(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_CHKQUERY|IF_REDUNDANT, d }
# define IOPZU(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_CHKQUERY|IF_UNINTERESTING,\
									d }
# define IOPQO(a,b,c,d) { a, b, c|IF_SET|IF_ASK|IF_ORQUERY,		d }
# define IOPQOR(a,b,c,d) { a, b, c|IF_SET|IF_ASK|IF_ORQUERY|IF_REDUNDANT, d }

# define IP(a,b,c,d) { a , b , c|IF_SET , d }
# define IPR(a,b,c,d) { a , b , c|IF_SET|IF_REDUNDANT , d }
# define OP(a,b,c,d) { a , b , c|IF_ASK , d }
# define OPU(a,b,c,d) { a , b , c|IF_ASK|IF_UNINTERESTING , d }
# define OPUR(a,b,c,d) { a , b , c|IF_ASK|IF_UNINTERESTING|IF_REDUNDANT , d }
# define OPR(a,b,c,d) { a , b , c|IF_ASK|IF_REDUNDANT , d }
# define P(a,b,c,d) { a , b , c , d }



#define DEV_DEFAULT	0x1

#endif
