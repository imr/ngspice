/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#ifndef DEV
#define DEV

#include "optdefs.h"
#include "gendefs.h"
#include "ifsim.h"
#include "cktdefs.h"
#include "noisedef.h"
#include "complex.h"

#ifdef __STDC__
double DEVlimvds(double,double);
double DEVpnjlim(double,double,double,double,int*);
double DEVfetlim(double,double,double);
void DEVcmeyer(double,double,double,double,double,double,double,double,double,
        double,double,double*,double*,double*,double,double,double,double);
void DEVqmeyer(double,double,double,double,double,double*,double*,double*,
        double,double);
#ifdef notdef
void DEVcap(CKTcircuit*, double, double, double, double, double, double,
        double, double, double, double, double, double, double, double,
        double*, double*, double*, double*, double*, double*, double*,
        double*, double*, double*, double*, double*, double, double,
        double, double*, double*, double);
#endif
double DEVpred(CKTcircuit*,int);
#else /* stdc */
double DEVlimvds();
double DEVpnjlim();
double DEVfetlim();
void DEVcmeyer();
void DEVqmeyer();
#ifdef notdef
void DEVcap();
#endif
double DEVpred();
#endif /* stdc */

typedef struct SPICEdev {
    IFdevice DEVpublic;

#ifdef __STDC__
    int (*DEVparam)(int,IFvalue*,GENinstance*,IFvalue *);  
        /* routine to input a parameter to a device instance */
    int (*DEVmodParam)(int,IFvalue*,GENmodel*);   
        /* routine to input a paramater to a model */
    int (*DEVload)(GENmodel*,CKTcircuit*);   
        /* routine to load the device into the matrix */
    int (*DEVsetup)(SMPmatrix*,GENmodel*,CKTcircuit*,int*);  
        /* setup routine to preprocess devices once before soloution begins */
    int (*DEVunsetup)(GENmodel*,CKTcircuit*);
	/* clean up before running again */
    int (*DEVpzSetup)(SMPmatrix*,GENmodel*,CKTcircuit*,int*);  
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
    void (*DEVdestroy)(GENmodel**);   
        /* subroutine to destroy all models and instances */
    int (*DEVmodDelete)(GENmodel**,IFuid,GENmodel*);  
        /* subroutine to delete a model and all instances */
    int (*DEVdelete)(GENmodel*,IFuid,GENinstance**); 
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

#else /* stdc */

    int (*DEVparam)();  /* routine to input a parameter to a device instance */
    int (*DEVmodParam)();   /* routine to input a paramater to a model */
    int (*DEVload)();   /* routine to load the device into the matrix */
    int (*DEVsetup)();  /* setup routine to preprocess devices once before
                         * soloution begins */
    int (*DEVunsetup)(); /* clean up before running again */

    int (*DEVpzSetup)();  /* setup routine to process devices specially for 
                         * pz analysis */
    int (*DEVtemperature)();    /* subroutine to do temperature dependent 
                                 * setup processing */
    int (*DEVtrunc)();  /* subroutine to perform truncation error calc. */
    int (*DEVfindBranch)(); /* subroutine to search for device branch eq.s */
    int (*DEVacLoad)(); /* ac analysis loading function */
    int (*DEVaccept)(); /* subroutine to call on acceptance of a timepoint */
    void (*DEVdestroy)();   /* subroutine to destroy all models and instances */
    int (*DEVmodDelete)();  /* subroutine to delete a model and all instances */
    int (*DEVdelete)(); /* subroutine to delete an instance */
    int (*DEVsetic)();  /* routine to pick up device init conds from rhs */
    int (*DEVask)();    /* routine to ask about device details*/
    int (*DEVmodAsk)(); /* routine to ask about model details*/
    int (*DEVpzLoad)(); /* routine to load for pole-zero analysis */
    int (*DEVconvTest)();   /* convergence test function */

    int (*DEVsenSetup)();   /* routine to setup the device sensitivity info */
    int (*DEVsenLoad)();    /* routine to load the device sensitivity info */
    int (*DEVsenUpdate)();  /* routine to update the device sensitivity info */
    int (*DEVsenAcLoad)();  /* routine to load  the device ac sensitivity info*/
    void (*DEVsenPrint)();   /* subroutine to print out sensitivity info */
    int (*DEVsenTrunc)();   /* subroutine to print out sensitivity info */
    int (*DEVdisto)();  /* distortion routine */
    int (*DEVnoise)(); /* noise routine */

#endif /* stdc */

    int *DEVinstSize;    /* size of an instance */
    int *DEVmodSize;     /* size of a model */

} SPICEdev;  /* instance of structure for each possible type of device */

/* IOP( )	Input/output parameter
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
# define IOPP(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_PRINCIPAL,		d }
# define IOPA(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_AC,			d }
# define IOPAU(a,b,c,d) { a, b, c|IF_SET|IF_ASK|IF_AC|IF_UNINTERESTING,d }
# define IOPAP(a,b,c,d) { a, b, c|IF_SET|IF_ASK|IF_AC|IF_PRINCIPAL,	d }
# define IOPAA(a,b,c,d) { a, b, c|IF_SET|IF_ASK|IF_AC_ONLY,		d }
# define IOPAAU(a,b,c,d) { a, b, c|IF_SET|IF_ASK|IF_AC_ONLY|IF_UNINTERESTING,d}
# define IOPPA(a,b,c,d) { a, b, c|IF_SET|IF_ASK|IF_AC_ONLY|IF_PRINCIPAL, d }
# define IOPN(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_NOISE,		d }
# define IOPR(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_REDUNDANT,		NULL }
# define IOPX(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_NONSENSE,		d }
# define IOPXU(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_NONSENSE|IF_UNINTERESTING,\
									d }
# define IOPQ(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_SETQUERY,		d }
# define IOPQU(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_SETQUERY|IF_UNINTERESTING,\
									d }
# define IOPZ(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_CHKQUERY,		d }
# define IOPZU(a,b,c,d)  { a, b, c|IF_SET|IF_ASK|IF_CHKQUERY|IF_UNINTERESTING,\
									d }
# define IOPQO(a,b,c,d) { a, b, c|IF_SET|IF_ASK|IF_ORQUERY,		d }

# define IP(a,b,c,d) { a , b , c|IF_SET , d }
# define OP(a,b,c,d) { a , b , c|IF_ASK , d }
# define OPU(a,b,c,d) { a , b , c|IF_ASK|IF_UNINTERESTING , d }
# define OPR(a,b,c,d) { a , b , c|IF_ASK|IF_REDUNDANT , d }
# define P(a,b,c,d) { a , b , c , d }



#define DEV_DEFAULT	0x1

#endif /*DEV*/
