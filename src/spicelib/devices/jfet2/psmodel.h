/*
 	Parker-Skellern MESFET model, UCB Spice Glue Header
 	
	Copyright (C) 1994, 1995, 1996  Macquarie University                    
	All Rights Reserved 
	Author: Anthony Parker
	Creation Date:	2 Feb 1994 
	Modified: 24 Mar 1994:   Parameters MVST and ETA added. 
          18 Apr 1994:   Added new parameters and comments
	  12 Sep 1995:   Changed _XXX to PS_XXX to aid portability
 */


#ifdef PSMODEL_C  /* PSMODEL_C defined when included from "psmodel.c" */
#include "ngspice/ngspice.h"
#include "jfet2defs.h"
#include "ngspice/const.h"
#endif

/* Glue definitions for cref modl and inst */
typedef  CKTcircuit    cref; /* circuit specific variables */
typedef  JFET2model     modl; /* model parameters for this type of device */
typedef  JFET2instance  inst; /* parameters specific to this device instance */

extern	void	PSinstanceinit(modl *,inst *);
extern	double	PSids(cref *,modl *,inst *,double,double,
		        double *,double *,double *,double *,double *,double *);
extern	void	PScharge(cref *,modl *,inst *,double,double,double *,double *);
extern  void    PSacload(cref *,modl *,inst *,double,double,double,double,
                                          double *,double *,double *,double *);

#ifdef PSMODEL_C /* PSMODEL_C defined when included from "psmodel.c" */
/* The following glue definitions need to be changed to suit the specific
   simulator.  */
/* simulator mode flags 
    TRAN_ANAL should be true during transient analysis iteration.
              (ie. false for other analysis functions and tran operating point.)
    TRAN_INIT should be true only during the first calculation of the initial 
              transient analysis time point. It should be false for remaining
              iterations at that time point and the rest of the transient analysis.
 */
#define TRAN_ANAL          (ckt->CKTmode & MODETRAN) 
#define TRAN_INIT          (ckt->CKTmode & MODEINITTRAN)

/* state variables */
/* initialized when TRAN_ANAL is false */
#define VGSTRAP_BEFORE     (*(ckt->CKTstate1 + here->JFET2vgstrap))
#define VGSTRAP_NOW        (*(ckt->CKTstate0 + here->JFET2vgstrap))
#define VGDTRAP_BEFORE     (*(ckt->CKTstate1 + here->JFET2vtrap))
#define VGDTRAP_NOW        (*(ckt->CKTstate0 + here->JFET2vtrap))
#define POWR_BEFORE        (*(ckt->CKTstate1 + here->JFET2pave))
#define POWR_NOW           (*(ckt->CKTstate0 + here->JFET2pave))

/* initialized when TRAN_INIT is true or TRAN_ANAL is false */
#define QGS_BEFORE         (*(ckt->CKTstate1 + here->JFET2qgs))
#define QGS_NOW            (*(ckt->CKTstate0 + here->JFET2qgs))
#define QGD_BEFORE         (*(ckt->CKTstate1 + here->JFET2qgd))
#define QGD_NOW            (*(ckt->CKTstate0 + here->JFET2qgd))

/* past terminal potentials used if TRAN_INIT is false and TRAN_ANAL is true */
#define VGS1               (*(ckt->CKTstate1 + here->JFET2vgs))
#define VGD1               (*(ckt->CKTstate1 + here->JFET2vgd))

/* simulator specific parameters */
#define GMIN    ckt->CKTgmin                       /* SPICE gmin (1E12 ohms) */
#define NVT     here->JFET2temp*CONSTKoverQ*model->JFET2n             /* nkT/q */
#define STEP    ckt->CKTdelta        /* time step of this transient solution */
#define FOURTH  0.25           /* eldo requires 2.5e-10 for units conversion */

/* model parameters */
/* dc model */
#define BETA    model->JFET2beta      /* transconductance scaling */
#define DELT    model->JFET2delta     /* thermal current reduction */
#define IBD     model->JFET2ibd       /* breakdown current */
#define IS      here->JFET2tSatCur    /* gate reverse saturation current */
#define LAM     model->JFET2lambda    /* channel length modulation */
#define LFGAM   model->JFET2lfgam     /* dc drain feedback */
#define LFG1    model->JFET2lfg1      /* dc drain feedback vgs modulation */
#define LFG2    model->JFET2lfg2      /* dc drain feedback vgd modulation */
#define MVST    model->JFET2mvst      /* subthreshold vds modulation */
#define MXI     model->JFET2mxi       /* saturation index vgs modulation */
#define P       model->JFET2p         /* power law in controlled resistance */
#define Q       model->JFET2q         /* power law in controlled current */
#define VBD     model->JFET2vbd       /* breakdown exponential coef */
#define VBI     here->JFET2tGatePot   /* junction built-in potential */
#define VSUB    model->JFET2vst       /* subthreshold exponential coef */
#define VTO     model->JFET2vto       /* pinch-off potential */
#define XI      model->JFET2xi        /* saturation index */
#define Z       model->JFET2z         /* saturation knee curvature */

/* ac model */
#define ACGAM   model->JFET2acgam     /* capacitance vds modulation */
#define CGS     here->JFET2tCGS       /* zero bias cgs */
#define CGD     here->JFET2tCGD       /* zero bias cgd */
#define HFETA   model->JFET2hfeta     /* ac source feedback */
#define HFE1    model->JFET2hfe1      /* ac source feedback vgd modulation */
#define HFE2    model->JFET2hfe2      /* ac source feedback vgs modulation */
#define HFGAM   model->JFET2hfgam     /* ac drain feedback */
#define HFG1    model->JFET2hfg1      /* ac drain feedback vgs modulation */
#define HFG2    model->JFET2hfg2      /* ac drain feedback vgd modulation */
#define TAUD    model->JFET2taud      /* thermal time constant */
#define TAUG    model->JFET2taug      /* dc ac feedback time constant */
#define XC      model->JFET2xc        /* cgs reduction at pinch-off */

/* device instance */
#define AREA       here->JFET2area       /* area factor of fet */

/* internally derived model parameters */
#define ALPHA   here->JFET2alpha      /* cgs cgd reversal interval */
#define D3      here->JFET2d3         /* dual power-law parameter */
#define VMAX    here->JFET2corDepCap  /* forward capacitance potential */
#define XI_WOO  here->JFET2xiwoo      /* saturation potential */
#define ZA      model->JFET2za        /* saturation knee parameter */
#endif
