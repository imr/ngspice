/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/
/*
Imported into HFET2 model: Paolo Nenzi 2001
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "hfet2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
HFET2mAsk(CKTcircuit *ckt, GENmodel *inst, int which, IFvalue *value)
{
    HFET2model *here = (HFET2model*)inst;

    NG_IGNORE(ckt);

    switch(which) {
        case HFET2_MOD_VTO:
            value->rValue = here->HFET2vto;
            return (OK);
        case HFET2_MOD_LAMBDA:
            value->rValue = here->HFET2lambda;
            return (OK);
        case HFET2_MOD_RD:
            value->rValue = here->HFET2rd;
            return (OK);
        case HFET2_MOD_RS:
            value->rValue = here->HFET2rs;
            return (OK);
        case HFET2_MOD_RDI:
            value->rValue = here->HFET2rdi;
            return (OK);
        case HFET2_MOD_RSI:
            value->rValue = here->HFET2rsi;
            return (OK);
        case HFET2_MOD_ETA:
            value->rValue = here->HFET2eta;
            return (OK);
        case HFET2_MOD_M:
            value->rValue = here->HFET2m;
            return (OK);
        case HFET2_MOD_MC:
            value->rValue = here->HFET2mc;
            return (OK);   
        case HFET2_MOD_GAMMA:
            value->rValue = here->HFET2gamma;
            return (OK);
        case HFET2_MOD_SIGMA0:
            value->rValue = here->HFET2sigma0;
            return (OK);
        case HFET2_MOD_VSIGMAT:
            value->rValue = here->HFET2vsigmat;
            return (OK);
        case HFET2_MOD_VSIGMA:
            value->rValue = here->HFET2vsigma;
            return (OK);
        case HFET2_MOD_MU:
            value->rValue = here->HFET2mu;
            return (OK);
        case HFET2_MOD_DI:
            value->rValue = here->HFET2di;
            return (OK);       
        case HFET2_MOD_DELTA:
            value->rValue = here->HFET2delta;
            return (OK);
        case HFET2_MOD_VS:
            value->rValue = here->HFET2vs;
            return (OK);
        case HFET2_MOD_NMAX:
            value->rValue = here->HFET2nmax;
            return (OK);
        case HFET2_MOD_DELTAD:
            value->rValue = here->HFET2deltad;
            return (OK);
        case HFET2_MOD_P:
            value->rValue = here->HFET2p;
            return (OK);
        case HFET2_MOD_JS:
            value->rValue = here->HFET2js;
            return (OK);
        case HFET2_MOD_ETA1:
            value->rValue = here->HFET2eta1;
            return (OK);
        case HFET2_MOD_D1:
            value->rValue = here->HFET2d1;
            return (OK);
        case HFET2_MOD_VT1:
            value->rValue = here->HFET2vt1;
            return (OK);
        case HFET2_MOD_ETA2:
            value->rValue = here->HFET2eta2;
            return (OK);           
        case HFET2_MOD_D2:
            value->rValue = here->HFET2d2;
            return (OK);
        case HFET2_MOD_VT2:
            value->rValue = here->HFET2vt2;
            return (OK);
        case HFET2_MOD_GGR:
            value->rValue = here->HFET2ggr;
            return (OK);
        case HFET2_MOD_DEL:
            value->rValue = here->HFET2del;
            return (OK);
        case HFET2_MOD_KLAMBDA:
            value->rValue = here->HFET2klambda;
            return (OK);       
        case HFET2_MOD_KMU:
            value->rValue = here->HFET2kmu;
            return (OK);
        case HFET2_MOD_KVTO:
            value->rValue = here->HFET2kvto;
            return (OK);
        case HFET2_MOD_EPSI:
            value->rValue = here->HFET2epsi;
            return (OK);       
        case HFET2_MOD_KNMAX:
            value->rValue = here->HFET2knmax;
            return (OK);
        case HFET2_MOD_N:
            value->rValue = here->HFET2n;
            return (OK);
        case HFET2_MOD_CF:
            value->rValue = here->HFET2cf;
            return (OK); 
            
        case HFET2_MOD_DRAINCONDUCT:
        	  value->rValue = here->HFET2drainConduct;
        	  return (OK);	
        case HFET2_MOD_SOURCECONDUCT:
        	  value->rValue = here->HFET2sourceConduct;
        	  return (OK);
      /*  case HFET2_MOD_DEPLETIONCAP:
        	  value->rValue = here->HFET2???;  
        	  return(OK); */       
	 /*  case HFET2_MOD_VCRIT:
	   	  value->rValue = here->HFET2vcrit; 
	       return (OK); */ 
	   
	   case HFET2_MOD_TYPE:
	    if (here->HFET2type == NHFET)
                value->sValue = "nhfet";
	    else
                value->sValue = "phfet";
	    return (OK);
        default:
            return (E_BADPARM);
    }
    /* NOTREACHED */
}
