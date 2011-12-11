/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/
/*
Imported into HFETA model: Paolo Nenzi 2001
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "hfetdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
HFETAmAsk(CKTcircuit *ckt, GENmodel *inst, int which, IFvalue *value)
{
    HFETAmodel *here = (HFETAmodel*)inst;

    NG_IGNORE(ckt);

    switch(which) {
        case HFETA_MOD_VTO:
            value->rValue = here->HFETAthreshold;
            return (OK);
        case HFETA_MOD_LAMBDA:
            value->rValue = here->HFETAlambda;
            return (OK);
        case HFETA_MOD_RD:
            value->rValue = here->HFETArd;
            return (OK);
        case HFETA_MOD_RS:
            value->rValue = here->HFETArs;
            return (OK);
        case HFETA_MOD_RG:
            value->rValue = here->HFETArg;
            return (OK);  
        case HFETA_MOD_RDI:
            value->rValue = here->HFETArdi;
            return (OK);
        case HFETA_MOD_RSI:
            value->rValue = here->HFETArsi;
            return (OK);
        case HFETA_MOD_RGS:
            value->rValue = here->HFETArgs;
            return (OK);        
        case HFETA_MOD_RGD:
            value->rValue = here->HFETArgd;
            return (OK);
        case HFETA_MOD_RI:
            value->rValue = here->HFETAri;
            return (OK);
        case HFETA_MOD_RF:
            value->rValue = here->HFETArf;
            return (OK);
        case HFETA_MOD_ETA:
            value->rValue = here->HFETAeta;
            return (OK);
        case HFETA_MOD_M:
            value->rValue = here->HFETAm;
            return (OK);
        case HFETA_MOD_MC:
            value->rValue = here->HFETAmc;
            return (OK);   
        case HFETA_MOD_GAMMA:
            value->rValue = here->HFETAgamma;
            return (OK);
        case HFETA_MOD_SIGMA0:
            value->rValue = here->HFETAsigma0;
            return (OK);
        case HFETA_MOD_VSIGMAT:
            value->rValue = here->HFETAvsigmat;
            return (OK);
        case HFETA_MOD_VSIGMA:
            value->rValue = here->HFETAvsigma;
            return (OK);
        case HFETA_MOD_MU:
            value->rValue = here->HFETAmu;
            return (OK);
        case HFETA_MOD_DI:
            value->rValue = here->HFETAdi;
            return (OK);       
        case HFETA_MOD_DELTA:
            value->rValue = here->HFETAdelta;
            return (OK);
        case HFETA_MOD_VS:
            value->rValue = here->HFETAvs;
            return (OK);
        case HFETA_MOD_NMAX:
            value->rValue = here->HFETAnmax;
            return (OK);
        case HFETA_MOD_DELTAD:
            value->rValue = here->HFETAdeltad;
            return (OK);
        case HFETA_MOD_JS1D:
            value->rValue = here->HFETAjs1d;
            return (OK);
        case HFETA_MOD_JS2D:
            value->rValue = here->HFETAjs2d;
            return (OK);       
        case HFETA_MOD_JS1S:
            value->rValue = here->HFETAjs1s;
            return (OK);
        case HFETA_MOD_JS2S:
            value->rValue = here->HFETAjs2s;
            return (OK);
        case HFETA_MOD_M1D:
            value->rValue = here->HFETAm1d;
            return (OK);
        case HFETA_MOD_M2D:
            value->rValue = here->HFETAm2d;
            return (OK);
        case HFETA_MOD_M1S:
            value->rValue = here->HFETAm1s;
            return (OK);
        case HFETA_MOD_M2S:
            value->rValue = here->HFETAm2s;
            return (OK);   
        case HFETA_MOD_EPSI:
            value->rValue = here->HFETAepsi;
            return (OK);
        case HFETA_MOD_P:
            value->rValue = here->HFETAp;
            return (OK);
        case HFETA_MOD_CM3:
            value->rValue = here->HFETAcm3;
            return (OK);
        case HFETA_MOD_A1:
            value->rValue = here->HFETAa1;
            return (OK);
        case HFETA_MOD_A2:
            value->rValue = here->HFETAa2;
            return (OK);
        case HFETA_MOD_MV1:
            value->rValue = here->HFETAmv1;
            return (OK);   
        case HFETA_MOD_KAPPA:
            value->rValue = here->HFETAkappa;
            return (OK);
        case HFETA_MOD_DELF:
            value->rValue = here->HFETAdelf;
            return (OK);
        case HFETA_MOD_FGDS:
            value->rValue = here->HFETAfgds;
            return (OK);
         case HFETA_MOD_TF:
            value->rValue = here->HFETAtf;
            return (OK);
        case HFETA_MOD_CDS:
            value->rValue = here->HFETAcds;
            return (OK);
        case HFETA_MOD_PHIB:
            value->rValue = here->HFETAphib;
            return (OK);   
        
        case HFETA_MOD_TALPHA:
            value->rValue = here->HFETAtalpha;
            return (OK);
        case HFETA_MOD_MT1:
            value->rValue = here->HFETAmt1;
            return (OK);
        case HFETA_MOD_MT2:
            value->rValue = here->HFETAmt2;
            return (OK);
        case HFETA_MOD_CK1:
            value->rValue = here->HFETAck1;
            return (OK);
        case HFETA_MOD_CK2:
            value->rValue = here->HFETAck2;
            return (OK);
        case HFETA_MOD_CM1:
            value->rValue = here->HFETAcm1;
            return (OK);           
        case HFETA_MOD_CM2:
            value->rValue = here->HFETAcm2;
            return (OK);
        case HFETA_MOD_ASTAR:
            value->rValue = here->HFETAastar;
            return (OK);
        case HFETA_MOD_ETA1:
            value->rValue = here->HFETAeta1;
            return (OK);
        case HFETA_MOD_D1:
            value->rValue = here->HFETAd1;
            return (OK);
        case HFETA_MOD_VT1:
            value->rValue = here->HFETAvt1;
            return (OK);
        case HFETA_MOD_ETA2:
            value->rValue = here->HFETAeta2;
            return (OK);           
        case HFETA_MOD_D2:
            value->rValue = here->HFETAd2;
            return (OK);
        case HFETA_MOD_VT2:
            value->rValue = here->HFETAvt2;
            return (OK);
        case HFETA_MOD_GGR:
            value->rValue = here->HFETAggr;
            return (OK);
        case HFETA_MOD_DEL:
            value->rValue = here->HFETAdel;
            return (OK);
        case HFETA_MOD_GATEMOD:
            value->iValue = here->HFETAgatemod;
            return (OK);
        case HFETA_MOD_KLAMBDA:
            value->rValue = here->HFETAklambda;
            return (OK);       
        case HFETA_MOD_KMU:
            value->rValue = here->HFETAkmu;
            return (OK);
        case HFETA_MOD_KVTO:
            value->rValue = here->HFETAkvto;
            return (OK);
            
        case HFETA_MOD_DRAINCONDUCT:
        	  value->rValue = here->HFETAdrainConduct;
        	  return (OK);	
        case HFETA_MOD_SOURCECONDUCT:
        	  value->rValue = here->HFETAsourceConduct;
        	  return (OK);
      /*  case HFETA_MOD_DEPLETIONCAP:
        	  value->rValue = here->HFETA???;  
        	  return(OK); */       
	/*   case HFETA_MOD_VCRIT:
	   	  value->rValue = here->HFETAvcrit; 
	       return (OK); */
	   
	   case HFETA_MOD_TYPE:
	    if (here->HFETAtype == NHFET)
                value->sValue = "nhfet";
	    else
                value->sValue = "phfet";
	    return (OK);
        default:
            return (E_BADPARM);
    }
    /* NOTREACHED */
}
