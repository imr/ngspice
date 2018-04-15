/**********
Imported from MacSpice3f4 - Antony Wilson
Modified: Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "hfetdefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/*
#define HFETAphibGiven
#define CHARGE 1.60219e-19
*/

int
HFETAsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
        /* load the diode structure with those pointers needed later 
         * for fast matrix loading 
         */
{
    HFETAmodel *model = (HFETAmodel*)inModel;
    HFETAinstance *here;
    int error;
    CKTnode *tmp;


    /*  loop through all the diode models */
    for( ; model != NULL; model = HFETAnextModel(model)) {
        if( (model->HFETAtype != NHFET) && (model->HFETAtype != PHFET) ) {
            model->HFETAtype = NHFET;
        }
        if(!model->HFETAthresholdGiven) {
            if(model->HFETAtype == NHFET)
              model->HFETAthreshold = 0.15;
            else
              model->HFETAthreshold = -0.15;
        }
        if(!model->HFETAdiGiven) {
            model->HFETAdi = 0.04e-6;
        }
        if(!model->HFETAlambdaGiven) {
            model->HFETAlambda = 0.15;
        }
        if(!model->HFETAetaGiven) {
            if(model->HFETAtype == NHFET)
              model->HFETAeta = 1.28;
            else
              model->HFETAeta = 1.4;
        }
        if(!model->HFETAmGiven) {
            model->HFETAm = 3.0;
        }
        if(!model->HFETAmcGiven) {
            model->HFETAmc = 3.0;
        }
        if(!model->HFETAgammaGiven) {
            model->HFETAgamma = 3.0;
        }
        if(!model->HFETAsigma0Given) {
            model->HFETAsigma0 = 0.057;
        }
        if(!model->HFETAvsigmatGiven) {
            model->HFETAvsigmat = 0.3;
        }
        if(!model->HFETAvsigmaGiven) {
            model->HFETAvsigma = 0.1;
        }
        if(!model->HFETAmuGiven) {
            if(model->HFETAtype == NHFET)
              model->HFETAmu = 0.4;
            else
              model->HFETAmu = 0.03;
        }
        if(!model->HFETAdeltaGiven) {
            model->HFETAdelta = 3.0;
        }
        if(!model->HFETAvsGiven) {
            if(model->HFETAtype == NHFET)
              model->HFETAvs = 1.5e5;
            else
              model->HFETAvs = 0.8e5;
        }
        if(!model->HFETAnmaxGiven) {
            model->HFETAnmax = 2e16;
        }
        if(!model->HFETAdeltadGiven) {
            model->HFETAdeltad = 4.5e-9;
        }
        if(!model->HFETAjs1dGiven) {
            model->HFETAjs1d = 1.0;
        }
        if(!model->HFETAjs2dGiven) {
            model->HFETAjs2d = 1.15e6;
        }
        if(!model->HFETAjs1sGiven) {
            model->HFETAjs1s = 1.0;
        }
        if(!model->HFETAjs2sGiven) {
            model->HFETAjs2s = 1.15e6;
        }
        if(!model->HFETAm1dGiven) {
            model->HFETAm1d = 1.32;
        }
        if(!model->HFETAm2dGiven) {
            model->HFETAm2d = 6.9;
        }
        if(!model->HFETAm1sGiven) {
            model->HFETAm1s = 1.32;
        }
        if(!model->HFETAm2sGiven) {
            model->HFETAm2s = 6.9;
        }
        if(!model->HFETArdGiven) {
            model->HFETArd = 0;
        }
        if(!model->HFETArsGiven) {
            model->HFETArs = 0;
        }
        if(!model->HFETArdiGiven) {
            model->HFETArdi = 0;
        }
        if(!model->HFETArsiGiven) {
            model->HFETArsi = 0;
        }
        if(!model->HFETArgsGiven) {
            model->HFETArgs = 90;
        }
        if(!model->HFETArgdGiven) {
            model->HFETArgd = 90;
        }
        if(!model->HFETAriGiven) {
            model->HFETAri = 0;
        }
        if(!model->HFETArfGiven) {
            model->HFETArf = 0;
        }
        if(!model->HFETAepsiGiven) {
            model->HFETAepsi = 12.244*8.85418e-12;
        }
        if(!model->HFETAa1Given) {
            model->HFETAa1 = 0;
        }
        if(!model->HFETAa2Given) {
            model->HFETAa2 = 0;
        }
        if(!model->HFETAmv1Given) {
            model->HFETAmv1 = 3;
        }
        if(!model->HFETApGiven) {
            model->HFETAp = 1;
        }
        if(!model->HFETAkappaGiven) {
            model->HFETAkappa = 0;
        }
        if(!model->HFETAdelfGiven) {
            model->HFETAdelf = 0;
        }
        if(!model->HFETAfgdsGiven) {
            model->HFETAfgds = 0;
        }
        if(!model->HFETAtfGiven) {
            model->HFETAtf = ckt->CKTtemp;
        }
        if(!model->HFETAcdsGiven) {
            model->HFETAcds = 0;
        }
        if(!model->HFETAphibGiven) {
            model->HFETAphib = 0.5*CHARGE;
        }
        if(!model->HFETAtalphaGiven) {
            model->HFETAtalpha = 1200;
        }
        if(!model->HFETAmt1Given) {
            model->HFETAmt1 = 3.5;
        }
        if(!model->HFETAmt2Given) {
            model->HFETAmt2 = 9.9;
        }
        if(!model->HFETAck1Given) {
            model->HFETAck1 = 1;
        }
        if(!model->HFETAck2Given) {
            model->HFETAck2 = 0;
        }
        if(!model->HFETAcm1Given) {
            model->HFETAcm1 = 3;
        }
        if(!model->HFETAcm2Given) {
            model->HFETAcm2 = 0;
        }
        if(!model->HFETAcm3Given) {
            model->HFETAcm3 = 0.17;
        }
        if(!model->HFETAastarGiven) {
            model->HFETAastar = 4.0e4;
        }
        if(!model->HFETAeta1Given) {
            model->HFETAeta1 = 2;
        }
        if(!model->HFETAd1Given) {
            model->HFETAd1 = 0.03e-6;
        }
        if(!model->HFETAeta2Given) {
            model->HFETAeta2 = 2;
        }
        if(!model->HFETAd2Given) {
            model->HFETAd2 = 0.2e-6;
        }
        if(!model->HFETAvt2Given) {
            /* initialized in HFETAtemp */
            model->HFETAvt2 = 0;
        }
        
        if(!model->HFETAggrGiven) {
            model->HFETAggr = 40;
        }
        if(!model->HFETAdelGiven) {
            model->HFETAdel = 0.04;
        }             
       if(!model->HFETAklambdaGiven)
         KLAMBDA = 0;
       if(!model->HFETAkmuGiven)
         KMU = 0;
       if(!model->HFETAkvtoGiven)
         KVTO = 0;
        
        /* loop through all the instances of the model */
        for (here = HFETAinstances(model); here != NULL ;
                here=HFETAnextInstance(here)) {
           
            if(!here->HFETAlengthGiven) {
                here->HFETAlength = 1e-6;
            }
            if(!here->HFETAwidthGiven) {
                here->HFETAwidth = 20e-6;
            }
            if(!here->HFETAmGiven) {
                here->HFETAm = 1.0;
            }
 
            here->HFETAstate = *states;
            *states += HFETAnumStates;

            if(model->HFETArs != 0) {
                if(here->HFETAsourcePrimeNode == 0) {
                error = CKTmkVolt(ckt,&tmp,here->HFETAname,"source");
                if(error) return(error);
                here->HFETAsourcePrimeNode = tmp->number;

/* XXX: Applied AlansFixes  */               
                 if (ckt->CKTcopyNodesets) {
		    CKTnode *tmpNode;
		    IFuid tmpName;

                  if (CKTinst2Node(ckt,here,3,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }
                }

            } else {
                here->HFETAsourcePrimeNode = here->HFETAsourceNode;
            }
            
            if(model->HFETArd != 0) {
                if(here->HFETAdrainPrimeNode == 0) {
                error = CKTmkVolt(ckt,&tmp,here->HFETAname,"drain");
                if(error) return(error);
                here->HFETAdrainPrimeNode = tmp->number;
       
                if (ckt->CKTcopyNodesets) {
		    CKTnode *tmpNode;
		    IFuid tmpName;

                  if (CKTinst2Node(ckt,here,1,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }
                }
                
            } else {
                here->HFETAdrainPrimeNode = here->HFETAdrainNode;
            }
            
            if(model->HFETArg != 0) {
                if(here->HFETAgatePrimeNode == 0) {
                error = CKTmkVolt(ckt,&tmp,here->HFETAname,"gate");
                if(error) return(error);
                here->HFETAgatePrimeNode = tmp->number;
         
                if (ckt->CKTcopyNodesets) {
		    CKTnode *tmpNode;
		    IFuid tmpName;

                  if (CKTinst2Node(ckt,here,2,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }     
                }     
                
            } else {
                here->HFETAgatePrimeNode = here->HFETAgateNode;
            }
            if(model->HFETArf != 0) {
                if(here->HFETAdrainPrmPrmNode == 0) {
                error = CKTmkVolt(ckt,&tmp,here->HFETAname,"gd");
                if(error) return(error);
                here->HFETAdrainPrmPrmNode = tmp->number;
         
                if (ckt->CKTcopyNodesets) {
		    CKTnode *tmpNode;
		    IFuid tmpName;

                  if (CKTinst2Node(ckt,here,1,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }
                }

            } else {
                here->HFETAdrainPrmPrmNode = here->HFETAdrainPrimeNode;
            }
            
            if(model->HFETAri != 0) {
                if(here->HFETAsourcePrmPrmNode == 0) {
                error = CKTmkVolt(ckt,&tmp,here->HFETAname,"gs");
                if(error) return(error);
                here->HFETAsourcePrmPrmNode = tmp->number;
      
                if (ckt->CKTcopyNodesets) {
		    CKTnode *tmpNode;
		    IFuid tmpName;

                  if (CKTinst2Node(ckt,here,3,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }
                }
                
            } else {
                here->HFETAsourcePrmPrmNode = here->HFETAsourcePrimeNode;
            }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

  TSTALLOC(HFETAdrainDrainPrimePtr,HFETAdrainNode,HFETAdrainPrimeNode);
  TSTALLOC(HFETAgatePrimeDrainPrimePtr,HFETAgatePrimeNode,HFETAdrainPrimeNode);
  TSTALLOC(HFETAgatePrimeSourcePrimePtr,HFETAgatePrimeNode,HFETAsourcePrimeNode);
  TSTALLOC(HFETAsourceSourcePrimePtr,HFETAsourceNode,HFETAsourcePrimeNode);
  TSTALLOC(HFETAdrainPrimeDrainPtr,HFETAdrainPrimeNode,HFETAdrainNode);
  TSTALLOC(HFETAdrainPrimeGatePrimePtr,HFETAdrainPrimeNode,HFETAgatePrimeNode);
  TSTALLOC(HFETAdrainPrimeSourcePrimePtr,HFETAdrainPrimeNode,HFETAsourcePrimeNode);
  TSTALLOC(HFETAsourcePrimeGatePrimePtr,HFETAsourcePrimeNode,HFETAgatePrimeNode);
  TSTALLOC(HFETAsourcePrimeSourcePtr,HFETAsourcePrimeNode,HFETAsourceNode);
  TSTALLOC(HFETAsourcePrimeDrainPrimePtr,HFETAsourcePrimeNode,HFETAdrainPrimeNode);
  TSTALLOC(HFETAdrainDrainPtr,HFETAdrainNode,HFETAdrainNode);
  TSTALLOC(HFETAgatePrimeGatePrimePtr,HFETAgatePrimeNode,HFETAgatePrimeNode);
  TSTALLOC(HFETAsourceSourcePtr,HFETAsourceNode,HFETAsourceNode);
  TSTALLOC(HFETAdrainPrimeDrainPrimePtr,HFETAdrainPrimeNode,HFETAdrainPrimeNode);
  TSTALLOC(HFETAsourcePrimeSourcePrimePtr,HFETAsourcePrimeNode,HFETAsourcePrimeNode);
  TSTALLOC(HFETAdrainPrimeDrainPrmPrmPtr,HFETAdrainPrimeNode,HFETAdrainPrmPrmNode);
  TSTALLOC(HFETAdrainPrmPrmDrainPrimePtr,HFETAdrainPrmPrmNode,HFETAdrainPrimeNode);
  TSTALLOC(HFETAdrainPrmPrmGatePrimePtr,HFETAdrainPrmPrmNode,HFETAgatePrimeNode);
  TSTALLOC(HFETAgatePrimeDrainPrmPrmPtr,HFETAgatePrimeNode,HFETAdrainPrmPrmNode);
  TSTALLOC(HFETAdrainPrmPrmDrainPrmPrmPtr,HFETAdrainPrmPrmNode,HFETAdrainPrmPrmNode);
  TSTALLOC(HFETAsourcePrimeSourcePrmPrmPtr,HFETAsourcePrimeNode,HFETAsourcePrmPrmNode);
  TSTALLOC(HFETAsourcePrmPrmSourcePrimePtr,HFETAsourcePrmPrmNode,HFETAsourcePrimeNode);
  TSTALLOC(HFETAsourcePrmPrmGatePrimePtr,HFETAsourcePrmPrmNode,HFETAgatePrimeNode);
  TSTALLOC(HFETAgatePrimeSourcePrmPrmPtr,HFETAgatePrimeNode,HFETAsourcePrmPrmNode);
  TSTALLOC(HFETAsourcePrmPrmSourcePrmPrmPtr,HFETAsourcePrmPrmNode,HFETAsourcePrmPrmNode);
  TSTALLOC(HFETAgateGatePtr,HFETAgateNode,HFETAgateNode);
  TSTALLOC(HFETAgateGatePrimePtr,HFETAgateNode,HFETAgatePrimeNode);
  TSTALLOC(HFETAgatePrimeGatePtr,HFETAgatePrimeNode,HFETAgateNode);
 }
 }
 return(OK);
}

int
HFETAunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    HFETAmodel *model;
    HFETAinstance *here;
 
    for (model = (HFETAmodel *)inModel; model != NULL;
            model = HFETAnextModel(model))
    {
        for (here = HFETAinstances(model); here != NULL;
                here=HFETAnextInstance(here))
        {
	    if (here->HFETAsourcePrmPrmNode > 0
        			&& here->HFETAsourcePrmPrmNode != here->HFETAsourcePrimeNode)
        		CKTdltNNum(ckt, here->HFETAsourcePrmPrmNode);
            here->HFETAsourcePrmPrmNode = 0;

	    if (here->HFETAdrainPrmPrmNode > 0
        			&& here->HFETAdrainPrmPrmNode != here->HFETAdrainPrimeNode)
        		CKTdltNNum(ckt, here->HFETAdrainPrmPrmNode);
            here->HFETAdrainPrmPrmNode = 0;

	    if (here->HFETAgatePrimeNode > 0
        			&& here->HFETAgatePrimeNode != here->HFETAgateNode)
        		CKTdltNNum(ckt, here->HFETAgatePrimeNode);
            here->HFETAgatePrimeNode = 0;
	    
            if (here->HFETAdrainPrimeNode > 0
                    && here->HFETAdrainPrimeNode != here->HFETAdrainNode)
                CKTdltNNum(ckt, here->HFETAdrainPrimeNode);
            here->HFETAdrainPrimeNode = 0;

            if (here->HFETAsourcePrimeNode > 0
                    && here->HFETAsourcePrimeNode != here->HFETAsourceNode)
                CKTdltNNum(ckt, here->HFETAsourcePrimeNode);
            here->HFETAsourcePrimeNode = 0;
	    
        }	
    }
    return OK;
}

