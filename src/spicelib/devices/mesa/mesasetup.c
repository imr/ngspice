/**********
Copyright 1993: T. Ytterdal, K. Lee, M. Shur and T. A. Fjeldly. All rights reserved.
Author: Trond Ytterdal
Modified: 2001 AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "mesadefs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int
MESAsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)

        /* load the diode structure with those pointers needed later 
         * for fast matrix loading 
         */
{
    MESAmodel *model = (MESAmodel*)inModel;
    MESAinstance *here;
    int error;
    CKTnode *tmp;


    /*  loop through all the diode models */
    for( ; model != NULL; model = MESAnextModel(model)) {
        if( (model->MESAtype != NMF) ) {
            fprintf(stderr, "Only nmf model type supported, set to nmf\n");
            model->MESAtype = NMF;
        }
        if(!model->MESAthresholdGiven) {
            model->MESAthreshold = -1.26;
        }
        if(!model->MESAdGiven) {
            model->MESAd = 0.12e-6;
        }
        if(!model->MESAduGiven) {
            model->MESAdu = 0.035e-6;
        }
        if(!model->MESAlambdaGiven) {
            model->MESAlambda = 0.045;
        }
        if(!model->MESAvsGiven) {
            model->MESAvs = 1.5e5;
        }
        if(!model->MESAbetaGiven) {
            model->MESAbeta = 0.0085;
        }
        if(!model->MESAetaGiven) {
            model->MESAeta = 1.73;
        }
        if(!model->MESAmGiven) {
            model->MESAm = 2.5;
        }
        if(!model->MESAmcGiven) {
            model->MESAmc = 3.0;
        }
        if(!model->MESAalphaGiven) {
            model->MESAalpha = 0.0;
        }
        if(!model->MESAsigma0Given) {
            model->MESAsigma0 = 0.081;
        }
        if(!model->MESAvsigmatGiven) {
            model->MESAvsigmat = 1.01;
        }
        if(!model->MESAvsigmaGiven) {
            model->MESAvsigma = 0.1;
        }
        if(!model->MESAmuGiven) {
            model->MESAmu = 0.23;
        }
        if(!model->MESAthetaGiven) {
            model->MESAtheta = 0;
        }
        if(!model->MESAmu1Given) {
            model->MESAmu1 = 0;
        }
        if(!model->MESAmu2Given) {
            model->MESAmu2 = 0;
        }
        if(!model->MESAndGiven) {
            model->MESAnd = 2.0e23;
        }
        if(!model->MESAnduGiven) {
            model->MESAndu = 1e22;
        }
        if(!model->MESAndeltaGiven) {
            model->MESAndelta = 6e24;
        }
        if(!model->MESAthGiven) {
            model->MESAth = 0.01e-6;
        }
        if(!model->MESAdeltaGiven) {
            model->MESAdelta = 5.0;
        }
        if(!model->MESAtcGiven) {
            model->MESAtc = 0.0;
        }
        if(!model->MESAdrainResistGiven) {
            model->MESAdrainResist = 0;
        }
        if(!model->MESAsourceResistGiven) {
            model->MESAsourceResist = 0;
        }
        if(!model->MESAgateResistGiven) {
            model->MESAgateResist = 0;
        }
        if(!model->MESAriGiven) {
            model->MESAri = 0;
        }
        if(!model->MESArfGiven) {
            model->MESArf = 0;
        }
        if(!model->MESArdiGiven) {
            model->MESArdi = 0;
        }
        if(!model->MESArsiGiven) {
            model->MESArsi = 0;
        }
        if(!model->MESAphibGiven) {
            model->MESAphib = 0.5*CHARGE;
        }
        if(!model->MESAphib1Given) {
            model->MESAphib1 = 0;
        }
        if(!model->MESAastarGiven) {
            model->MESAastar = 4.0e4;
        }
        if(!model->MESAggrGiven) {
            model->MESAggr = 40;
        }
        if(!model->MESAdelGiven) {
            model->MESAdel = 0.04;
        }
        if(!model->MESAxchiGiven) {
            model->MESAxchi = 0.033;
        }
        if(!model->MESAnGiven) {
            model->MESAn = 1;
        }
        if(!model->MESAtvtoGiven) {
            model->MESAtvto = 0;
        }
        if(!model->MESAtlambdaGiven) {
            model->MESAtlambda = DBL_MAX;
        }
        if(!model->MESAteta0Given) {
            model->MESAteta0 = DBL_MAX;
        }
        if(!model->MESAteta1Given) {
            model->MESAteta1 = 0;
        }
        if(!model->MESAtmuGiven) {
            model->MESAtmu = 300.15;
        }
        if(!model->MESAxtm0Given) {
            model->MESAxtm0 = 0;
        }
        if(!model->MESAxtm1Given) {
            model->MESAxtm1 = 0;
        }
        if(!model->MESAxtm2Given) {
            model->MESAxtm2 = 0;
        }
        if(!model->MESAksGiven) {
            model->MESAks = 0;
        }
        if(!model->MESAvsgGiven) {
            model->MESAvsg = 0;
        }
        if(!model->MESAtfGiven) {
            model->MESAtf = ckt->CKTtemp;
        }
        if(!model->MESAfloGiven) {
            model->MESAflo = 0;
        }
        if(!model->MESAdelfoGiven) {
            model->MESAdelfo = 0;
        }
        if(!model->MESAagGiven) {
            model->MESAag = 0;
        }
        if(!model->MESAtc1Given) {
            model->MESAtc1 = 0;
        }
        if(!model->MESAtc2Given) {
            model->MESAtc2 = 0;
        }
        if(!model->MESAzetaGiven) {
            model->MESAzeta = 1;
        }
        if(!model->MESAlevelGiven) {
            model->MESAlevel = 2;
        }
        if(!model->MESAnmaxGiven) {
            model->MESAnmax = 2e16;
        }
        if(!model->MESAgammaGiven) {
            model->MESAgamma = 3.0;
        }
        if(!model->MESAepsiGiven) {
            model->MESAepsi = 12.244*8.85418e-12;
        }
        if(!model->MESAcasGiven) {
            model->MESAcas = 1;
        }                      
        if(!model->MESAcbsGiven) {
            model->MESAcbs = 1;
        }

        if(model->MESAdrainResist != 0) {
            model->MESAdrainConduct = 1./model->MESAdrainResist;
        } else {
            model->MESAdrainConduct = DBL_MAX;
        }
        if(model->MESAsourceResist != 0) {
            model->MESAsourceConduct = 1./model->MESAsourceResist;
        } else {
            model->MESAsourceConduct = DBL_MAX;
        }

        model->MESAvcrit = 0.; /* until model has changed */

        /* loop through all the instances of the model */
        for (here = MESAinstances(model); here != NULL ;
                here=MESAnextInstance(here)) {
         
            if(!here->MESAlengthGiven) {
                here->MESAlength = 1e-6;
            }
            if(!here->MESAwidthGiven) {
                here->MESAwidth = 20e-6;
            }
            if(!here->MESAmGiven) {
                here->MESAm = 1.0;
            }
            if(!here->MESAdtempGiven) {
                here->MESAdtemp = 0.0;
            }
            if(!here->MESAtdGiven) {
                here->MESAtd = ckt->CKTtemp + here->MESAdtemp;
            }
            if(!here->MESAtsGiven) {
                here->MESAts = ckt->CKTtemp + here->MESAdtemp;
            }


            here->MESAstate = *states;
            *states += MESAnumStates;

            if(model->MESAsourceResist != 0) {
                if(here->MESAsourcePrimeNode == 0) {
                error = CKTmkVolt(ckt,&tmp,here->MESAname,"source");
                if(error) return(error);
                here->MESAsourcePrimeNode = tmp->number;
                
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
                here->MESAsourcePrimeNode = here->MESAsourceNode;
            }
            
            if(model->MESAdrainResist != 0) {
                if(here->MESAdrainPrimeNode == 0) {
                error = CKTmkVolt(ckt,&tmp,here->MESAname,"drain");
                if(error) return(error);
                here->MESAdrainPrimeNode = tmp->number;
                
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
                here->MESAdrainPrimeNode = here->MESAdrainNode;
            }
            if(model->MESAgateResist != 0) {
                if(here->MESAgatePrimeNode == 0) {
                error = CKTmkVolt(ckt,&tmp,here->MESAname,"gate");
                if(error) return(error);
                here->MESAgatePrimeNode = tmp->number;
                
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
                here->MESAgatePrimeNode = here->MESAgateNode;
            }
            
            
            if(model->MESAri != 0) {
                if(here->MESAsourcePrmPrmNode == 0) {
                error = CKTmkVolt(ckt,&tmp,here->MESAname,"gs");
                if(error) return(error);
                here->MESAsourcePrmPrmNode = tmp->number;
                
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
                here->MESAsourcePrmPrmNode = here->MESAsourcePrimeNode;
            }
            if(model->MESArf != 0) {
                if(here->MESAdrainPrmPrmNode == 0) {
                error = CKTmkVolt(ckt,&tmp,here->MESAname,"gd");
                if(error) return(error);
                here->MESAdrainPrmPrmNode = tmp->number;
                
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
                here->MESAdrainPrmPrmNode = here->MESAdrainPrimeNode;
            }

#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

  TSTALLOC(MESAdrainDrainPtr,MESAdrainNode,MESAdrainNode);
  TSTALLOC(MESAdrainPrimeDrainPrimePtr,MESAdrainPrimeNode,MESAdrainPrimeNode);
  TSTALLOC(MESAdrainPrmPrmDrainPrmPrmPtr,MESAdrainPrmPrmNode,MESAdrainPrmPrmNode);
  TSTALLOC(MESAgateGatePtr,MESAgateNode,MESAgateNode);
  TSTALLOC(MESAgatePrimeGatePrimePtr,MESAgatePrimeNode,MESAgatePrimeNode);
  TSTALLOC(MESAsourceSourcePtr,MESAsourceNode,MESAsourceNode);
  TSTALLOC(MESAsourcePrimeSourcePrimePtr,MESAsourcePrimeNode,MESAsourcePrimeNode);
  TSTALLOC(MESAsourcePrmPrmSourcePrmPrmPtr,MESAsourcePrmPrmNode,MESAsourcePrmPrmNode);
  TSTALLOC(MESAdrainDrainPrimePtr,MESAdrainNode,MESAdrainPrimeNode);
  TSTALLOC(MESAdrainPrimeDrainPtr,MESAdrainPrimeNode,MESAdrainNode);
  TSTALLOC(MESAgatePrimeDrainPrimePtr,MESAgatePrimeNode,MESAdrainPrimeNode);
  TSTALLOC(MESAdrainPrimeGatePrimePtr,MESAdrainPrimeNode,MESAgatePrimeNode);
  TSTALLOC(MESAgatePrimeSourcePrimePtr,MESAgatePrimeNode,MESAsourcePrimeNode);
  TSTALLOC(MESAsourcePrimeGatePrimePtr,MESAsourcePrimeNode,MESAgatePrimeNode)  ;
  TSTALLOC(MESAsourceSourcePrimePtr,MESAsourceNode,MESAsourcePrimeNode);
  TSTALLOC(MESAsourcePrimeSourcePtr,MESAsourcePrimeNode,MESAsourceNode);
  TSTALLOC(MESAdrainPrimeSourcePrimePtr,MESAdrainPrimeNode,MESAsourcePrimeNode);
  TSTALLOC(MESAsourcePrimeDrainPrimePtr,MESAsourcePrimeNode,MESAdrainPrimeNode);
  TSTALLOC(MESAgatePrimeGatePtr,MESAgatePrimeNode,MESAgateNode);
  TSTALLOC(MESAgateGatePrimePtr,MESAgateNode,MESAgatePrimeNode);
  TSTALLOC(MESAsourcePrmPrmSourcePrimePtr,MESAsourcePrmPrmNode,MESAsourcePrimeNode);
  TSTALLOC(MESAsourcePrimeSourcePrmPrmPtr,MESAsourcePrimeNode,MESAsourcePrmPrmNode);
  TSTALLOC(MESAsourcePrmPrmGatePrimePtr,MESAsourcePrmPrmNode,MESAgatePrimeNode);
  TSTALLOC(MESAgatePrimeSourcePrmPrmPtr,MESAgatePrimeNode,MESAsourcePrmPrmNode);
  TSTALLOC(MESAdrainPrmPrmDrainPrimePtr,MESAdrainPrmPrmNode,MESAdrainPrimeNode);
  TSTALLOC(MESAdrainPrimeDrainPrmPrmPtr,MESAdrainPrimeNode,MESAdrainPrmPrmNode);
  TSTALLOC(MESAdrainPrmPrmGatePrimePtr,MESAdrainPrmPrmNode,MESAgatePrimeNode);
  TSTALLOC(MESAgatePrimeDrainPrmPrmPtr,MESAgatePrimeNode,MESAdrainPrmPrmNode);
  }
  }
    return(OK);
}


int
MESAunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    MESAmodel *model;
    MESAinstance *here;
 
    for (model = (MESAmodel *)inModel; model != NULL;
            model = MESAnextModel(model))
    {
        for (here = MESAinstances(model); here != NULL;
                here=MESAnextInstance(here))
        {
            if (here->MESAdrainPrmPrmNode > 0
                    && here->MESAdrainPrmPrmNode != here->MESAdrainPrimeNode)
                CKTdltNNum(ckt, here->MESAdrainPrmPrmNode);
            here->MESAdrainPrmPrmNode = 0;

            if (here->MESAsourcePrmPrmNode > 0
                    && here->MESAsourcePrmPrmNode != here->MESAsourcePrimeNode)
                CKTdltNNum(ckt, here->MESAsourcePrmPrmNode);
            here->MESAsourcePrmPrmNode = 0;

            if (here->MESAgatePrimeNode > 0
                    && here->MESAgatePrimeNode != here->MESAgateNode)
                CKTdltNNum(ckt, here->MESAgatePrimeNode);
            here->MESAgatePrimeNode = 0;

            if (here->MESAdrainPrimeNode > 0
                    && here->MESAdrainPrimeNode != here->MESAdrainNode)
                CKTdltNNum(ckt, here->MESAdrainPrimeNode);
            here->MESAdrainPrimeNode = 0;

            if (here->MESAsourcePrimeNode > 0
                    && here->MESAsourcePrimeNode != here->MESAsourceNode)
                CKTdltNNum(ckt, here->MESAsourcePrimeNode);
            here->MESAsourcePrimeNode = 0;

        
        }
    }
    return OK;
}
