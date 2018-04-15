/**********
Imported from MacSpice3f4 - Antony Wilson
Modified: Paolo Nenzi
**********/

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "hfet2defs.h"
#include "ngspice/const.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int HFET2setup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *states)
{
    
  HFET2model *model = (HFET2model*)inModel;
  HFET2instance *here;
  int error;
  CKTnode *tmp;

  for( ; model != NULL; model = HFET2nextModel(model)) {
    if((TYPE != NHFET) && (TYPE != PHFET) )
      TYPE = NHFET;
    if(!model->HFET2cfGiven)
      CF = 0;
    if(!model->HFET2d1Given)
      D1 = 0.03e-6;
    if(!model->HFET2d2Given)
      D2 = 0.2e-6;
    if(!model->HFET2delGiven)
      DEL = 0.04;
    if(!model->HFET2deltaGiven)
      DELTA = 3.0;
    if(!model->HFET2deltadGiven)
      DELTAD = 4.5e-9;
    if(!model->HFET2diGiven)
      DI = 0.04e-6;
    if(!model->HFET2epsiGiven)
      EPSI = 12.244*8.85418e-12;
    if(!model->HFET2etaGiven)
      {
        if(TYPE == NHFET)
          ETA = 1.28;
        else
          ETA = 1.4;
      }
    if(!model->HFET2eta1Given)
      ETA1 = 2;
    if(!model->HFET2eta2Given)
      ETA2 = 2;
    if(!model->HFET2gammaGiven)
      GAMMA = 3.0;
    if(!model->HFET2ggrGiven)
      GGR = 0;
    if(!model->HFET2jsGiven)
      JS = 0;
    if(!model->HFET2klambdaGiven)
      KLAMBDA = 0;
    if(!model->HFET2kmuGiven)
      KMU = 0;
    if(!model->HFET2knmaxGiven)
      KNMAX = 0;
    if(!model->HFET2kvtoGiven)
      KVTO = 0;
    if(!model->HFET2lambdaGiven)
      LAMBDA = 0.15;
    if(!model->HFET2mGiven)
      M = 3.0;
    if(!model->HFET2mcGiven)
      MC = 3.0;
    if(!model->HFET2muGiven)
      {
        if(TYPE == NHFET)
          MU = 0.4;
        else
          MU = 0.03;
      }
    if(!model->HFET2nGiven)
      N = 5.0;
    if(!model->HFET2nmaxGiven)
      NMAX = 2e16;
    if(!model->HFET2pGiven)
      PP = 1;
    if(!model->HFET2rdGiven)
      RD = 0;
    if(!model->HFET2rdiGiven)
      RDI = 0;
    if(!model->HFET2rsGiven)
      RS = 0;
    if(!model->HFET2rsiGiven)
      RSI = 0;
    if(!model->HFET2sigma0Given)
      SIGMA0 = 0.057;
    if(!model->HFET2vsGiven)
      {
        if(TYPE == NHFET)
          VS = 1.5e5;
        else
          VS = 0.8e5;
      }
    if(!model->HFET2vsigmaGiven)
      VSIGMA = 0.1;
    if(!model->HFET2vsigmatGiven)
      VSIGMAT = 0.3;
    if(!model->HFET2vt1Given)
      /* initialized in HFET2temp */
      HFET2_VT1 = 0;
    if(!model->HFET2vt2Given)
      /* initialized in HFET2temp */
      VT2 = 0;
    if(!model->HFET2vtoGiven) {
      if(model->HFET2type == NHFET)
        VTO = 0.15;
      else
        VTO = -0.15;
    }
   
    /* loop through all the instances of the model */
   
    
    for (here = HFET2instances(model); here != NULL; 
         here=HFET2nextInstance(here)) {
      
      CKTnode *tmpNode;
      IFuid tmpName;
   
      here->HFET2state = *states;
      *states += HFET2numStates;
      
      if(!here->HFET2lengthGiven)
        L = 1e-6;      
      if(!here->HFET2widthGiven)
        W = 20e-6;
      if(!here->HFET2mGiven)
        here->HFET2m = 1.0;

      if(model->HFET2rs != 0) {
          if(here->HFET2sourcePrimeNode == 0) {
        error = CKTmkVolt(ckt,&tmp,here->HFET2name,"source");
        if(error) return(error);
        here->HFET2sourcePrimeNode = tmp->number;
        
        if (ckt->CKTcopyNodesets) {
                  if (CKTinst2Node(ckt,here,3,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }
          }
        
      } else {
        here->HFET2sourcePrimeNode = here->HFET2sourceNode;
      }
      if(model->HFET2rd != 0) {
          if(here->HFET2drainPrimeNode == 0) {
        error = CKTmkVolt(ckt,&tmp,here->HFET2name,"drain");
        if(error) return(error);
        here->HFET2drainPrimeNode = tmp->number;
        
        if (ckt->CKTcopyNodesets) {
                  if (CKTinst2Node(ckt,here,1,&tmpNode,&tmpName)==OK) {
                     if (tmpNode->nsGiven) {
                       tmp->nodeset=tmpNode->nodeset; 
                       tmp->nsGiven=tmpNode->nsGiven; 
                     }
                  }
                }
          }
        
      } else {
        here->HFET2drainPrimeNode = here->HFET2drainNode;
      }

/* macro to make elements with built in test for out of memory */
#define TSTALLOC(ptr,first,second) \
do { if((here->ptr = SMPmakeElt(matrix, here->first, here->second)) == NULL){\
    return(E_NOMEM);\
} } while(0)

    TSTALLOC(HFET2drainDrainPrimePtr,HFET2drainNode,HFET2drainPrimeNode);
    TSTALLOC(HFET2gateDrainPrimePtr,HFET2gateNode,HFET2drainPrimeNode);
    TSTALLOC(HFET2gateSourcePrimePtr,HFET2gateNode,HFET2sourcePrimeNode);
    TSTALLOC(HFET2sourceSourcePrimePtr,HFET2sourceNode,HFET2sourcePrimeNode);
    TSTALLOC(HFET2drainPrimeDrainPtr,HFET2drainPrimeNode,HFET2drainNode);
    TSTALLOC(HFET2drainPrimeGatePtr,HFET2drainPrimeNode,HFET2gateNode);
    TSTALLOC(HFET2drainPriHFET2ourcePrimePtr,HFET2drainPrimeNode,HFET2sourcePrimeNode);
    TSTALLOC(HFET2sourcePrimeGatePtr,HFET2sourcePrimeNode,HFET2gateNode);
    TSTALLOC(HFET2sourcePriHFET2ourcePtr,HFET2sourcePrimeNode,HFET2sourceNode);
    TSTALLOC(HFET2sourcePrimeDrainPrimePtr,HFET2sourcePrimeNode,HFET2drainPrimeNode);
    TSTALLOC(HFET2drainDrainPtr,HFET2drainNode,HFET2drainNode);
    TSTALLOC(HFET2gateGatePtr,HFET2gateNode,HFET2gateNode);
    TSTALLOC(HFET2sourceSourcePtr,HFET2sourceNode,HFET2sourceNode);
    TSTALLOC(HFET2drainPrimeDrainPrimePtr,HFET2drainPrimeNode,HFET2drainPrimeNode);
    TSTALLOC(HFET2sourcePriHFET2ourcePrimePtr,HFET2sourcePrimeNode,HFET2sourcePrimeNode);
  
    }
  }
  return(OK);
  
}


int
HFET2unsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    HFET2model *model;
    HFET2instance *here;
 
    for (model = (HFET2model *)inModel; model != NULL;
            model = HFET2nextModel(model))
    {
        for (here = HFET2instances(model); here != NULL;
                here=HFET2nextInstance(here))
        {
            if (here->HFET2drainPrimeNode > 0
                    && here->HFET2drainPrimeNode != here->HFET2drainNode)
                CKTdltNNum(ckt, here->HFET2drainPrimeNode);
            here->HFET2drainPrimeNode = 0;

            if (here->HFET2sourcePrimeNode > 0
                    && here->HFET2sourcePrimeNode != here->HFET2sourceNode)
                CKTdltNNum(ckt, here->HFET2sourcePrimeNode);
            here->HFET2sourcePrimeNode = 0;
        }
    
    }
    return OK;
}
