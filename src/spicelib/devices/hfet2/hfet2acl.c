
#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "hfet2defs.h"
#include "sperror.h"
#include "suffix.h"


int HFET2acLoad(inModel, ckt)
GENmodel *inModel;
CKTcircuit *ckt;
{
  
  HFET2model *model = (HFET2model*)inModel;
  HFET2instance *here;
  double gdpr;
  double gspr;
  double gm;
  double gds;
  double ggs;
  double xgs;
  double ggd;
  double xgd;

  for( ; model != NULL; model = model->HFET2nextModel ) 
  {
    for( here = model->HFET2instances; here != NULL; here = here->HFET2nextInstance) 
    {
      gdpr=model->HFET2drainConduct;
      gspr=model->HFET2sourceConduct;
      gm= *(ckt->CKTstate0 + here->HFET2gm) ;
      gds= *(ckt->CKTstate0 + here->HFET2gds) ;
      ggs= *(ckt->CKTstate0 + here->HFET2ggs) ;
      xgs= *(ckt->CKTstate0 + here->HFET2qgs) * ckt->CKTomega ;
      ggd= *(ckt->CKTstate0 + here->HFET2ggd) ;
      xgd= *(ckt->CKTstate0 + here->HFET2qgd) * ckt->CKTomega ;
      *(here->HFET2drainDrainPtr ) += gdpr;
      *(here->HFET2gateGatePtr ) += ggd+ggs;
      *(here->HFET2gateGatePtr +1) += xgd+xgs;
      *(here->HFET2sourceSourcePtr ) += gspr;
      *(here->HFET2drainPrimeDrainPrimePtr ) += gdpr+gds+ggd;
      *(here->HFET2drainPrimeDrainPrimePtr +1) += xgd;
      *(here->HFET2sourcePriHFET2ourcePrimePtr ) += gspr+gds+gm+ggs;
      *(here->HFET2sourcePriHFET2ourcePrimePtr +1) += xgs;
      *(here->HFET2drainDrainPrimePtr ) -= gdpr;
      *(here->HFET2gateDrainPrimePtr ) -= ggd;
      *(here->HFET2gateDrainPrimePtr +1) -= xgd;
      *(here->HFET2gateSourcePrimePtr ) -= ggs;
      *(here->HFET2gateSourcePrimePtr +1) -= xgs;
      *(here->HFET2sourceSourcePrimePtr ) -= gspr;
      *(here->HFET2drainPrimeDrainPtr ) -= gdpr;
      *(here->HFET2drainPrimeGatePtr ) += (-ggd+gm);
      *(here->HFET2drainPrimeGatePtr +1) -= xgd;
      *(here->HFET2drainPriHFET2ourcePrimePtr ) += (-gds-gm);
      *(here->HFET2sourcePrimeGatePtr ) += (-ggs-gm);
      *(here->HFET2sourcePrimeGatePtr +1) -= xgs;
      *(here->HFET2sourcePriHFET2ourcePtr ) -= gspr;
      *(here->HFET2sourcePrimeDrainPrimePtr ) -= gds;
    }
  }
  return(OK);
  
}
