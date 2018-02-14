/**********
Author: 2003 Paolo Nenzi
**********/
/*
 */


#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "hfet2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int HFET2pzLoad(GENmodel *inModel, CKTcircuit *ckt, SPcomplex *s)
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

  double m;

  for( ; model != NULL; model = HFET2nextModel(model)) 
  {
    for( here = HFET2instances(model); here != NULL; 
         here = HFET2nextInstance(here)) {

      m = here->HFET2m;

      gdpr=model->HFET2drainConduct;
      gspr=model->HFET2sourceConduct;
      gm= *(ckt->CKTstate0 + here->HFET2gm) ;
      gds= *(ckt->CKTstate0 + here->HFET2gds) ;
      ggs= *(ckt->CKTstate0 + here->HFET2ggs) ;
      xgs= *(ckt->CKTstate0 + here->HFET2qgs) ;
      ggd= *(ckt->CKTstate0 + here->HFET2ggd) ;
      xgd= *(ckt->CKTstate0 + here->HFET2qgd) ;

      *(here->HFET2drainDrainPtr )                 += m * (gdpr);
      *(here->HFET2gateGatePtr )                   += m * (ggd+ggs);
      *(here->HFET2gateGatePtr)                    += m * ((xgd+xgs) * s->real);  
      *(here->HFET2gateGatePtr +1)                 += m * ((xgd+xgs) * s->imag);      
      *(here->HFET2sourceSourcePtr )               += m * (gspr);
      *(here->HFET2drainPrimeDrainPrimePtr )       += m * (gdpr+gds+ggd);
      *(here->HFET2drainPrimeDrainPrimePtr)        += m * (xgd * s->real);
      *(here->HFET2drainPrimeDrainPrimePtr +1)     += m * (xgd * s->imag);
      *(here->HFET2sourcePriHFET2ourcePrimePtr )   += m * (gspr+gds+gm+ggs);
      *(here->HFET2sourcePriHFET2ourcePrimePtr)    += m * (xgs * s->real);
      *(here->HFET2sourcePriHFET2ourcePrimePtr +1) += m * (xgs * s->imag);
      *(here->HFET2drainDrainPrimePtr )            -= m * (gdpr);
      *(here->HFET2gateDrainPrimePtr )             -= m * (ggd);
      *(here->HFET2gateDrainPrimePtr)              -= m * (xgd * s->real);
      *(here->HFET2gateDrainPrimePtr +1)           -= m * (xgd * s->imag);
      *(here->HFET2gateSourcePrimePtr )            -= m * (ggs);
      *(here->HFET2gateSourcePrimePtr)             -= m * (xgs * s->real);
      *(here->HFET2gateSourcePrimePtr +1)          -= m * (xgs * s->imag);
      *(here->HFET2sourceSourcePrimePtr )          -= m * (gspr);
      *(here->HFET2drainPrimeDrainPtr )            -= m * (gdpr);
      *(here->HFET2drainPrimeGatePtr )             += m * (-ggd+gm);
      *(here->HFET2drainPrimeGatePtr)              -= m * (xgd * s->real);
      *(here->HFET2drainPrimeGatePtr +1)           -= m * (xgd * s->imag);
      *(here->HFET2drainPriHFET2ourcePrimePtr )    += m * (-gds-gm);
      *(here->HFET2sourcePrimeGatePtr )            += m * (-ggs-gm);
      *(here->HFET2sourcePrimeGatePtr)             -= m * (xgs * s->real);
      *(here->HFET2sourcePrimeGatePtr +1)          -= m * (xgs * s->imag);
      *(here->HFET2sourcePriHFET2ourcePtr )        -= m * (gspr);
      *(here->HFET2sourcePrimeDrainPrimePtr )      -= m * (gds);
    }
  }
  return(OK);
  
}
