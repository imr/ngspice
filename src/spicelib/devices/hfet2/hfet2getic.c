/**********
Imported from MacSpice3f4 - Antony Wilson
Modified: Paolo Nenzi
**********/

#include "ngspice.h"
#include "cktdefs.h"
#include "hfet2defs.h"
#include "sperror.h"
#include "suffix.h"


int HFET2getic(GENmodel *inModel, CKTcircuit *ckt)
{

  HFET2model *model = (HFET2model*)inModel;
  HFET2instance *here;

  for( ; model ; model = model->HFET2nextModel) {
    for(here = model->HFET2instances; here ; here = here->HFET2nextInstance) {
        if (here->HFET2owner != ARCHme) continue;

      if(!here->HFET2icVDSGiven) {
        here->HFET2icVDS = *(ckt->CKTrhs + here->HFET2drainNode) - 
                           *(ckt->CKTrhs + here->HFET2sourceNode);
      }
      if(!here->HFET2icVGSGiven) {
        here->HFET2icVGS = *(ckt->CKTrhs + here->HFET2gateNode) - 
                           *(ckt->CKTrhs + here->HFET2sourceNode);
      }
    }
  }
  return(OK);
  
}
