
#include "ngspice.h"
#include <stdio.h>
#include "cktdefs.h"
#include "hfet2defs.h"
#include "sperror.h"
#include "suffix.h"


int HFET2trunc(inModel, ckt, tiHFET2tep)
GENmodel *inModel;
CKTcircuit *ckt;
double *tiHFET2tep;
{

  HFET2model *model = (HFET2model*)inModel;
  HFET2instance *here;

  for( ; model != NULL; model = model->HFET2nextModel) {
    for(here=model->HFET2instances;here!=NULL;
        here = here->HFET2nextInstance){
      
       if (here->HFET2owner != ARCHme) continue;

      CKTterr(here->HFET2qgs,ckt,tiHFET2tep);
      CKTterr(here->HFET2qgd,ckt,tiHFET2tep);
    }
  }
  return(OK);
  
}
