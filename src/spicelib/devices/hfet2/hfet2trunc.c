#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/cktdefs.h"
#include "hfet2defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


int HFET2trunc(
GENmodel *inModel,
CKTcircuit *ckt,
double *tiHFET2tep)
{

  HFET2model *model = (HFET2model*)inModel;
  HFET2instance *here;

  for( ; model != NULL; model = HFET2nextModel(model)) {
    for(here=HFET2instances(model);here!=NULL;
        here = HFET2nextInstance(here)){

      CKTterr(here->HFET2qgs,ckt,tiHFET2tep);
      CKTterr(here->HFET2qgd,ckt,tiHFET2tep);
    }
  }
  return(OK);
  
}
