/**********
Imported from MacSpice3f4 - Antony Wilson
Modified: Paolo Nenzi
**********/

#include "ngspice.h"
#include "hfet2defs.h"
#include "suffix.h"


void HFET2destroy(GENmodel **inModel)
{

  HFET2model **model = (HFET2model**)inModel;
  HFET2instance *here;
  HFET2instance *prev = NULL;
  HFET2model *mod = *model;
  HFET2model *oldmod = NULL;

  for( ; mod ; mod = mod->HFET2nextModel) {
    if(oldmod) FREE(oldmod);
    oldmod = mod;
    prev = (HFET2instance *)NULL;
    for(here = mod->HFET2instances ; here ; here = here->HFET2nextInstance) {
      if(prev) FREE(prev);
      prev = here;
    }
    if(prev) FREE(prev);
  }
  if(oldmod) FREE(oldmod);
  *model = NULL;
  return;
  
}
