/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1990 Jaijeet S. Roychowdhury
**********/

#include "ngspice.h"
#include "ltradefs.h"
#include "sperror.h"
#include "suffix.h"

int
LTRAdelete(GENmodel *inModel, IFuid name, GENinstance **kill)
{
  LTRAinstance **fast = (LTRAinstance **) kill;
  LTRAmodel *model = (LTRAmodel *) inModel;
  LTRAinstance **prev = NULL;
  LTRAinstance *here;

  for (; model; model = model->LTRAnextModel) {
    prev = &(model->LTRAinstances);
    for (here = *prev; here; here = *prev) {
      if (here->LTRAname == name || (fast && here == *fast)) {
	*prev = here->LTRAnextInstance;
	FREE(here);
	return (OK);
      }
      prev = &(here->LTRAnextInstance);
    }
  }
  return (E_NODEV);
}
