/**********
Permit to use it as your wish.
Author:	2007 Gong Ding, gdiso@ustc.edu 
University of Science and Technology of China 
**********/

#include "ngspice/ngspice.h"
#include "ndevdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
NDEVdelete(
  GENmodel *inModel,
  IFuid name,
  GENinstance **kill )

{
    NDEVmodel *model = (NDEVmodel *)inModel;
    NDEVinstance **fast = (NDEVinstance **)kill;
    NDEVinstance **prev = NULL;
    NDEVinstance *here;

    for( ; model ; model = model->NDEVnextModel) {
        prev = &(model->NDEVinstances);
        for(here = *prev; here ; here = *prev) {
            if(here->NDEVname == name || (fast && here==*fast) ) {
                *prev= here->NDEVnextInstance;
                FREE(here);
                return(OK);
            }
            prev = &(here->NDEVnextInstance);
        }
    }
    return(E_NODEV);
 
  return (E_NODEV);
}
