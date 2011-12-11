/**********
Permit to use it as your wish.
Author:	2007 Gong Ding, gdiso@ustc.edu 
University of Science and Technology of China 
**********/


#include "ngspice/ngspice.h"
#include "ndevdefs.h"
#include "ngspice/suffix.h"

void
NDEVdestroy(GENmodel **inModel)
{
    
    NDEVmodel **model = (NDEVmodel **)inModel;
    NDEVinstance *here;
    NDEVinstance *prev = NULL;
    NDEVmodel *mod = *model;
    NDEVmodel *oldmod = NULL;

    for( ; mod ; mod = mod->NDEVnextModel) {
        if(oldmod) FREE(oldmod);
        oldmod = mod;
        prev = NULL;
        for(here = mod->NDEVinstances ; here ; here = here->NDEVnextInstance) {
            if(prev) FREE(prev);
            prev = here;
        }
        if(prev) FREE(prev);
	close(mod->sock);
	fprintf(stdout,"Disconnect to remote NDEV server %s:%d\n",mod->host,mod->port);
    }
    if(oldmod) FREE(oldmod);
    *model = NULL;
}
