/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include <config.h>
#include <cktdefs.h>
#include <sperror.h>


int
CKTfndDev(void *Ckt, int *type, void **fast, IFuid name, void *modfast, IFuid modname)
{
    CKTcircuit *ckt=(CKTcircuit *)Ckt;
    GENinstance *here;
    GENmodel *mods;

    if((GENinstance **)fast != (GENinstance **)NULL && 
       *(GENinstance **)fast != (GENinstance *)NULL) {
        /* already have fast, so nothing much to do just get & set
	   type */
        if (type)
	    *type = (*((GENinstance**)fast))->GENmodPtr->GENmodType;
        return(OK);
    }

    if(modfast) {
        /* have model, just need device */
        mods = (GENmodel*)modfast;
        for (here = mods->GENinstances;
	     here != NULL;
	     here = here->GENnextInstance) {
            if (here->GENname == name) {
                if (fast != NULL)
		    *(GENinstance **)fast = here;

                if (type)
		    *type = mods->GENmodType;

                return OK;
            }
        }
        return E_NODEV;
    }

    if (*type >= 0 && *type < DEVmaxnum) {
        /* have device type, need to find model & device */
        /* look through all models */
        for (mods = (GENmodel *)ckt->CKThead[*type];
	     mods != NULL ; 
	     mods = mods->GENnextModel) {
            /* and all instances */
            if (modname == (char *)NULL || mods->GENmodName == modname) {
                for (here = mods->GENinstances;
		     here != NULL; 
		     here = here->GENnextInstance) {
                    if (here->GENname == name) {
                        if (fast != 0)
			    *(GENinstance **)fast = here;
                        return OK;
                    }
                }
                if(mods->GENmodName == modname) {
                    return E_NODEV;
                }
            }
        }
        return E_NOMOD;
    } else if (*type == -1) {
        /* look through all types (UGH - worst case - take forever) */ 
        for(*type = 0; *type < DEVmaxnum; (*type)++) {
            /* need to find model & device */
            /* look through all models */
            for(mods=(GENmodel *)ckt->CKThead[*type];mods!=NULL;
                    mods = mods->GENnextModel) {
                /* and all instances */
                if(modname == (char *)NULL || mods->GENmodName == modname) {
                    for (here = mods->GENinstances;
			 here != NULL; 
			 here = here->GENnextInstance) {
                        if (here->GENname == name) {
                            if(fast != 0)
				*(GENinstance **)fast = here;
                            return OK;
                        }
                    }
                    if(mods->GENmodName == modname) {
                        return E_NODEV;
                    }
                }
            }
        }
        *type = -1;
        return E_NODEV;
    } else
	return E_BADPARM;
}
