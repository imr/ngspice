/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "cktdefs.h"
#include "ifsim.h"
#include "sperror.h"



int
CKTfndMod(void *ckt, int *type, void **modfast, IFuid modname)
{
    GENmodel *mods;

    if(modfast != NULL && *(GENmodel **)modfast != NULL) {
        /* already have  modfast, so nothing to do */
        if(type) *type = (*(GENmodel **)modfast)->GENmodType;
        return(OK);
    } 
    if(*type >=0 && *type < DEVmaxnum) {
        /* have device type, need to find model */
        /* look through all models */
        for(mods=((CKTcircuit *)ckt)->CKThead[*type]; mods != NULL ; 
                mods = mods->GENnextModel) {
            if(mods->GENmodName == modname) {
                *modfast = (char *)mods;
                return(OK);
            }
        }
        return(E_NOMOD);
    } else if(*type == -1) {
        /* look through all types (UGH - worst case - take forever) */ 
        for(*type = 0;*type <DEVmaxnum;(*type)++) {
            /* need to find model & device */
            /* look through all models */
            for(mods=((CKTcircuit *)ckt)->CKThead[*type];mods!=NULL;
                    mods = mods->GENnextModel) {
                if(mods->GENmodName == modname) {
                    *modfast = (char *)mods;
                    return(OK);
                }
            }
        }
        *type = -1;
        return(E_NOMOD);
    } else return(E_BADPARM);
}
