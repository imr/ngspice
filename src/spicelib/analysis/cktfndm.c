/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/sperror.h"



GENmodel *
CKTfndMod(CKTcircuit *ckt, int *type, GENmodel **modfast, IFuid modname)
{
    GENmodel *mods;

    if(modfast != NULL && *modfast != NULL) {
        /* already have  modfast, so nothing to do */
        if(type) *type = (*modfast)->GENmodType;
        return *modfast;
    } 
    if(*type >=0 && *type < DEVmaxnum) {
        /* have device type, need to find model */
        /* look through all models */
        for(mods=ckt->CKThead[*type]; mods != NULL ; 
                mods = mods->GENnextModel) {
            if(mods->GENmodName == modname) {
                *modfast = mods;
                return *modfast;
            }
        }
        return NULL;
    } else if(*type == -1) {
        /* look through all types (UGH - worst case - take forever) */ 
        for(*type = 0;*type <DEVmaxnum;(*type)++) {
            /* need to find model & device */
            /* look through all models */
            for(mods=ckt->CKThead[*type];mods!=NULL;
                    mods = mods->GENnextModel) {
                if(mods->GENmodName == modname) {
                    *modfast = mods;
                    return *modfast;
                }
            }
        }
        *type = -1;
        return NULL;
    } else return NULL;
}
