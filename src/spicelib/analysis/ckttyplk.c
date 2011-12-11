/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/*  look up the 'type' in the device description struct and return the
 *  appropriate strchr for the device found, or -1 for not found 
 */

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"


int
CKTtypelook(char *type)
{

    int i;
    for(i=0;i<DEVmaxnum;i++) {
        if(DEVices[i] && !strcmp(type, DEVices[i]->DEVpublic.name)) {
            /*found the device - return it */
            return(i);
        }
    }
    return(-1);
}

