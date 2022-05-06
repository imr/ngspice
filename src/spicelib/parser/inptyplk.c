/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/*  look up the 'type' in the device description struct and return the
 *  appropriate strchr for the device found, or -1 for not found 
 */

#include "ngspice/ngspice.h"
#include "ngspice/inpdefs.h"
#include "ngspice/cpdefs.h"
#include "ngspice/fteext.h"
#include "ngspice/ifsim.h"
#include "inpxx.h"

int INPtypelook(char *type)
{

    int i;

#ifdef TRACE
    /* SDB debug statement */
        printf("In INPtypelook, examining model type = %s . . .\n", type); 
#endif

    for (i = 0; i < ft_sim->numDevices; i++) {

#ifdef TRACE
       /* SDB debug statement */
       if ((ft_sim->devices[i]) == NULL)
             printf("In INPtypelook, checking model type = %s against existing model = %s, . . .\n", type, "NULL");		
       else
             printf("In INPtypelook, checking model type = %s against existing model = %s, . . .\n", type, ft_sim->devices[i]->name);
#endif

	if (ft_sim->devices[i] && strcasecmp(type, ft_sim->devices[i]->name) == 0) {
	    /* found the device - return it */

#ifdef TRACE
	/* SDB debug statement */
	printf("In INPtypelook, found the device -- returning it!!!\n"); 
#endif

	    return i;
	}
    }

#ifdef TRACE
    /* SDB debug statement */
    printf("In INPtypelook, device not found!\n"); 
#endif

    return -1;
}
