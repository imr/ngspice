/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /*
     * INPaName()
     *
     *  Take a parameter by Name and ask for the specified value 
     * *dev is -1 if type unknown, otherwise, device type
     * **fast is a device, and will be set if possible.
     */

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/cpdefs.h"
#include "ngspice/fteext.h"
#include "ngspice/ifsim.h"
#include "ngspice/iferrmsg.h"
#include "inpxx.h"

int
INPaName ( char *parm,      // the name of the parameter to set,
        IFvalue *val,       // the parameter union containing the value to set,
        CKTcircuit *ckt,    // the circuit this device is a member of,
        int *dev,           // the device type code to the device being parsed,
        char *devnam,       // the name of the device,
        GENinstance **fast, // direct pointer to device being parsed,
        IFsimulator *sim,   // the simulator data structure,
        int *dataType,      // the datatype of the returned value structure,
        IFvalue *selector)  // data sub-selector for questions.
{
    int error;			/* int to store evaluate error return codes in */
    int i;

    /* find the instance - don't know about model, so use null there,
     * otherwise pass on as much info as we have about the device
     * (name, type, direct pointer) - the type and direct pointer
     * WILL be set on return unless error is not OK
     */
    if (*fast == NULL)
        *fast = sim->findInstance (ckt, devnam);
    if (*fast == NULL)
	return (E_NODEV);

    *dev = (*fast)->GENmodPtr->GENmodType;

    /* now find the parameter - hunt through the parameter tables for
     * this device type and look for a name match of an 'ask'able
     * parameter.
     */
    for (i = 0; i < *(sim->devices[*dev]->numInstanceParms); i++) {
	if (strcmp(parm, sim->devices[*dev]->instanceParms[i].keyword) == 0
	    && (sim->devices[*dev]->instanceParms[i].dataType & IF_ASK)) {
	    /* found it, so we ask the question using the device info we got
	     * above and put the results in the IFvalue structure our caller
	     * gave us originally
	     */
	    error = sim->askInstanceQuest (ckt, *fast,
                 sim->devices[*dev]->instanceParms[i].id, val,
						selector);
	    if (dataType)
		*dataType =
		    sim->devices[*dev]->instanceParms[i].dataType;
	    return (error);
	}
    }
    return (E_BADPARM);
}


// mhx: specialized (faster) version that directly returns/uses the index of the parameter we seek
int INPaSpecName (
    char *parm,         // the name of the parameter to set,
    IFvalue *val,       // the parameter union containing the value to set,
    CKTcircuit *ckt,    // the circuit this device is a member of,
    int *dev,           // the device type code to the device being parsed,
    char *devnam,       // the name of the device,
    GENinstance **fast, // direct pointer to device being parsed,
    IFsimulator *sim,   // the simulator data structure,
    int *dataType,      // the datatype of the returned value structure,
    IFvalue *selector,  // data sub-selector for questions.
    int *spindex)       // special index
{
    int error;          // int to store evaluate error return codes in
    int i;

    /*
     * find the instance - don't know about model, so use null there,
     * otherwise pass on as much info as we have about the device
     * (name, type, direct pointer) - the type and direct pointer
     * WILL be set on return unless error is not OK
     */
    if (*fast == NULL)
        *fast = sim->findInstance(ckt, devnam);
    if (*fast == NULL)
        return (E_NODEV);

    *dev = (*fast)->GENmodPtr->GENmodType;

    /* now find the parameter - hunt through the parameter tables for
     * this device type and look for a name match of an 'ask'able
     * parameter.
     */
    if (*spindex != -1) { // mhx: shortcut for repeated question
        error = sim->askInstanceQuest(ckt, *fast, sim->devices[*dev]->instanceParms[*spindex].id, val, selector);
        if (dataType)
            *dataType = sim->devices[*dev]->instanceParms[*spindex].dataType;
        return error;
    }

    for (i = 0; i < *(sim->devices[*dev]->numInstanceParms); i++) {
        if (strcmp(parm, sim->devices[*dev]->instanceParms[i].keyword) == 0
            && (sim->devices[*dev]->instanceParms[i].dataType & IF_ASK)) {
            /*
             * found it, so we ask the question using the device info we got
             * above and put the results in the IFvalue structure our caller gave us originally
             */
            *spindex = i; /* mhx: shortcut for repeated questioning */
            error = sim->askInstanceQuest(ckt, *fast, sim->devices[*dev]->instanceParms[i].id, val, selector);
            if (dataType)
                *dataType = sim->devices[*dev]->instanceParms[i].dataType; // probably double
            return error;
        }
    }

    return E_BADPARM;
}
