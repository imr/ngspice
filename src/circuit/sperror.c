/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* 
 *  provide the error message appropriate for the given error code
 */

#include "ngspice.h"
#include <stdio.h>
#include "fteext.h"
#include "sperror.h"
#include "cktdefs.h"
#include "ifsim.h"
#include "inp.h"

#define MSG(a) \
    msg = a; \
    break;

static char *unknownError = "Unknown error code";
static char *Pause =        "Pause requested";
static char *intern =       "Impossible error - can't occur";
static char *exists =       "Device already exists, existing one being used";
static char *nodev =        "No such device";
static char *noterm =       "No such terminal on this device";
static char *nomod =        "No such model";
static char *badparm =      "No such parameter on this device";
static char *nomem =        "Out of Memory";
static char *nodecon =      "Node already connected; connection replaced";
static char *unsupp =       "operation not supported";
static char *parmval =      "parameter value out of range or the wrong type";
static char *badmatrix =    "Matrix can't be decomposed as is";
static char *singular =     "Matrix is singular";
static char *iterlim =      "Iteration limit reached";
static char *order =        "Unsupported integration order";
static char *method =       "Unsupported integration method";
static char *timestep =     "Timestep too small";
static char *xmission =     "transmission lines not supported by pole-zero";
static char *toobig =       "magnitude overflow";
static char *isshort =      "input or output shorted";
static char *inisout =      "transfer function is 1";
static char *nodisto =	    "Distortion analysis not present";
static char *nonoise =	    "Noise analysis not present";
static char *noacinput =    "AC input not found";
static char *noanal =	    "No such analysis type";
static char *nochange =     "Unsupported action; no change made";
static char *notfound =     "Not found";
static char *nof2src =      "No F2 source for IM disto analysis";
#ifdef PARALLEL_ARCH
static char *multierr =	    "Multiple errors detected by parallel machine";
#endif /* PARALLEL_ARCH */

char *
SPerror(int type)
{
    char *val;
    char *msg;

    switch(type) {
        default:
            MSG(unknownError)
        case E_PAUSE:
            MSG(Pause)
        case OK:
            return(NULL);
        case E_INTERN:
            MSG(intern)
        case E_EXISTS:
            MSG(exists)
        case E_NODEV:
            MSG(nodev)
        case E_NOMOD:
            MSG(nomod)
        case E_NOTERM:
            MSG(noterm)
        case E_BADPARM:
            MSG(badparm)
        case E_NOMEM:
            MSG(nomem)
        case E_NODECON:
            MSG(nodecon)
        case E_UNSUPP:
            MSG(unsupp)
        case E_PARMVAL:
            MSG(parmval)
        case E_BADMATRIX:
            MSG(badmatrix)
        case E_SINGULAR:
            MSG(singular)
        case E_ITERLIM:
            MSG(iterlim)
        case E_ORDER:
            MSG(order)
        case E_METHOD:
            MSG(method)
        case E_TIMESTEP:
            MSG(timestep)
        case E_XMISSIONLINE:
            MSG(xmission)
        case E_MAGEXCEEDED:
            MSG(toobig)
        case E_SHORT:
            MSG(isshort)
        case E_INISOUT:
            MSG(inisout)
	case E_NODISTO:
	    MSG(nodisto)
	case E_NONOISE:
	    MSG(nonoise)
	case E_NOANAL:
	    MSG(noanal)
	case E_NOCHANGE:
	    MSG(nochange)
	case E_NOTFOUND:
	    MSG(notfound)
	case E_NOACINPUT:
	    MSG(noacinput)
	case E_NOF2SRC:
	    MSG(nof2src)
#ifdef PARALLEL_ARCH
	case E_MULTIERR:
	    MSG(multierr)
#endif /* PARALLEL_ARCH */
    }

    val = MALLOC(strlen(msg) + 1);
    if (val) {
	(void) strcpy(val, msg);
    }
    return(val);
}
