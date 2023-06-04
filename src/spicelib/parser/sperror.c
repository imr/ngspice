/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

/* 
 *  provide the error message appropriate for the given error code
 */

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/fteext.h"
#include "ngspice/sperror.h"
#include "ngspice/cktdefs.h"
#include "ngspice/ifsim.h"
#include "inpxx.h"

const char *SPerror(int type)
{
    char *msg;

    switch (type) {
    case E_PAUSE:
	msg = "pause requested";
	break;
    case E_INTERN:
	msg = "impossible error - can't occur";
	break;
    case E_EXISTS:
	msg = "device already exists, existing one being used";
	break;
    case E_EXISTS_BAD:
	msg = "device already exists, bail out";
	break;
	case E_NODEV:
	msg = "no such device";
	break;
    case E_NOMOD:
	msg = "no such model";
	break;
    case E_NOTERM:
	msg = "no such terminal on this device";
	break;
    case E_BADPARM:
	msg = "no such parameter on this device";
	break;
    case E_NOMEM:
	msg = "out of memory";
	break;
    case E_NODECON:
	msg = "node already connected; connection replaced";
	break;
    case E_UNSUPP:
	msg = "operation not supported";
	break;
    case E_PARMVAL:
	msg = "parameter value out of range or the wrong type";
	break;
    case E_BADMATRIX:
	msg = "matrix can't be decomposed as is";
	break;
    case E_SINGULAR:
	msg = "matrix is singular";
	break;
    case E_ITERLIM:
	msg = "iteration limit reached";
	break;
    case E_ORDER:
	msg = "unsupported integration order";
	break;
    case E_METHOD:
	msg = "unsupported integration method";
	break;
    case E_TIMESTEP:
	msg = "timestep too small";
	break;
    case E_XMISSIONLINE:
	msg = "transmission lines not supported by pole-zero";
	break;
    case E_MAGEXCEEDED:
	msg = "magnitude overflow";
	break;
    case E_SHORT:
	msg = "input or output shorted";
	break;
    case E_INISOUT:
	msg = "transfer function is 1";
	break;
    case E_NODISTO:
	msg = "distortion analysis not present";
	break;
    case E_NONOISE:
	msg = "noise analysis not present";
	break;
    case E_NOANAL:
	msg = "no such analysis type";
	break;
    case E_NOCHANGE:
	msg = "unsupported action; no change made";
	break;
    case E_NOTFOUND:
	msg = "not found";
	break;
    case E_NOACINPUT:
	msg = "ac input not found";
	break;
    case E_NOF2SRC:
	msg = "no F2 source for IM disto analysis";
	break;
    case OK:
	return (NULL);
    default:
	msg = "Unknown error code";
    }

    return msg;
}
