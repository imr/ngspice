/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Model Author: 1995 Colin McAndrew Motorola
Spice3 Implementation: 2003 Dietmar Warning DAnalyse GmbH
**********/

/*
 * This routine sets instance parameters for
 * VBICs in the circuit.
 */

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "vbicdefs.h"
#include "sperror.h"
#include "suffix.h"


/* ARGSUSED */
int
VBICparam(int param, IFvalue *value, GENinstance *instPtr, IFvalue *select)
{
    VBICinstance *here = (VBICinstance*)instPtr;

    switch(param) {
        case VBIC_AREA:
            here->VBICarea = value->rValue;
            here->VBICareaGiven = TRUE;
            break;
        case VBIC_OFF:
            here->VBICoff = value->iValue;
            break;
        case VBIC_IC_VBE:
            here->VBICicVBE = value->rValue;
            here->VBICicVBEGiven = TRUE;
            break;
        case VBIC_IC_VCE:
            here->VBICicVCE = value->rValue;
            here->VBICicVCEGiven = TRUE;
            break;
        case VBIC_TEMP:
            here->VBICtemp = value->rValue+CONSTCtoK;
            here->VBICtempGiven = TRUE;
            break;
        case VBIC_DTEMP:
            here->VBICdtemp = value->rValue;
            here->VBICdtempGiven = TRUE;
            break;
        case VBIC_M:
            here->VBICm = value->rValue;
            here->VBICmGiven = TRUE;
            break;
        case VBIC_IC :
            switch(value->v.numValue) {
                case 2:
                    here->VBICicVCE = *(value->v.vec.rVec+1);
                    here->VBICicVCEGiven = TRUE;
                case 1:
                    here->VBICicVBE = *(value->v.vec.rVec);
                    here->VBICicVBEGiven = TRUE;
                    break;
                default:
                    return(E_BADPARM);
            }
            break;
        default:
            return(E_BADPARM);
    }
    return(OK);
}
