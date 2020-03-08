/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
Modified by Paolo Nenzi 2002
File: b3soifdpar.c          98/5/01
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMFD2.1 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "b3soifddef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
B3SOIFDparam(int param, IFvalue *value, GENinstance *inst, 
             IFvalue *select)
{
    B3SOIFDinstance *here = (B3SOIFDinstance*)inst;

    NG_IGNORE(select);

    switch (param) {
        case B3SOIFD_W:
            here->B3SOIFDw = value->rValue;
            here->B3SOIFDwGiven = TRUE;
            break;
        case B3SOIFD_L:
            here->B3SOIFDl = value->rValue;
            here->B3SOIFDlGiven = TRUE;
            break;
	case B3SOIFD_M:
            here->B3SOIFDm = value->rValue;
            here->B3SOIFDmGiven = TRUE;
            break;    
        case B3SOIFD_AS:
            here->B3SOIFDsourceArea = value->rValue;
            here->B3SOIFDsourceAreaGiven = TRUE;
            break;
        case B3SOIFD_AD:
            here->B3SOIFDdrainArea = value->rValue;
            here->B3SOIFDdrainAreaGiven = TRUE;
            break;
        case B3SOIFD_PS:
            here->B3SOIFDsourcePerimeter = value->rValue;
            here->B3SOIFDsourcePerimeterGiven = TRUE;
            break;
        case B3SOIFD_PD:
            here->B3SOIFDdrainPerimeter = value->rValue;
            here->B3SOIFDdrainPerimeterGiven = TRUE;
            break;
        case B3SOIFD_NRS:
            here->B3SOIFDsourceSquares = value->rValue;
            here->B3SOIFDsourceSquaresGiven = TRUE;
            break;
        case B3SOIFD_NRD:
            here->B3SOIFDdrainSquares = value->rValue;
            here->B3SOIFDdrainSquaresGiven = TRUE;
            break;
        case B3SOIFD_OFF:
            here->B3SOIFDoff = value->iValue;
            break;
        case B3SOIFD_IC_VBS:
            here->B3SOIFDicVBS = value->rValue;
            here->B3SOIFDicVBSGiven = TRUE;
            break;
        case B3SOIFD_IC_VDS:
            here->B3SOIFDicVDS = value->rValue;
            here->B3SOIFDicVDSGiven = TRUE;
            break;
        case B3SOIFD_IC_VGS:
            here->B3SOIFDicVGS = value->rValue;
            here->B3SOIFDicVGSGiven = TRUE;
            break;
        case B3SOIFD_IC_VES:
            here->B3SOIFDicVES = value->rValue;
            here->B3SOIFDicVESGiven = TRUE;
            break;
        case B3SOIFD_IC_VPS:
            here->B3SOIFDicVPS = value->rValue;
            here->B3SOIFDicVPSGiven = TRUE;
            break;
        case B3SOIFD_BJTOFF:
            here->B3SOIFDbjtoff = value->iValue;
            here->B3SOIFDbjtoffGiven= TRUE;
            break;
        case B3SOIFD_DEBUG:
            here->B3SOIFDdebugMod = value->iValue;
            here->B3SOIFDdebugModGiven= TRUE;
            break;
        case B3SOIFD_RTH0:
            here->B3SOIFDrth0= value->rValue;
            here->B3SOIFDrth0Given = TRUE;
            break;
        case B3SOIFD_CTH0:
            here->B3SOIFDcth0= value->rValue;
            here->B3SOIFDcth0Given = TRUE;
            break;
        case B3SOIFD_NRB:
            here->B3SOIFDbodySquares = value->rValue;
            here->B3SOIFDbodySquaresGiven = TRUE;
            break;
        case B3SOIFD_IC:
            /* FALLTHROUGH added to suppress GCC warning due to
             * -Wimplicit-fallthrough flag */
            switch (value->v.numValue) {
                case 5:
                    here->B3SOIFDicVPS = *(value->v.vec.rVec+4);
                    here->B3SOIFDicVPSGiven = TRUE;
                    /* FALLTHROUGH */
                case 4:
                    here->B3SOIFDicVES = *(value->v.vec.rVec+3);
                    here->B3SOIFDicVESGiven = TRUE;
                    /* FALLTHROUGH */
                case 3:
                    here->B3SOIFDicVBS = *(value->v.vec.rVec+2);
                    here->B3SOIFDicVBSGiven = TRUE;
                    /* FALLTHROUGH */
                case 2:
                    here->B3SOIFDicVGS = *(value->v.vec.rVec+1);
                    here->B3SOIFDicVGSGiven = TRUE;
                    /* FALLTHROUGH */
                case 1:
                    here->B3SOIFDicVDS = *(value->v.vec.rVec);
                    here->B3SOIFDicVDSGiven = TRUE;
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



