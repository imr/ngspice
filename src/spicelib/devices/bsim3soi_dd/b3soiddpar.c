/**********
Copyright 1999 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soiddpar.c          98/5/01
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.1  99/9/27 Pin Su 
 * BSIMDD2.1 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "b3soidddef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
B3SOIDDparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    B3SOIDDinstance *here = (B3SOIDDinstance*)inst;

    NG_IGNORE(select);

    /* FALLTHROUGH added to suppress GCC warning due to
     * -Wimplicit-fallthrough flag */
    switch(param) {
        case B3SOIDD_W:
            here->B3SOIDDw = value->rValue;
            here->B3SOIDDwGiven = TRUE;
            break;
        case B3SOIDD_L:
            here->B3SOIDDl = value->rValue;
            here->B3SOIDDlGiven = TRUE;
            break;
	case B3SOIDD_M:
            here->B3SOIDDm = value->rValue;
            here->B3SOIDDmGiven = TRUE;
            break;
        case B3SOIDD_AS:
            here->B3SOIDDsourceArea = value->rValue;
            here->B3SOIDDsourceAreaGiven = TRUE;
            break;
        case B3SOIDD_AD:
            here->B3SOIDDdrainArea = value->rValue;
            here->B3SOIDDdrainAreaGiven = TRUE;
            break;
        case B3SOIDD_PS:
            here->B3SOIDDsourcePerimeter = value->rValue;
            here->B3SOIDDsourcePerimeterGiven = TRUE;
            break;
        case B3SOIDD_PD:
            here->B3SOIDDdrainPerimeter = value->rValue;
            here->B3SOIDDdrainPerimeterGiven = TRUE;
            break;
        case B3SOIDD_NRS:
            here->B3SOIDDsourceSquares = value->rValue;
            here->B3SOIDDsourceSquaresGiven = TRUE;
            break;
        case B3SOIDD_NRD:
            here->B3SOIDDdrainSquares = value->rValue;
            here->B3SOIDDdrainSquaresGiven = TRUE;
            break;
        case B3SOIDD_OFF:
            here->B3SOIDDoff = value->iValue;
            break;
        case B3SOIDD_IC_VBS:
            here->B3SOIDDicVBS = value->rValue;
            here->B3SOIDDicVBSGiven = TRUE;
            break;
        case B3SOIDD_IC_VDS:
            here->B3SOIDDicVDS = value->rValue;
            here->B3SOIDDicVDSGiven = TRUE;
            break;
        case B3SOIDD_IC_VGS:
            here->B3SOIDDicVGS = value->rValue;
            here->B3SOIDDicVGSGiven = TRUE;
            break;
        case B3SOIDD_IC_VES:
            here->B3SOIDDicVES = value->rValue;
            here->B3SOIDDicVESGiven = TRUE;
            break;
        case B3SOIDD_IC_VPS:
            here->B3SOIDDicVPS = value->rValue;
            here->B3SOIDDicVPSGiven = TRUE;
            break;
        case B3SOIDD_BJTOFF:
            here->B3SOIDDbjtoff = value->iValue;
            here->B3SOIDDbjtoffGiven= TRUE;
            break;
        case B3SOIDD_DEBUG:
            here->B3SOIDDdebugMod = value->iValue;
            here->B3SOIDDdebugModGiven= TRUE;
            break;
        case B3SOIDD_RTH0:
            here->B3SOIDDrth0= value->rValue;
            here->B3SOIDDrth0Given = TRUE;
            break;
        case B3SOIDD_CTH0:
            here->B3SOIDDcth0= value->rValue;
            here->B3SOIDDcth0Given = TRUE;
            break;
        case B3SOIDD_NRB:
            here->B3SOIDDbodySquares = value->rValue;
            here->B3SOIDDbodySquaresGiven = TRUE;
            break;
        case B3SOIDD_IC:
            switch (value->v.numValue) {
                case 5:
                    here->B3SOIDDicVPS = *(value->v.vec.rVec+4);
                    here->B3SOIDDicVPSGiven = TRUE;
                    /* FALLTHROUGH */
                case 4:
                    here->B3SOIDDicVES = *(value->v.vec.rVec+3);
                    here->B3SOIDDicVESGiven = TRUE;
                    /* FALLTHROUGH */
                case 3:
                    here->B3SOIDDicVBS = *(value->v.vec.rVec+2);
                    here->B3SOIDDicVBSGiven = TRUE;
                    /* FALLTHROUGH */
                case 2:
                    here->B3SOIDDicVGS = *(value->v.vec.rVec+1);
                    here->B3SOIDDicVGSGiven = TRUE;
                    /* FALLTHROUGH */
                case 1:
                    here->B3SOIDDicVDS = *(value->v.vec.rVec);
                    here->B3SOIDDicVDSGiven = TRUE;
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



