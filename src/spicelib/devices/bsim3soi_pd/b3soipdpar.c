/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1998 Samuel Fung, Dennis Sinitsky and Stephen Tang
File: b3soipdpar.c          98/5/01
Modified by Pin Su	99/2/15
Modified by Pin Su      01/2/15
Modified by Paolo Nenzi 2002
**********/

/*
 * Revision 2.2.3  02/3/5  Pin Su 
 * BSIMPD2.2.3 release
 */

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "b3soipddef.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

int
B3SOIPDparam(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    B3SOIPDinstance *here = (B3SOIPDinstance*)inst;

    NG_IGNORE(select);

    /* FALLTHROUGH added to suppress GCC warning due to
     * -Wimplicit-fallthrough flag */
    switch(param) {
        case B3SOIPD_W:
            here->B3SOIPDw = value->rValue;
            here->B3SOIPDwGiven = TRUE;
            break;
        case B3SOIPD_L:
            here->B3SOIPDl = value->rValue;
            here->B3SOIPDlGiven = TRUE;
            break;
	case B3SOIPD_M:
            here->B3SOIPDm = value->rValue;
            here->B3SOIPDmGiven = TRUE;
            break;    
        case B3SOIPD_AS:
            here->B3SOIPDsourceArea = value->rValue;
            here->B3SOIPDsourceAreaGiven = TRUE;
            break;
        case B3SOIPD_AD:
            here->B3SOIPDdrainArea = value->rValue;
            here->B3SOIPDdrainAreaGiven = TRUE;
            break;
        case B3SOIPD_PS:
            here->B3SOIPDsourcePerimeter = value->rValue;
            here->B3SOIPDsourcePerimeterGiven = TRUE;
            break;
        case B3SOIPD_PD:
            here->B3SOIPDdrainPerimeter = value->rValue;
            here->B3SOIPDdrainPerimeterGiven = TRUE;
            break;
        case B3SOIPD_NRS:
            here->B3SOIPDsourceSquares = value->rValue;
            here->B3SOIPDsourceSquaresGiven = TRUE;
            break;
        case B3SOIPD_NRD:
            here->B3SOIPDdrainSquares = value->rValue;
            here->B3SOIPDdrainSquaresGiven = TRUE;
            break;
        case B3SOIPD_OFF:
            here->B3SOIPDoff = value->iValue;
            here->B3SOIPDoffGiven = TRUE;
            break;
        case B3SOIPD_IC_VBS:
            here->B3SOIPDicVBS = value->rValue;
            here->B3SOIPDicVBSGiven = TRUE;
            break;
        case B3SOIPD_IC_VDS:
            here->B3SOIPDicVDS = value->rValue;
            here->B3SOIPDicVDSGiven = TRUE;
            break;
        case B3SOIPD_IC_VGS:
            here->B3SOIPDicVGS = value->rValue;
            here->B3SOIPDicVGSGiven = TRUE;
            break;
        case B3SOIPD_IC_VES:
            here->B3SOIPDicVES = value->rValue;
            here->B3SOIPDicVESGiven = TRUE;
            break;
        case B3SOIPD_IC_VPS:
            here->B3SOIPDicVPS = value->rValue;
            here->B3SOIPDicVPSGiven = TRUE;
            break;
        case B3SOIPD_BJTOFF:
            here->B3SOIPDbjtoff = value->iValue;
            here->B3SOIPDbjtoffGiven= TRUE;
            break;
        case B3SOIPD_DEBUG:
            here->B3SOIPDdebugMod = value->iValue;
            here->B3SOIPDdebugModGiven= TRUE;
            break;
        case B3SOIPD_RTH0:
            here->B3SOIPDrth0= value->rValue;
            here->B3SOIPDrth0Given = TRUE;
            break;
        case B3SOIPD_CTH0:
            here->B3SOIPDcth0= value->rValue;
            here->B3SOIPDcth0Given = TRUE;
            break;
        case B3SOIPD_NRB:
            here->B3SOIPDbodySquares = value->rValue;
            here->B3SOIPDbodySquaresGiven = TRUE;
            break;
        case B3SOIPD_FRBODY:
            here->B3SOIPDfrbody = value->rValue;
            here->B3SOIPDfrbodyGiven = TRUE;
            break;


/* v2.0 release */
        case B3SOIPD_NBC:
            here->B3SOIPDnbc = value->rValue;
            here->B3SOIPDnbcGiven = TRUE;
            break;
        case B3SOIPD_NSEG:
            here->B3SOIPDnseg = value->rValue;
            here->B3SOIPDnsegGiven = TRUE;
            break;
        case B3SOIPD_PDBCP:
            here->B3SOIPDpdbcp = value->rValue;
            here->B3SOIPDpdbcpGiven = TRUE;
            break;
        case B3SOIPD_PSBCP:
            here->B3SOIPDpsbcp = value->rValue;
            here->B3SOIPDpsbcpGiven = TRUE;
            break;
        case B3SOIPD_AGBCP:
            here->B3SOIPDagbcp = value->rValue;
            here->B3SOIPDagbcpGiven = TRUE;
            break;
        case B3SOIPD_AEBCP:
            here->B3SOIPDaebcp = value->rValue;
            here->B3SOIPDaebcpGiven = TRUE;
            break;
        case B3SOIPD_VBSUSR:
            here->B3SOIPDvbsusr = value->rValue;
            here->B3SOIPDvbsusrGiven = TRUE;
            break;
        case B3SOIPD_TNODEOUT:
            here->B3SOIPDtnodeout = value->iValue;
            here->B3SOIPDtnodeoutGiven = TRUE;
            break;


        case B3SOIPD_IC:
            switch (value->v.numValue) {
                case 5:
                    here->B3SOIPDicVPS = *(value->v.vec.rVec+4);
                    here->B3SOIPDicVPSGiven = TRUE;
                    /* FALLTHROUGH */
                case 4:
                    here->B3SOIPDicVES = *(value->v.vec.rVec+3);
                    here->B3SOIPDicVESGiven = TRUE;
                    /* FALLTHROUGH */
                case 3:
                    here->B3SOIPDicVBS = *(value->v.vec.rVec+2);
                    here->B3SOIPDicVBSGiven = TRUE;
                    /* FALLTHROUGH */
                case 2:
                    here->B3SOIPDicVGS = *(value->v.vec.rVec+1);
                    here->B3SOIPDicVGSGiven = TRUE;
                    /* FALLTHROUGH */
                case 1:
                    here->B3SOIPDicVDS = *(value->v.vec.rVec);
                    here->B3SOIPDicVDSGiven = TRUE;
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



