/**********
STAG version 2.7
Copyright 2000 owned by the United Kingdom Secretary of State for Defence
acting through the Defence Evaluation and Research Agency.
Developed by :     Jim Benson,
                   Department of Electronics and Computer Science,
                   University of Southampton,
                   United Kingdom.
With help from :   Nele D'Halleweyn, Ketan Mistry, Bill Redman-White, and Craig Easson.

Based on STAG version 2.1
Developed by :     Mike Lee,
With help from :   Bernard Tenbroek, Bill Redman-White, Mike Uren, Chris Edwards
                   and John Bunyan.
Acknowledgements : Rupert Howes and Pete Mole.
**********/

/********** 
Modified by Paolo Nenzi 2002
ngspice integration
**********/

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "soi3defs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/* ARGSUSED */
int
SOI3param(int param, IFvalue *value, GENinstance *inst, IFvalue *select)
{
    SOI3instance *here = (SOI3instance *)inst;

    NG_IGNORE(select);

    switch (param) {
        case SOI3_L:
            here->SOI3l = value->rValue;
            here->SOI3lGiven = TRUE;
            break;
        case SOI3_W:
            here->SOI3w = value->rValue;
            here->SOI3wGiven = TRUE;
            break;
	case SOI3_M:
            here->SOI3m = value->rValue;
            here->SOI3mGiven = TRUE;
            break;
	case SOI3_AS:
            here->SOI3as = value->rValue;
            here->SOI3asGiven = TRUE;
            break;
        case SOI3_AD:
            here->SOI3ad = value->rValue;
            here->SOI3adGiven = TRUE;
            break;
        case SOI3_AB:
            here->SOI3ab = value->rValue;
            here->SOI3abGiven = TRUE;
            break;    
        case SOI3_NRD:
            here->SOI3drainSquares = value->rValue;
            here->SOI3drainSquaresGiven = TRUE;
            break;
       case SOI3_NRS:
            here->SOI3sourceSquares = value->rValue;
            here->SOI3sourceSquaresGiven = TRUE;
            break;
        case SOI3_OFF:
            here->SOI3off = (value->iValue != 0);
            break;
        case SOI3_IC_VDS:
            here->SOI3icVDS = value->rValue;
            here->SOI3icVDSGiven = TRUE;
            break;
        case SOI3_IC_VGFS:
            here->SOI3icVGFS = value->rValue;
            here->SOI3icVGFSGiven = TRUE;
            break;
        case SOI3_IC_VGBS:
            here->SOI3icVGBS = value->rValue;
            here->SOI3icVGBSGiven = TRUE;
            break;
        case SOI3_IC_VBS:
            here->SOI3icVBS = value->rValue;
            here->SOI3icVBSGiven = TRUE;
            break;
        case SOI3_TEMP:
            here->SOI3temp = value->rValue+CONSTCtoK;
            here->SOI3tempGiven = TRUE;
            break;
        case SOI3_RT:
            here->SOI3rt = value->rValue;
            here->SOI3rtGiven = TRUE;
            break;
        case SOI3_CT:
            here->SOI3ct = value->rValue;
            here->SOI3ctGiven = TRUE;
            break;
        case SOI3_RT1:
            here->SOI3rt1 = value->rValue;
            here->SOI3rt1Given = TRUE;
            break;
        case SOI3_CT1:
            here->SOI3ct1 = value->rValue;
            here->SOI3ct1Given = TRUE;
            break;
        case SOI3_RT2:
            here->SOI3rt2 = value->rValue;
            here->SOI3rt2Given = TRUE;
            break;
        case SOI3_CT2:
            here->SOI3ct2 = value->rValue;
            here->SOI3ct2Given = TRUE;
            break;
        case SOI3_RT3:
            here->SOI3rt3 = value->rValue;
            here->SOI3rt3Given = TRUE;
            break;
        case SOI3_CT3:
            here->SOI3ct3 = value->rValue;
            here->SOI3ct3Given = TRUE;
            break;
        case SOI3_RT4:
            here->SOI3rt4 = value->rValue;
            here->SOI3rt4Given = TRUE;
            break;
        case SOI3_CT4:
            here->SOI3ct4 = value->rValue;
            here->SOI3ct4Given = TRUE;
            break;
        case SOI3_IC:
            /* FALLTHROUGH added to suppress GCC warning due to
             * -Wimplicit-fallthrough flag */
            switch (value->v.numValue) {
                case 4:
                    here->SOI3icVBS = *(value->v.vec.rVec+3);
                    here->SOI3icVBSGiven = TRUE;
                    /* FALLTHROUGH */
                case 3:
                    here->SOI3icVGBS = *(value->v.vec.rVec+2);
                    here->SOI3icVGBSGiven = TRUE;
                    /* FALLTHROUGH */
                case 2:
                    here->SOI3icVGFS = *(value->v.vec.rVec+1);
                    here->SOI3icVGFSGiven = TRUE;
                    /* FALLTHROUGH */
                case 1:
                    here->SOI3icVDS = *(value->v.vec.rVec);
                    here->SOI3icVDSGiven = TRUE;
                    break;
                default:
                    return(E_BADPARM);
            }
            break;
/*      case SOI3_L_SENS:
            if(value->iValue) {
                here->SOI3senParmNo = 1;
                here->SOI3sens_l = 1;
            }
            break;
        case SOI3_W_SENS:
            if(value->iValue) {
                here->SOI3senParmNo = 1;
                here->SOI3sens_w = 1;
            }
            break;                            */
        default:
            return(E_BADPARM);
    }
    return(OK);
}
