/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/
/*
 */

#include "ngspice.h"
#include "const.h"
#include "ifsim.h"
#include "jfetdefs.h"
#include "sperror.h"
#include "suffix.h"


int
JFETmParam(int param, IFvalue *value, GENmodel *inModels)
{
    JFETmodel *model = (JFETmodel*)inModels;
    switch(param) {
        case JFET_MOD_TNOM:
            model->JFETtnomGiven = TRUE;
            model->JFETtnom = value->rValue+CONSTCtoK;
            break;
        case JFET_MOD_VTO:
            model->JFETthresholdGiven = TRUE;
            model->JFETthreshold = value->rValue;
            break;
        case JFET_MOD_BETA:
            model->JFETbetaGiven = TRUE;
            model->JFETbeta = value->rValue;
            break;
        case JFET_MOD_LAMBDA:
            model->JFETlModulationGiven = TRUE;
            model->JFETlModulation = value->rValue;
            break;
        case JFET_MOD_RD:
            model->JFETdrainResistGiven = TRUE;
            model->JFETdrainResist = value->rValue;
            break;
        case JFET_MOD_RS:
            model->JFETsourceResistGiven = TRUE;
            model->JFETsourceResist = value->rValue;
            break;
        case JFET_MOD_CGS:
            model->JFETcapGSGiven = TRUE;
            model->JFETcapGS = value->rValue;
            break;
        case JFET_MOD_CGD:
            model->JFETcapGDGiven = TRUE;
            model->JFETcapGD = value->rValue;
            break;
        case JFET_MOD_PB:
            model->JFETgatePotentialGiven = TRUE;
            model->JFETgatePotential = value->rValue;
            break;
        case JFET_MOD_IS:
            model->JFETgateSatCurrentGiven = TRUE;
            model->JFETgateSatCurrent = value->rValue;
            break;
        case JFET_MOD_FC:
            model->JFETdepletionCapCoeffGiven = TRUE;
            model->JFETdepletionCapCoeff = value->rValue;
            break;
        case JFET_MOD_NJF:
            if(value->iValue) {
                model->JFETtype = NJF;
            }
            break;
        case JFET_MOD_PJF:
            if(value->iValue) {
                model->JFETtype = PJF;
            }
            break;
	case JFET_MOD_KF:
	    model->JFETfNcoefGiven = TRUE;
	    model->JFETfNcoef = value->rValue;
	    break;
	case JFET_MOD_AF:
	    model->JFETfNexpGiven = TRUE;
	    model->JFETfNexp = value->rValue;
	    break;
	/* Modification for Sydney University JFET model */
	case JFET_MOD_B:
	    model->JFETbGiven = TRUE;
	    model->JFETb = value->rValue;
	    return(OK);
	/* end Sydney University mod */
        default:
            return(E_BADPARM);
    }
    return(OK);
}
