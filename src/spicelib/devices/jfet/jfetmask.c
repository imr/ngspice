/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Mathew Lew and Thomas L. Quarles
Sydney University mods Copyright(c) 1989 Anthony E. Parker, David J. Skellern
        Laboratory for Communication Science Engineering
        Sydney University Department of Electrical Engineering, Australia
**********/
/*
 */

#include "ngspice/ngspice.h"
#include "ngspice/const.h"
#include "ngspice/ifsim.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "jfetdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"


/*ARGSUSED*/
int
JFETmAsk(CKTcircuit *ckt, GENmodel *inModel, int which, IFvalue *value)
{
    JFETmodel *model = (JFETmodel*)inModel;

    NG_IGNORE(ckt);

    switch(which) {
        case JFET_MOD_TNOM:
            value->rValue = model->JFETtnom-CONSTCtoK;
            return(OK);
        case JFET_MOD_VTO:
            value->rValue = model->JFETthreshold;
            return(OK);
        case JFET_MOD_BETA:
            value->rValue = model->JFETbeta;
            return(OK);
        case JFET_MOD_LAMBDA:
            value->rValue = model->JFETlModulation;
            return(OK);
        /* Modification for Sydney University JFET model */
        case JFET_MOD_B:
            value->rValue = model->JFETb;
            return(OK);
        /* end Sydney University mod */
        case JFET_MOD_RD:
            value->rValue = model->JFETdrainResist;
            return(OK);
        case JFET_MOD_RS:
            value->rValue = model->JFETsourceResist;
            return(OK);
        case JFET_MOD_CGS:
            value->rValue = model->JFETcapGS;
            return(OK);
        case JFET_MOD_CGD:
            value->rValue = model->JFETcapGD;
            return(OK);
        case JFET_MOD_PB:
            value->rValue = model->JFETgatePotential;
            return(OK);
        case JFET_MOD_IS:
            value->rValue = model->JFETgateSatCurrent;
            return(OK);
        case JFET_MOD_FC:
            value->rValue = model->JFETdepletionCapCoeff;
            return(OK);
        case JFET_MOD_DRAINCONDUCT:
            value->rValue = model->JFETdrainConduct;
            return(OK);
        case JFET_MOD_SOURCECONDUCT:
            value->rValue = model->JFETsourceConduct;
            return(OK);
        case JFET_MOD_TCV:
            value->rValue = model->JFETtcv;
            return(OK);
        case JFET_MOD_VTOTC:
            value->rValue = model->JFETvtotc;
            return(OK);
        case JFET_MOD_BEX:
            value->rValue = model->JFETbex;
            return(OK);
        case JFET_MOD_BETATCE:
            value->rValue = model->JFETbetatce;
            return(OK);
        case JFET_MOD_XTI:
            value->rValue = model->JFETxti;
            return(OK);
        case JFET_MOD_EG:
            value->rValue = model->JFETeg;
            return(OK);
        case JFET_MOD_TYPE:
            if (model->JFETtype == NJF)
                value->sValue = "njf";
            else
                value->sValue = "pjf";
            return(OK);
        case JFET_MOD_KF:
            value->rValue = model->JFETfNcoef;
            return(OK);
        case JFET_MOD_AF:
            value->rValue = model->JFETfNexp;
            return(OK);
        case JFET_MOD_NLEV:
            value->iValue = model->JFETnlev;
            return(OK);
        case JFET_MOD_GDSNOI:
            value->rValue = model->JFETgdsnoi;
            return(OK);
        default:
            return(E_BADPARM);
    }
    /* NOTREACHED */
}
