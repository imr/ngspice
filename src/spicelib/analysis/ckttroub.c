/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/trandefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "vsrc/vsrcdefs.h"
#include "isrc/isrcdefs.h"
#include "res/resdefs.h"
#include "ngspice/jobdefs.h"

#include "analysis.h"


extern SPICEanalysis *analInfo[];

char *
CKTtrouble(CKTcircuit *ckt, char *optmsg)
{
    char	msg_buf[513];
    char	*emsg;
    TRCV	*cv;
    int		vcode, icode, rcode;
    char	*msg_p;
    SPICEanalysis *an;
    int		i;

    if (!ckt || !ckt->CKTcurJob)
	return NULL;

    an = analInfo[ckt->CKTcurJob->JOBtype];

    if (optmsg && *optmsg) {
       sprintf(msg_buf, "%s:  %s; ", an->if_analysis.name, optmsg);
    } else {
       sprintf(msg_buf, "%s:  ", an->if_analysis.name);
    }

    msg_p = msg_buf + strlen(msg_buf);

    switch (an->domain) {
    case TIMEDOMAIN:
	if (ckt->CKTtime == 0.0)
	    sprintf(msg_p, "initial timepoint: ");
	else
	    sprintf(msg_p, "time = %g, timestep = %g: ", ckt->CKTtime,
		ckt->CKTdelta);
	break;

    case FREQUENCYDOMAIN:
	sprintf(msg_p, "frequency = %g: ", ckt->CKTomega / (2.0 * M_PI));
	break;

    case SWEEPDOMAIN:
	cv = (TRCV*) ckt->CKTcurJob;
	vcode = CKTtypelook("Vsource");
	icode = CKTtypelook("Isource");
	rcode = CKTtypelook("Resistor");

	for (i = 0; i <= cv->TRCVnestLevel; i++) {
	    msg_p += strlen(msg_p);
		if (cv->TRCVvType[i] == vcode) { /* voltage source */
			sprintf(msg_p, " %s = %g: ", cv->TRCVvName[i],
				((VSRCinstance*)(cv->TRCVvElt[i]))->VSRCdcValue);
		}
		else if (cv->TRCVvType[i] == TEMP_CODE) { /* temp sweep, if optran fails) */
			sprintf(msg_p, " %s = %g: ", cv->TRCVvName[i], ckt->CKTtemp - CONSTCtoK);
		}
		else if (cv->TRCVvType[i] == rcode) {
			sprintf(msg_p, " %s = %g: ", cv->TRCVvName[i],
				((RESinstance*)(cv->TRCVvElt[i]))->RESresist);
	    } else {
		sprintf(msg_p, " %s = %g: ", cv->TRCVvName[i],
		    ((ISRCinstance*)(cv->TRCVvElt[i]))->ISRCdcValue);
	    }
	}
	break;

    case NODOMAIN:
    default:
	break;
    }

    msg_p += strlen(msg_p);

    if (ckt->CKTtroubleNode) {
	sprintf(msg_p, "trouble with node \"%s\"\n",
		CKTnodName(ckt, ckt->CKTtroubleNode));
    } else if (ckt->CKTtroubleElt) {
	/* "-" for dop */
	sprintf(msg_p, "trouble with %s-instance %s\n",
	    ckt->CKTtroubleElt->GENmodPtr->GENmodName,
	    ckt->CKTtroubleElt->GENname);
    } else {
	sprintf(msg_p, "cause unrecorded.\n");
    }

    emsg = TMALLOC(char, strlen(msg_buf) + 1);
    strcpy(emsg,msg_buf);

    return emsg;
}
