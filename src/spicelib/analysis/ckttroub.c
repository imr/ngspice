/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

#include "ngspice.h"
#include "trandefs.h"
#include "cktdefs.h"
#include "devdefs.h"
#include "vsrc/vsrcdefs.h"
#include "isrc/isrcdefs.h"
#include "jobdefs.h"

#include "analysis.h"

extern SPICEdev **DEVices;


extern SPICEanalysis *analInfo[];

char *
CKTtrouble(void *cktp, char *optmsg)
{
    CKTcircuit	*ckt = (CKTcircuit *) cktp;
    char	msg_buf[513];
    char	*emsg;
    TRCV	*cv;
    int		vcode, icode;
    char	*msg_p;
    SPICEanalysis *an;
    int		i;

    if (!ckt || !ckt->CKTcurJob)
	return NULL;

    an = analInfo[ckt->CKTcurJob->JOBtype];

    if (optmsg && *optmsg) {
       sprintf(msg_buf, "%s:  %s; ", an->public.name, optmsg);
    } else {
       sprintf(msg_buf, "%s:  ", an->public.name);
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

	for (i = 0; i <= cv->TRCVnestLevel; i++) {
	    msg_p += strlen(msg_p);
	    if(cv->TRCVvType[i]==vcode) { /* voltage source */
		sprintf(msg_p, " %s = %g: ", cv->TRCVvName[i],
		    ((VSRCinstance*)(cv->TRCVvElt[i]))->VSRCdcValue);
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

    emsg = MALLOC(strlen(msg_buf)+1);
    strcpy(emsg,msg_buf);

    return emsg;
}
