/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

#include "ngspice/ngspice.h"
#include "ngspice/complex.h"
#include "ngspice/cktdefs.h"
#include "ngspice/smpdefs.h"
#include "ngspice/pzdefs.h"
#include "ngspice/trandefs.h"   /* only to get the 'mode' definitions */
#include "ngspice/sperror.h"


#define DEBUG	if (0)

/* ARGSUSED */
int
PZan(CKTcircuit *ckt, int reset)
{
    PZAN *job = (PZAN *) ckt->CKTcurJob;

    int error;
    int numNames;
    IFuid *nameList;
    runDesc *plot = NULL;

    NG_IGNORE(reset);

    error = PZinit(ckt);
    if (error != OK) return error;

    /* Calculate small signal parameters at the operating point */
    error = CKTop(ckt, (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITJCT,
            (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITFLOAT,
            ckt->CKTdcMaxIter);
    if (error)
	return(error);

    ckt->CKTmode = (ckt->CKTmode & MODEUIC) | MODEDCOP | MODEINITSMSIG;
    error = CKTload(ckt); /* Make sure that all small signal params are
			       * set */
    if (error)
	return(error);

    if (ckt->CKTkeepOpInfo) {
	/* Dump operating point. */
	error = CKTnames(ckt,&numNames,&nameList);
	if(error) return(error);
        error = SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                                           "Distortion Operating Point",
                                           NULL, IF_REAL,
                                           numNames, nameList, IF_REAL,
                                           &plot);
	if(error) return(error);
	CKTdump(ckt, 0.0, plot);
	SPfrontEnd->OUTendPlot (plot);
    }

    if (job->PZwhich & PZ_DO_POLES) {
	error = CKTpzSetup(ckt, PZ_DO_POLES);
	if (error != OK)
	    return error;
        error = CKTpzFindZeros(ckt, &job->PZpoleList, &job->PZnPoles);
        if (error != OK)
	    return(error);
    }

    if (job->PZwhich & PZ_DO_ZEROS) {
	error = CKTpzSetup(ckt, PZ_DO_ZEROS);
	if (error != OK)
	    return error;
        error = CKTpzFindZeros(ckt, &job->PZzeroList, &job->PZnZeros);
        if (error != OK)
	    return(error);
    }

    return PZpost(ckt);
}

/*
 * Perform error checking
 */

int
PZinit(CKTcircuit *ckt)
{
    PZAN *job = (PZAN *) ckt->CKTcurJob;
    int	i;

    i = CKTtypelook("transmission line");
    if (i == -1) {
	    i = CKTtypelook("Tranline");
	    if (i == -1)
		i = CKTtypelook("LTRA");
    }
    if (i != -1 && ckt->CKThead[i] != NULL)
	MERROR(E_XMISSIONLINE, "Transmission lines not supported");

    job->PZpoleList = NULL;
    job->PZzeroList = NULL;
    job->PZnPoles = 0;
    job->PZnZeros = 0;

    if (job->PZin_pos == job->PZin_neg)
	MERROR(E_SHORT, "Input is shorted");

    if (job->PZout_pos == job->PZout_neg)
	MERROR(E_SHORT, "Output is shorted");

    if (job->PZin_pos == job->PZout_pos
        && job->PZin_neg == job->PZout_neg
	&& job->PZinput_type == PZ_IN_VOL)
	MERROR(E_INISOUT, "Transfer function is unity");
    else if (job->PZin_pos == job->PZout_neg
        && job->PZin_neg == job->PZout_pos
	&& job->PZinput_type == PZ_IN_VOL)
	MERROR(E_INISOUT, "Transfer function is -1");

    return(OK);
}

/*
 * PZpost  Post-processing of the pole-zero analysis results
 */

int
PZpost(CKTcircuit *ckt)
{
    PZAN	*job = (PZAN *) ckt->CKTcurJob;
    runDesc	*pzPlotPtr = NULL; /* the plot pointer for front end */
    IFcomplex	*out_list;
    IFvalue	outData;    /* output variable (points to out_list) */
    IFuid	*namelist;
    PZtrial	*root;
    char	name[50];
    int		i, j;

    namelist = TMALLOC(IFuid, job->PZnPoles + job->PZnZeros);
    out_list = TMALLOC(IFcomplex, job->PZnPoles + job->PZnZeros);

    j = 0;
    for (i = 0; i < job->PZnPoles; i++) {
	sprintf(name, "pole(%-u)", i+1);
	SPfrontEnd->IFnewUid (ckt, &(namelist[j++]), NULL, name, UID_OTHER, NULL);
    }
    for (i = 0; i < job->PZnZeros; i++) {
	sprintf(name, "zero(%-u)", i+1);
	SPfrontEnd->IFnewUid (ckt, &(namelist[j++]), NULL, name, UID_OTHER, NULL);
    }

    SPfrontEnd->OUTpBeginPlot (ckt, ckt->CKTcurJob,
                               ckt->CKTcurJob->JOBname,
                               NULL, 0,
                               job->PZnPoles + job->PZnZeros, namelist, IF_COMPLEX,
                               &pzPlotPtr);

    j = 0;
    if (job->PZnPoles > 0) {
	for (root = job->PZpoleList; root != NULL; root = root->next) {
	    for (i = 0; i < root->multiplicity; i++) {
		out_list[j].real = root->s.real;
		out_list[j].imag = root->s.imag;
		j += 1;
		if (root->s.imag != 0.0) {
		    out_list[j].real = root->s.real;
		    out_list[j].imag = -root->s.imag;
		    j += 1;
		}
	    }
	    DEBUG printf("LIST pole: (%g,%g) x %d\n",
		root->s.real, root->s.imag, root->multiplicity);
	}
    }

    if (job->PZnZeros > 0) {
	for (root = job->PZzeroList; root != NULL; root = root->next) {
	    for (i = 0; i < root->multiplicity; i++) {
		out_list[j].real = root->s.real;
		out_list[j].imag = root->s.imag;
		j += 1;
		if (root->s.imag != 0.0) {
		    out_list[j].real = root->s.real;
		    out_list[j].imag = -root->s.imag;
		    j += 1;
		}
	    }
	    DEBUG printf("LIST zero: (%g,%g) x %d\n",
		root->s.real, root->s.imag, root->multiplicity);
	}
    }

    outData.v.numValue = job->PZnPoles + job->PZnZeros;
    outData.v.vec.cVec = out_list;

    SPfrontEnd->OUTpData (pzPlotPtr, NULL, &outData);
    SPfrontEnd->OUTendPlot (pzPlotPtr);

    return(OK);
}
