/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
**********/

#include "ngspice.h"
#include "complex.h"
#include "cktdefs.h"
#include "smpdefs.h"
#include "pzdefs.h"
#include "trandefs.h"   /* only to get the 'mode' definitions */
#include "sperror.h"


#define DEBUG	if (0)

/* ARGSUSED */
int
PZan(CKTcircuit *ckt, int reset)
{
    PZAN *pzan = (PZAN *) ckt->CKTcurJob;
    int error;
    int numNames;
    IFuid *nameList;
    void *plot = NULL;

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
	error = (*(SPfrontEnd->OUTpBeginPlot))((void *)ckt,
	    (void*)ckt->CKTcurJob, "Distortion Operating Point",
	    (IFuid)NULL,IF_REAL,numNames,nameList, IF_REAL,&plot);
	if(error) return(error);
	CKTdump(ckt,(double)0,plot);
	(*(SPfrontEnd->OUTendPlot))(plot);
    }

    if (pzan->PZwhich & PZ_DO_POLES) {
	error = CKTpzSetup(ckt, PZ_DO_POLES);
	if (error != OK)
	    return error;
        error = CKTpzFindZeros(ckt, &pzan->PZpoleList, &pzan->PZnPoles);
        if (error != OK)
	    return(error);
    }

    if (pzan->PZwhich & PZ_DO_ZEROS) {
	error = CKTpzSetup(ckt, PZ_DO_ZEROS);
	if (error != OK)
	    return error;
        error = CKTpzFindZeros(ckt, &pzan->PZzeroList, &pzan->PZnZeros);
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
    PZAN *pzan = (PZAN *) ckt->CKTcurJob;
    int	i;

    i = CKTtypelook("transmission line");
    if (i == -1) {
	    i = CKTtypelook("Tranline");
	    if (i == -1)
		i = CKTtypelook("LTRA");
    }
    if (i != -1 && ckt->CKThead[i] != NULL)
	ERROR(E_XMISSIONLINE, "Transmission lines not supported")

    pzan->PZpoleList = (PZtrial *) NULL;
    pzan->PZzeroList = (PZtrial *) NULL;
    pzan->PZnPoles = 0;
    pzan->PZnZeros = 0;

    if (pzan->PZin_pos == pzan->PZin_neg)
	ERROR(E_SHORT, "Input is shorted")

    if (pzan->PZout_pos == pzan->PZout_neg)
	ERROR(E_SHORT, "Output is shorted")

    if (pzan->PZin_pos == pzan->PZout_pos
        && pzan->PZin_neg == pzan->PZout_neg
	&& pzan->PZinput_type == PZ_IN_VOL)
	ERROR(E_INISOUT, "Transfer function is unity")
    else if (pzan->PZin_pos == pzan->PZout_neg
        && pzan->PZin_neg == pzan->PZout_pos
	&& pzan->PZinput_type == PZ_IN_VOL)
	ERROR(E_INISOUT, "Transfer function is -1")

    return(OK);
}

/*
 * PZpost  Post-processing of the pole-zero analysis results
 */

int
PZpost(CKTcircuit *ckt)
{
    PZAN	*pzan = (PZAN *) ckt->CKTcurJob;
    void	*pzPlotPtr = NULL; /* the plot pointer for front end */
    IFcomplex	*out_list;
    IFvalue	outData;    /* output variable (points to out_list) */
    IFuid	*namelist;
    PZtrial	*root;
    char	name[50];
    int		i, j;

    namelist = (IFuid *) MALLOC((pzan->PZnPoles
	+ pzan->PZnZeros)*sizeof(IFuid));
    out_list = (IFcomplex *)MALLOC((pzan->PZnPoles
	+ pzan->PZnZeros)*sizeof(IFcomplex));

    j = 0;
    for (i = 0; i < pzan->PZnPoles; i++) {
	sprintf(name, "pole(%-u)", i+1);
	(*(SPfrontEnd->IFnewUid))(ckt,&(namelist[j++]),(IFuid)NULL,
		name,UID_OTHER,(void **)NULL);
    }
    for (i = 0; i < pzan->PZnZeros; i++) {
	sprintf(name, "zero(%-u)", i+1);
	(*(SPfrontEnd->IFnewUid))(ckt,&(namelist[j++]),(IFuid)NULL,
		name,UID_OTHER,(void **)NULL);
    }

    (*SPfrontEnd->OUTpBeginPlot)(ckt, (void *)pzan, pzan->JOBname,
	    (IFuid)NULL, (int)0, pzan->PZnPoles + pzan->PZnZeros, namelist,
	    IF_COMPLEX, &pzPlotPtr);

    j = 0;
    if (pzan->PZnPoles > 0) {
	for (root = pzan->PZpoleList; root != NULL; root = root->next) {
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

    if (pzan->PZnZeros > 0) {
	for (root = pzan->PZzeroList; root != NULL; root = root->next) {
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

    outData.v.numValue = pzan->PZnPoles + pzan->PZnZeros;
    outData.v.vec.cVec = out_list;

    (*SPfrontEnd->OUTpData)(pzPlotPtr, (IFvalue *) 0, &outData); 
    (*(SPfrontEnd->OUTendPlot))(pzPlotPtr);

    return(OK);
}
