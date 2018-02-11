/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "urcdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/sperror.h"
#include "ngspice/suffix.h"

/* ARGSUSED */
int
URCsetup(SMPmatrix *matrix, GENmodel *inModel, CKTcircuit *ckt, int *state)
        /* create the resistors/capacitors used to model the URC
         */
{
    URCmodel *model = (URCmodel *)inModel;
    URCinstance *here;
    int rtype;
    int ctype;
    int dtype;
    CKTnode * lowl;
    CKTnode * lowr;
    CKTnode * hil;
    CKTnode * hir;
    char *nameelt;
    char *namehi;
    CKTnode *nodehi;
    CKTnode *nodelo;
    char *namelo;
    double r;
    double c;
    IFvalue ptemp;
    double p;
    double r0;
    double c0;
    double i0;
    double r1;
    double c1;
    double i1;
    double rd;
    double wnorm;
    double prop;
    int i;
    GENinstance  *fast;
    GENmodel *modfast; /* capacitor or diode model */
    GENmodel *rmodfast; /* resistor model */
    int error;
    IFuid dioUid;
    IFuid resUid;
    IFuid capUid;
    IFuid eltUid;

    NG_IGNORE(state);
    NG_IGNORE(matrix);

    rtype = CKTtypelook("Resistor");
    ctype = CKTtypelook("Capacitor");
    dtype = CKTtypelook("Diode");
    /*  loop through all the URC models */
    for( ; model != NULL; model = URCnextModel(model)) {
	if(!model->URCkGiven)
	    model->URCk = 1.5;
	if(!model->URCfmaxGiven)
	    model->URCfmax = 1e9;
	if(!model->URCrPerLGiven)
	    model->URCrPerL = 1000;
	if(!model->URCcPerLGiven)
	    model->URCcPerL = 1e-12;

/* may need to put in limits:  k>=1.1, freq <=1e9, rperl >=.1 */

        /* loop through all the instances of the model */
        for (here = URCinstances(model); here != NULL ;
                here=URCnextInstance(here))
	{
            p = model->URCk;
            r0 = here->URClength * model->URCrPerL;
            c0 = here->URClength * model->URCcPerL;
            i0 = here->URClength * model->URCisPerL;
            if(!here->URClumpsGiven) {
                wnorm = model->URCfmax * r0 * c0 * 2.0 * M_PI;
                here->URClumps=(int)MAX(3.0,log(wnorm*(((p-1)/p)*((p-1)/p)))/log(p));
                if(wnorm <35) here->URClumps=3;
                /* may want to limit lumps to <= 100 or something like that */
            }
            r1 = (r0*(p-1))/((2*(pow(p,(double)here->URClumps)))-2);
            c1 = (c0 * (p-1))/((pow(p,(double)(here->URClumps-1)))*(p+1)-2);
            i1 = (i0 * (p-1))/((pow(p,(double)(here->URClumps-1)))*(p+1)-2);
            rd = here->URClength * here->URClumps * model->URCrsPerL;
            /* may want to check that c1 > 0 */
            prop=1;

            if(model->URCisPerLGiven) {
                error = SPfrontEnd->IFnewUid (ckt, &dioUid, here->URCname,
                        "diodemod", UID_MODEL, NULL);
                if(error) return(error);
                modfast = NULL;
                error = CKTmodCrt(ckt,dtype,&modfast,
                        dioUid);
                if(error) return(error);
                ptemp.rValue = c1;
                error= CKTpModName("cjo",&ptemp,ckt,dtype,dioUid,&modfast);
                if(error) return(error);
                ptemp.rValue = rd;
                error = CKTpModName("rs",&ptemp,ckt,dtype,dioUid,&modfast);
                if(error) return(error);
                ptemp.rValue = i1;
                error = CKTpModName("is",&ptemp,ckt,dtype,dioUid,&modfast);
                if(error) return(error);
            } else {
                error = SPfrontEnd->IFnewUid (ckt, &capUid,
                        here->URCname, "capmod", UID_MODEL, NULL);
                if(error) return(error);
                modfast = NULL;
                error = CKTmodCrt(ckt,ctype,&modfast,
                        capUid);
                if(error) return(error);
            }

            error = SPfrontEnd->IFnewUid (ckt, &resUid, here->URCname,
                    "resmod", UID_MODEL, NULL);
            if(error) return(error);
            rmodfast = NULL;
            error = CKTmodCrt(ckt,rtype,&rmodfast,resUid);
            if(error) return(error);
            lowl = CKTnum2nod(ckt,here->URCposNode);
            hir = CKTnum2nod(ckt,here->URCnegNode);
            for(i=1;i<=here->URClumps;i++) {
                namehi = tprintf("hi%d", i);
                error = CKTmkVolt(ckt, &nodehi, here->URCname, namehi);
                if(error) return(error);
                hil = nodehi;
                if(i==here->URClumps) {
                    lowr = hil;
                } else {
                    namelo = tprintf("lo%d", i);
                    error = CKTmkVolt(ckt, &nodelo, here->URCname,
                            namelo);
                    if(error) return(error);
                    lowr = nodelo;
                }
                r = prop*r1;
                c = prop*c1;

                nameelt = tprintf("rlo%d", i);
                error = SPfrontEnd->IFnewUid (ckt, &eltUid, here->URCname,
                        nameelt, UID_INSTANCE, NULL);
                if(error) return(error);
                error = CKTcrtElt(ckt,rmodfast,
                        &fast,eltUid);
                if(error) return(error);
                error = CKTbindNode(ckt,fast,1,lowl);
                if(error) return(error);
                error = CKTbindNode(ckt,fast,2,lowr);
                if(error) return(error);
                ptemp.rValue = r;
                error = CKTpName("resistance",&ptemp,ckt,rtype,nameelt,&fast);
                if(error) return(error);

                nameelt = tprintf("rhi%d", i);
                error = SPfrontEnd->IFnewUid (ckt, &eltUid, here->URCname,
                        nameelt, UID_INSTANCE, NULL);
                if(error) return(error);
                error = CKTcrtElt(ckt,rmodfast,
                        &fast,eltUid);
                if(error) return(error);
                error = CKTbindNode(ckt,fast,1,hil);
                if(error) return(error);
                error = CKTbindNode(ckt,fast,2,hir);
                if(error) return(error);
                ptemp.rValue = r;
                error = CKTpName("resistance",&ptemp,ckt,rtype,nameelt,&fast);
                if(error) return(error);

                if(model->URCisPerLGiven) {
                    /* use diode */
                    nameelt = tprintf("dlo%d", i);
                    error = SPfrontEnd->IFnewUid (ckt, &eltUid,
                            here->URCname,nameelt,UID_INSTANCE, 
                            NULL);
                    if(error) return(error);
                    error = CKTcrtElt(ckt,modfast,
                            &fast, eltUid);
                    if(error) return(error);
                    error = CKTbindNode(ckt,fast,1,lowr);
                    if(error) return(error);
                    error = CKTbindNode(ckt,fast,2,
                            CKTnum2nod(ckt, here->URCgndNode));
                    if(error) return(error);
                    ptemp.rValue = prop;
                    error = CKTpName("area",&ptemp,ckt,dtype,nameelt,&fast);
                    if(error) return(error);
                } else {
                    /* use simple capacitor */
                    nameelt = tprintf("clo%d", i);
                    error = SPfrontEnd->IFnewUid (ckt, &eltUid, here->URCname
                            ,nameelt, UID_INSTANCE, NULL);
                    if(error) return(error);
                    error = CKTcrtElt(ckt,modfast,
                            &fast, eltUid);
                    if(error) return(error);
                    error = CKTbindNode(ckt,fast,1,lowr);
                    if(error) return(error);
                    error = CKTbindNode(ckt,fast,2,
                            CKTnum2nod(ckt, here->URCgndNode));
                    if(error) return(error);
                    ptemp.rValue = c;
                    error = CKTpName("capacitance",&ptemp,ckt,ctype,nameelt,
                            &fast);
                    if(error) return(error);
                }

                if(i!=here->URClumps){
                    if(model->URCisPerLGiven) {
                        /* use diode */
                        nameelt = tprintf("dhi%d", i);
                        error = SPfrontEnd->IFnewUid (ckt, &eltUid,
                                here->URCname,nameelt,UID_INSTANCE,
                                NULL);
                        if(error) return(error);
                        error = CKTcrtElt(ckt,modfast,
                                &fast,eltUid);
                        if(error) return(error);
                        error = CKTbindNode(ckt,fast,1,hil);
                        if(error) return(error);
                        error = CKTbindNode(ckt,fast,2,
                                CKTnum2nod(ckt, here->URCgndNode));
                        if(error) return(error);
                        ptemp.rValue = prop;
                        error=CKTpName("area",&ptemp,ckt,dtype,nameelt,&fast);
                        if(error) return(error);
                    } else {
                        /* use simple capacitor */
                        nameelt = tprintf("chi%d", i);
                        error = SPfrontEnd->IFnewUid (ckt, &eltUid,
                                here->URCname,nameelt,UID_INSTANCE,
                                NULL);
                        if(error) return(error);
                        error = CKTcrtElt(ckt,modfast,
                                &fast,eltUid);
                        if(error) return(error);
                        error = CKTbindNode(ckt,fast,1,hil);
                        if(error) return(error);
                        error = CKTbindNode(ckt,fast,2,
                                CKTnum2nod(ckt, here->URCgndNode));
                        if(error) return(error);
                        ptemp.rValue = c;
                        error =CKTpName("capacitance",&ptemp,ckt,ctype,nameelt,
                                &fast);
                        if(error) return(error);
                    }
                }
                prop *= p;
                lowl = lowr;
                hir = hil;
            }
        }
    }
    return(OK);
}

int
URCunsetup(GENmodel *inModel, CKTcircuit *ckt)
{
    IFuid	varUid;
    int	error;
    URCinstance *here;
    URCmodel *model = (URCmodel *) inModel;
    GENinstance *in;
    GENmodel *modfast;

    /* Delete models, devices, and intermediate nodes; */

    for ( ; model; model = URCnextModel(model)) {
	for (here = URCinstances(model); here;
	     here = URCnextInstance(here))
	{
	    if(model->URCisPerLGiven) {
		/* Diodes */
		error = SPfrontEnd->IFnewUid (ckt, &varUid,
						  here->URCname,
						  "diodemod",
						  UID_MODEL,
						  NULL);
	    } else {
		/* Capacitors */
		error = SPfrontEnd->IFnewUid (ckt, &varUid,
						  here->URCname,
						  "capmod",
						  UID_MODEL,
						  NULL);
	    }
	    if (error && error != E_EXISTS)
		return error;

	    modfast = CKTfndMod(ckt, varUid);
	    if (!modfast)
		return E_NOMOD;

	    for (in = modfast->GENinstances; in; in = in->GENnextInstance)
		CKTdltNNum(ckt, GENnode(in)[0]);

	    CKTdltMod(ckt, modfast);	/* Does the elements too */

	    /* Resistors */
	    error = SPfrontEnd->IFnewUid (ckt, &varUid, here->URCname,
					      "resmod", UID_MODEL, NULL);
	    if (error && error != E_EXISTS)
		return error;

	    modfast = CKTfndMod(ckt, varUid);
	    if (!modfast)
		return E_NOMOD;
	    CKTdltMod(ckt, modfast);
	}
    }
    return OK;
}
