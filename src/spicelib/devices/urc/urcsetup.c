/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1987 Thomas L. Quarles
**********/

#include "ngspice.h"
#include "ifsim.h"
#include "urcdefs.h"
#include "cktdefs.h"
#include "gendefs.h"
#include "sperror.h"
#include "suffix.h"

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

    rtype = CKTtypelook("Resistor");
    ctype = CKTtypelook("Capacitor");
    dtype = CKTtypelook("Diode");
    /*  loop through all the URC models */
    for( ; model != NULL; model = model->URCnextModel ) {
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
        for (here = model->URCinstances; here != NULL ;
                here=here->URCnextInstance)
	{
            p = model->URCk;
            r0 = here->URClength * model->URCrPerL;
            c0 = here->URClength * model->URCcPerL;
            i0 = here->URClength * model->URCisPerL;
            if(!here->URClumpsGiven) {
                wnorm = model->URCfmax * r0 * c0 * 2.0 * M_PI;
                here->URClumps=MAX(3.0,log(wnorm*(((p-1)/p)*((p-1)/p)))/log(p));
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
                error = (*(SPfrontEnd->IFnewUid))(ckt,&dioUid,here->URCname,
                        "diodemod",UID_MODEL,(void **)NULL);
                if(error) return(error);
                modfast = (GENmodel *)NULL;
                error = CKTmodCrt((void *)ckt,dtype,(void **)&modfast,
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
                error = (*(SPfrontEnd->IFnewUid))((void *)ckt,&capUid,
                        here->URCname, "capmod",UID_MODEL,(void **)NULL);
                if(error) return(error);
                modfast = (GENmodel *)NULL;
                error = CKTmodCrt((void *)ckt,ctype,(void **)&modfast,
                        capUid);
                if(error) return(error);
            }

            error = (*(SPfrontEnd->IFnewUid))(ckt,&resUid,here->URCname,
                    "resmod",UID_MODEL,(void **)NULL);
            if(error) return(error);
            rmodfast = (GENmodel *)NULL;
            error = CKTmodCrt((void *)ckt,rtype,(void **)&rmodfast,resUid);
            if(error) return(error);
            lowl = CKTnum2nod(ckt,here->URCposNode);
            hir = CKTnum2nod(ckt,here->URCnegNode);
            for(i=1;i<=here->URClumps;i++) {
                namehi = (char *)MALLOC(10*sizeof(char));
                (void)sprintf(namehi,"hi%d",i);
                error = CKTmkVolt(ckt,(CKTnode**)&nodehi,here->URCname,namehi);
                if(error) return(error);
                hil = nodehi;
                if(i==here->URClumps) {
                    lowr = hil;
                } else {
                    namelo = (char *)MALLOC(10*sizeof(char));
                    (void)sprintf(namelo,"lo%d",i);
                    error = CKTmkVolt(ckt,(CKTnode**)&nodelo,here->URCname,
                            namelo);
                    if(error) return(error);
                    lowr = nodelo;
                }
                r = prop*r1;
                c = prop*c1;

                nameelt = (char *)MALLOC(10*sizeof(char));
                (void)sprintf(nameelt,"rlo%d",i);
                error = (*(SPfrontEnd->IFnewUid))(ckt,&eltUid,here->URCname,
                        nameelt,UID_INSTANCE, (void **)NULL);
                if(error) return(error);
                error = CKTcrtElt((void *)ckt,(void *)rmodfast,
                        (void **)&fast,eltUid);
                if(error) return(error);
                error = CKTbindNode((void *)ckt,(void *)fast,1,lowl);
                if(error) return(error);
                error = CKTbindNode((void *)ckt,(void *)fast,2,lowr);
                if(error) return(error);
                ptemp.rValue = r;
                error = CKTpName("resistance",&ptemp,ckt,rtype,nameelt,&fast);
                if(error) return(error);
		fast->GENowner = here->URCowner;

                nameelt = (char *)MALLOC(10*sizeof(char));
                (void)sprintf(nameelt,"rhi%d",i);
                error = (*(SPfrontEnd->IFnewUid))(ckt,&eltUid,here->URCname,
                        nameelt,UID_INSTANCE, (void **)NULL);
                if(error) return(error);
                error = CKTcrtElt((void *)ckt,(void *)rmodfast,
                        (void **)&fast,eltUid);
                if(error) return(error);
                error = CKTbindNode((void *)ckt,(void *)fast,1,hil);
                if(error) return(error);
                error = CKTbindNode((void *)ckt,(void *)fast,2,hir);
                if(error) return(error);
                ptemp.rValue = r;
                error = CKTpName("resistance",&ptemp,ckt,rtype,nameelt,&fast);
                if(error) return(error);
		fast->GENowner = here->URCowner;

                if(model->URCisPerLGiven) {
                    /* use diode */
                    nameelt = (char *)MALLOC(10*sizeof(char));
                    (void)sprintf(nameelt,"dlo%d",i);
                    error = (*(SPfrontEnd->IFnewUid))(ckt,&eltUid,
                            here->URCname,nameelt,UID_INSTANCE, 
                            (void **)NULL);
                    if(error) return(error);
                    error = CKTcrtElt((void *)ckt,(void *)modfast,
                            (void **)&fast, eltUid);
                    if(error) return(error);
                    error = CKTbindNode((void *)ckt,(void *)fast,1,lowr);
                    if(error) return(error);
                    error = CKTbindNode((void *)ckt,(void *)fast,2,
                            (void *)CKTnum2nod(ckt, here->URCgndNode));
                    if(error) return(error);
                    ptemp.rValue = prop;
                    error = CKTpName("area",&ptemp,ckt,dtype,nameelt,&fast);
                    if(error) return(error);
		    fast->GENowner = here->URCowner;
                } else {
                    /* use simple capacitor */
                    nameelt = (char *)MALLOC(10*sizeof(char));
                    (void)sprintf(nameelt,"clo%d",i);
                    error = (*(SPfrontEnd->IFnewUid))(ckt,&eltUid,here->URCname
                            ,nameelt,UID_INSTANCE, (void **)NULL);
                    if(error) return(error);
                    error = CKTcrtElt((void *)ckt,(void *)modfast,
                            (void **)&fast, eltUid);
                    if(error) return(error);
                    error = CKTbindNode((void *)ckt,(void *)fast,1,lowr);
                    if(error) return(error);
                    error = CKTbindNode((void *)ckt,(void *)fast,2,
                            (void  *)CKTnum2nod(ckt, here->URCgndNode));
                    if(error) return(error);
                    ptemp.rValue = c;
                    error = CKTpName("capacitance",&ptemp,ckt,ctype,nameelt,
                            &fast);
                    if(error) return(error);
		    fast->GENowner = here->URCowner;
                }

                if(i!=here->URClumps){
                    if(model->URCisPerLGiven) {
                        /* use diode */
                        nameelt = (char *)MALLOC(10*sizeof(char));
                        (void)sprintf(nameelt,"dhi%d",i);
                        error = (*(SPfrontEnd->IFnewUid))(ckt,&eltUid,
                                here->URCname,nameelt,UID_INSTANCE,
                                (void **)NULL);
                        if(error) return(error);
                        error = CKTcrtElt((void *)ckt,(void *)modfast,
                                (void **) &fast,eltUid);
                        if(error) return(error);
                        error = CKTbindNode((void *)ckt,(void *)fast,1,hil);
                        if(error) return(error);
                        error = CKTbindNode((void *)ckt,(void *)fast,2,
                                (void *)CKTnum2nod(ckt, here->URCgndNode));
                        if(error) return(error);
                        ptemp.rValue = prop;
                        error=CKTpName("area",&ptemp,ckt,dtype,nameelt,&fast);
                        if(error) return(error);
			fast->GENowner = here->URCowner;
                    } else {
                        /* use simple capacitor */
                        nameelt = (char *)MALLOC(10*sizeof(char));
                        (void)sprintf(nameelt,"chi%d",i);
                        error = (*(SPfrontEnd->IFnewUid))(ckt,&eltUid,
                                here->URCname,nameelt,UID_INSTANCE,
                                (void **)NULL);
                        if(error) return(error);
                        error = CKTcrtElt((void *)ckt,(void *)modfast,
                                (void **)&fast,eltUid);
                        if(error) return(error);
                        error = CKTbindNode((void *)ckt,(void *)fast,1,hil);
                        if(error) return(error);
                        error = CKTbindNode((void *)ckt,(void *)fast,2,
                                (void *)CKTnum2nod(ckt, here->URCgndNode));
                        if(error) return(error);
                        ptemp.rValue = c;
                        error =CKTpName("capacitance",&ptemp,ckt,ctype,nameelt,
                                &fast);
                        if(error) return(error);
			fast->GENowner = here->URCowner;
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
    int type;

    /* Delete models, devices, and intermediate nodes; */

    for ( ; model; model = model->URCnextModel) {
	for (here = model->URCinstances; here;
	     here = here->URCnextInstance)
	{
	    if(model->URCisPerLGiven) {
		/* Diodes */
		error = (*(SPfrontEnd->IFnewUid))(ckt, &varUid,
						  here->URCname,
						  "diodemod",
						  UID_MODEL,
						  (void **)NULL);
	    } else {
		/* Capacitors */
		error = (*(SPfrontEnd->IFnewUid))((void *)ckt, &varUid,
						  here->URCname,
						  "capmod",
						  UID_MODEL,
						  (void **)NULL);
	    }
	    if (error && error != E_EXISTS)
		return error;

	    modfast = NULL;
	    type = -1;
	    error = CKTfndMod(ckt, &type, (void **) &modfast, varUid);
	    if (error)
		return error;

	    for (in = modfast->GENinstances; in; in = in->GENnextInstance)
		CKTdltNNum(ckt, in->GENnode1);

	    CKTdltMod(ckt, modfast);	/* Does the elements too */

	    /* Resistors */
	    error = (*(SPfrontEnd->IFnewUid))(ckt,&varUid,here->URCname,
					      "resmod",UID_MODEL,(void **)NULL);
	    if (error && error != E_EXISTS)
		return error;

	    modfast = NULL;
	    type = -1;
	    error = CKTfndMod(ckt, &type, (void **) &modfast, varUid);
	    if (error)
		return error;

	    CKTdltMod(ckt, modfast);
	}
    }
    return OK;
}
