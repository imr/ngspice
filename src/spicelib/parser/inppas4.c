/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/inpdefs.h"
#include "ngspice/inpmacs.h"
#include "ngspice/fteext.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/cktdefs.h"

#include "inppas4.h"
#include "inpxx.h"

#ifdef XSPICE
/* gtri - add - wbk - 11/9/90 - include function prototypes */
#include "ngspice/mifproto.h"
/* gtri - end - wbk - 11/9/90 */
#endif


/* uncomment to trace in this file */
/*#define TRACE*/

/* pass 4 - If option cshunt is given,
add a capacitor to each voltage node */
void INPpas4(CKTcircuit *ckt, INPtables *tab)
{
    CKTnode* node;
    int mytype = -1;
    IFuid uid;           /* uid for default cap model */
    int error;           /* error code temporary */
    GENinstance* fast;   /* pointer to the actual instance */
    IFvalue ptemp;       /* a value structure to package capacitance into */
    int nadded;          /* capacitors added */
    double csval = 0.;        /* cshunt capacitors value */

    /* get the cshunt value */
    if (cp_getvar("cshunt_value", CP_REAL, &csval, 0)) {

        if ((mytype = INPtypelook("Capacitor")) < 0) {
            fprintf(stderr, "Device type Capacitor not supported by this binary\n");
            return;
        }

        nadded = 0;

        if (!tab->defCmod) {    /* create default C model */
            IFnewUid(ckt, &uid, NULL, "C", UID_MODEL, NULL);
            error = (*(ft_sim->newModel))(ckt, mytype, &(tab->defCmod), uid);
        }

        /* scan through all nodes, add a new C device for each voltage node */
        for (node = ckt->CKTnodes; node; node = node->next) {
            if (node->type == NODE_VOLTAGE && (node->number > 0)) {
                int nn = node->number;
                char* devname = tprintf("capac%dshunt", nn);

                (*(ft_sim->newInstance))(ckt, tab->defCmod, &fast, devname);

                /* the top node, second node is gnd automatically */
                (*(ft_sim->bindNode))(ckt, fast, 1, node);

                /* value of the capacitance */
                ptemp.rValue = csval;
                error = INPpName("capacitance", &ptemp, ckt, mytype, fast);

                /* add device numbers for statistics */
                ckt->CKTstat->STATdevNum[mytype].instNum++;
                ckt->CKTstat->STATtotalDev++;

                nadded++;
            }
        }
        printf("Option cshunt: %d capacitors added with %g F each\n", nadded, csval);
    }
    /* get the cshunt value for optran */
    if (cp_getvar("optran_cshunt_val", CP_REAL, &csval, 0)) {

        if ((mytype = INPtypelook("Capacitor")) < 0) {
            fprintf(stderr, "Device type Capacitor not supported by this binary\n");
            return;
        }

        nadded = 0;

        if (!tab->defCmod) {    /* create default C model */
            IFnewUid(ckt, &uid, NULL, "C", UID_MODEL, NULL);
            error = (*(ft_sim->newModel))(ckt, mytype, &(tab->defCmod), uid);
        }

        /* scan through all nodes, add a new C device for each voltage node */
        for (node = ckt->CKTnodes; node; node = node->next) {
            if (node->type == NODE_VOLTAGE && (node->number > 0)) {
                int nn = node->number;
                char* devname = tprintf("coptran%dshunt", nn);

                (*(ft_sim->newInstance))(ckt, tab->defCmod, &fast, devname);

                /* the top node, second node is gnd automatically */
                (*(ft_sim->bindNode))(ckt, fast, 1, node);

                /* value of the capacitance */
                ptemp.rValue = csval;
                error = INPpName("capacitance", &ptemp, ckt, mytype, fast);

                /* add device numbers for statistics */
                ckt->CKTstat->STATdevNum[mytype].instNum++;
                ckt->CKTstat->STATtotalDev++;

                nadded++;
            }
        }
        if(ft_ngdebug)
            printf("optran C-shunt: %d capacitors added with %g F each\n", nadded, csval);
    }
}

/* Remove the optran C-shunt instances from the list of instances in capacitor model "C".
   Models are at the bottom of the instance linked list.*/
void remoptrancshunt(void)
{
    int delnum = 0;
    GENinstance *inst, *interm;
    /* get the model pointer for C */
    GENmodel* inModel = ft_curckt->ci_symtab->defCmod;
    /* get the pointer to the hash table of instances */
    NGHASHPTR delhash = ft_curckt->ci_ckt->DEVnameHash;
    inst = inModel->GENinstances;
    while (prefix("coptran", inst->GENname)) {
        interm = inst;
        inst = inst->GENnextInstance;
        nghash_delete(delhash, interm->GENname);
        tfree(interm->GENname);
        tfree(interm);
        delnum++;
    }
    /* restore the new instance list */
    inModel->GENinstances = inst;
    if(ft_ngdebug)
        printf("optran C-shunt: %d capacitors deleted from C instance list\n", delnum);
}

#if (0)
void remoptrancshunt_old(void)
{
    GENinstance *fast, *inst1, *inst2, *interm, *prev;
    /* get the model for C */
    GENmodel *inModel = ft_curckt->ci_symtab->defCmod;
    inst1 = prev = inModel->GENinstances;
    inst2 = fast = inModel->GENinstances->GENnextInstance;
    while (fast) {
        if (prefix("coptran", fast->GENname)) {
            /* delete the instance */
            interm = fast;
            fast = fast->GENnextInstance;
            nghash_delete(ft_curckt->ci_ckt->DEVnameHash, interm->GENname);
            tfree(interm->GENname);
            tfree(interm);
            prev->GENnextInstance = fast;
        }
        else {
            prev = fast;
            fast = fast->GENnextInstance;
        }
    }
    if (prefix("coptran", inst1->GENname)) {
        inModel->GENinstances = inst1->GENnextInstance;
        tfree(inst1->GENname);
        tfree(inst1);
    }
}
#endif
