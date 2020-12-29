/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
Modified: 2001 Paolo Nenzi (Cider Integration)
**********/

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/ifsim.h"
#include "ngspice/inpdefs.h"
#include "ngspice/inpmacs.h"
#include "ngspice/fteext.h"
#include "inpxx.h"


static int
model_numnodes(int type)
{
    if (type == INPtypelook("B4SOI") ||     /* 7 ; B4SOInames */
        type == INPtypelook("B3SOIPD") ||   /* 7 ; B3SOIPDnames */
        type == INPtypelook("B3SOIFD") ||   /* 7 ; B3SOIFDnames */
        type == INPtypelook("B3SOIDD"))     /* 7 ; B3SOIDDnames */
    {
        return 7;
    }

    if (type == INPtypelook("HiSIMHV1") ||  /* 6 ; HSMHVnames */
        type == INPtypelook("HiSIMHV2") ||  /* 6 ; HSMHV2names */
        type == INPtypelook("SOI3"))        /* 6 ; SOI3names */
    {
        return 6;
    }

#ifdef ADMS
    if (type == INPtypelook("BSIMBULK") ||  /* bsimbulk.va */
        type == INPtypelook("BSIMCMG"))     /* bsimcmg.va */
    {
        return 5;
    }
#endif

    if (type == INPtypelook("VDMOS"))       /* 3 ; VDMOSnames */
    {
        return 5;
    }

    return 4;
}


void
INP2M(CKTcircuit *ckt, INPtables *tab, struct card *current)
{
    /* Mname <node> <node> <node> <node> <model> [L=<val>]
     *       [W=<val>] [AD=<val>] [AS=<val>] [PD=<val>]
     *       [PS=<val>] [NRD=<val>] [NRS=<val>] [OFF]
     *       [IC=<val>,<val>,<val>]
     */

    int type;                  /* the type the model says it is */
    char *line;                /* the part of the current line left to parse */
    char *name;                /* the resistor's name */
    const int max_i = 7;
    CKTnode *node[7];
    int error;                 /* error code temporary */
    int numnodes;              /* flag indicating 4 or 5 (or 6 or 7) nodes */
    GENinstance *fast;         /* pointer to the actual instance */
    int waslead;               /* flag to indicate that funny unlabeled number was found */
    double leadval;            /* actual value of unlabeled number */
    INPmodel *thismodel;       /* pointer to model description for user's model */
    GENmodel *mdfast;          /* pointer to the actual model */
    int i;

#ifdef TRACE
    printf("INP2M: Parsing '%s'\n", current->line);
#endif

    line = current->line;

    INPgetNetTok(&line, &name, 1);
    INPinsert(&name, tab);

    for (i = 0; ; i++) {
        char *token;
        INPgetNetTok(&line, &token, 1);

        if (i >= 3) {
            txfree(INPgetMod(ckt, token, &thismodel, tab));

            /* check if using model binning -- pass in line since need 'l' and 'w' */
            if (!thismodel)
                txfree(INPgetModBin(ckt, token, &thismodel, tab, line));

            if (thismodel) {
                INPinsert(&token, tab);
                break;
            }
        }
        if (i >= max_i) {
            LITERR ("could not find a valid modelname");
            return;
        }
        INPtermInsert(ckt, &token, tab, &node[i]);
    }

    /* We have at least 4 nodes, except for VDMOS */
    if (i == 3 && thismodel->INPmodType != INPtypelook("VDMOS")) {
        LITERR("not enough nodes");
        return;
    }

    int model_numnodes_ = model_numnodes(thismodel->INPmodType);
    if (i > model_numnodes_) {
        LITERR ("too many nodes connected to instance");
        return;
    }

    numnodes = i;

    if (thismodel->INPmodType != INPtypelook("Mos1") &&
        thismodel->INPmodType != INPtypelook("Mos2") &&
        thismodel->INPmodType != INPtypelook("Mos3") &&
        thismodel->INPmodType != INPtypelook("Mos5") &&
        thismodel->INPmodType != INPtypelook("Mos6") &&
        thismodel->INPmodType != INPtypelook("Mos8") &&
        thismodel->INPmodType != INPtypelook("Mos9") &&
        thismodel->INPmodType != INPtypelook("BSIM1") &&
        thismodel->INPmodType != INPtypelook("BSIM2") &&
        thismodel->INPmodType != INPtypelook("BSIM3") &&
        thismodel->INPmodType != INPtypelook("BSIM3v32") &&
        thismodel->INPmodType != INPtypelook("B4SOI") &&
        thismodel->INPmodType != INPtypelook("B3SOIPD") &&
        thismodel->INPmodType != INPtypelook("B3SOIFD") &&
        thismodel->INPmodType != INPtypelook("B3SOIDD") &&
        thismodel->INPmodType != INPtypelook("BSIM4") &&
        thismodel->INPmodType != INPtypelook("BSIM4v5") &&
        thismodel->INPmodType != INPtypelook("BSIM4v6") &&
        thismodel->INPmodType != INPtypelook("BSIM4v7") &&
        thismodel->INPmodType != INPtypelook("BSIM3v0") &&
        thismodel->INPmodType != INPtypelook("BSIM3v1") &&
        thismodel->INPmodType != INPtypelook("SOI3") &&
#ifdef CIDER
        thismodel->INPmodType != INPtypelook("NUMOS") &&
#endif
#ifdef ADMS
        thismodel->INPmodType != INPtypelook("ekv") &&
        thismodel->INPmodType != INPtypelook("psp102") &&
        thismodel->INPmodType != INPtypelook("psp103") &&
        thismodel->INPmodType != INPtypelook("bsimbulk") &&
        thismodel->INPmodType != INPtypelook("bsimcmg") &&
#endif
        thismodel->INPmodType != INPtypelook("HiSIM2") &&
        thismodel->INPmodType != INPtypelook("HiSIMHV1") &&
        thismodel->INPmodType != INPtypelook("HiSIMHV2") &&
        thismodel->INPmodType != INPtypelook("VDMOS"))
    {
        LITERR ("incorrect model type");
        return;
    }
    type = thismodel->INPmodType;
    mdfast = thismodel->INPmodfast;

    IFC (newInstance, (ckt, mdfast, &fast, name));

    for (i = 0; i < model_numnodes_; i++)
        if (i < numnodes)
            IFC (bindNode, (ckt, fast, i + 1, node[i]));
        else
            GENnode(fast)[i] = -1;

    PARSECALL ((&line, ckt, type, fast, &leadval, &waslead, tab));
    if (waslead)
        LITERR (" error:  no unlabeled parameter permitted on mosfet\n");
}
