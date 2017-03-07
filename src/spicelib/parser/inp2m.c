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


static bool
valid_numnodes(int numnodes, INPmodel *thismodel, card *current)
{
    bool valid;

    switch (numnodes) {
    case 7:
        valid =
            thismodel->INPmodType == INPtypelook("B4SOI") ||
            thismodel->INPmodType == INPtypelook("B3SOIPD") ||
            thismodel->INPmodType == INPtypelook("B3SOIFD") ||
            thismodel->INPmodType == INPtypelook("B3SOIDD");
        if (!valid)
            LITERR ("only level 55-58: B3SOI(PD|FD|DD) and B4SOI can have 7 nodes");
        return valid;

    case 6:
        valid =
            thismodel->INPmodType == INPtypelook("B4SOI") ||
            thismodel->INPmodType == INPtypelook("B3SOIPD") ||
            thismodel->INPmodType == INPtypelook("B3SOIFD") ||
            thismodel->INPmodType == INPtypelook("B3SOIDD") ||
            thismodel->INPmodType == INPtypelook("HiSIMHV1") ||
            thismodel->INPmodType == INPtypelook("HiSIMHV2") ||
            thismodel->INPmodType == INPtypelook("SOI3");
        if (!valid)
            LITERR ("only level 55-58,61,62: B3SOI(PD|FD|DD), B4SOI, STAG (SOI3) and HiSIMHV can have 6 nodes");
        return valid;

    case 5:
        valid =
            thismodel->INPmodType == INPtypelook("B4SOI") ||
            thismodel->INPmodType == INPtypelook("B3SOIPD") ||
            thismodel->INPmodType == INPtypelook("B3SOIFD") ||
            thismodel->INPmodType == INPtypelook("B3SOIDD") ||
            thismodel->INPmodType == INPtypelook("HiSIMHV1") ||
            thismodel->INPmodType == INPtypelook("HiSIMHV2") ||
            thismodel->INPmodType == INPtypelook("SOI3");
        if (!valid)
            LITERR ("only level 55-58,61,62: B3SOI(PD|FD|DD), B4SOI, STAG (SOI3) and HiSIMHV can have 5 nodes");
        return valid;

    default:
        return TRUE;
    }
}


void
INP2M(CKTcircuit *ckt, INPtables *tab, card *current)
{
    /* Mname <node> <node> <node> <node> <model> [L=<val>]
     *       [W=<val>] [AD=<val>] [AS=<val>] [PD=<val>]
     *       [PS=<val>] [NRD=<val>] [NRS=<val>] [OFF]
     *       [IC=<val>,<val>,<val>]
     */

    int type;                  /* the type the model says it is */
    char *line;                /* the part of the current line left to parse */
    char *name;                /* the resistor's name */
    char *nname[8];
    char *save;                /* saj - used to save the posn of the start of
                                  the parameters if the model is a mosfet*/
    CKTnode *node[7];
    int error;                 /* error code temporary */
    int numnodes;              /* flag indicating 4 or 5 (or 6 or 7) nodes */
    GENinstance *fast;         /* pointer to the actual instance */
    int waslead;               /* flag to indicate that funny unlabeled number was found */
    double leadval;            /* actual value of unlabeled number */
    char *model;               /* the name of the model */
    INPmodel *thismodel;       /* pointer to model description for user's model */
    GENmodel *mdfast;          /* pointer to the actual model */
    IFuid uid;                 /* uid for default model */
    char *err_msg;
    int i;

#ifdef TRACE
    printf("INP2M: Parsing '%s'\n", current->line);
#endif

    numnodes = 4;               /* initially specify a 4 terminal device */
    line = current->line;

    INPgetTok(&line, &name, 1);
    INPinsert(&name, tab);

    INPgetNetTok(&line, &nname[0], 1);
    INPtermInsert(ckt, &nname[0], tab, &node[0]);
    INPgetNetTok(&line, &nname[1], 1);
    INPtermInsert(ckt, &nname[1], tab, &node[1]);
    INPgetNetTok(&line, &nname[2], 1);
    INPtermInsert(ckt, &nname[2], tab, &node[2]);
    INPgetNetTok(&line, &nname[3], 1);
    INPtermInsert(ckt, &nname[3], tab, &node[3]);

    node[4] = NULL;
    node[5] = NULL;
    node[6] = NULL;

    for (numnodes = 4; numnodes < 8; numnodes++) {

        INPgetNetTok(&line, &nname[numnodes], 1);

        if (numnodes == 4)
            save = line;                     /* saj - save the posn for later if
                                                the default mosfet model is used */

        err_msg = INPgetMod(ckt, nname[numnodes], &thismodel, tab);
        tfree(err_msg);

        /* check if using model binning -- pass in line since need 'l' and 'w' */
        if (!thismodel && numnodes < 5)
            INPgetModBin(ckt, nname[numnodes], &thismodel, tab, line);

        if (thismodel)
            break;
    }

    /* nothing found, reset and process as if it were a 4 node device */
    if (numnodes >= 8) {
        numnodes = 4;
        line = save;
    }

    if (!valid_numnodes(numnodes, thismodel, current))
        return;

    for (i = 4; i < numnodes; i++)
        INPtermInsert(ckt, &nname[i], tab, &node[i]);
    model = nname[numnodes];

    INPinsert(&model, tab);

#ifdef TRACE
    printf("INP2M: Looking up model\n");
#endif

    err_msg = INPgetMod(ckt, model, &thismodel, tab);
    if (!thismodel) {
        INPgetModBin(ckt, model, &thismodel, tab, save);
        if (!thismodel) {
            current->error = err_msg;
            err_msg = NULL;
        }
    }
    tfree(err_msg);

    if (thismodel) {
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
#endif
            thismodel->INPmodType != INPtypelook("HiSIM2") &&
            thismodel->INPmodType != INPtypelook("HiSIMHV1") &&
            thismodel->INPmodType != INPtypelook("HiSIMHV2"))
        {
            LITERR ("incorrect model type");
            return;
        }
        type = thismodel->INPmodType;
        mdfast = thismodel->INPmodfast;
    } else {
        type = INPtypelook("Mos1");
        if (type < 0) {
            LITERR ("Device type MOS1 not supported by this binary\n");
            return;
        }
        if (!tab->defMmod) {
            /* create default M model */
            IFnewUid(ckt, &uid, NULL, "M", UID_MODEL, NULL);
            IFC (newModel, (ckt, type, &(tab->defMmod), uid));
        }
        mdfast = tab->defMmod;
    }

    IFC (newInstance, (ckt, mdfast, &fast, name));
    IFC (bindNode, (ckt, fast, 1, node[0]));
    IFC (bindNode, (ckt, fast, 2, node[1]));
    IFC (bindNode, (ckt, fast, 3, node[2]));
    IFC (bindNode, (ckt, fast, 4, node[3]));

    /* use type - not thismodel->INPmodType as it might not exist! */
    if (type == INPtypelook("B4SOI") ||
        type == INPtypelook("B3SOIPD") ||
        type == INPtypelook("B3SOIFD") ||
        type == INPtypelook("B3SOIDD") ||
        type == INPtypelook("HiSIMHV1") ||
        type == INPtypelook("HiSIMHV2") ||
        type == INPtypelook("SOI3"))
    {
        switch (numnodes) {
        case 4:
            fast->GENnode[4] = -1;
            fast->GENnode[5] = -1;
            fast->GENnode[6] = -1;
            break;
        case 5:
            IFC (bindNode, (ckt, fast, 4 + 1, node[4]));
            fast->GENnode[5] = -1;
            fast->GENnode[6] = -1;
            break;
        case 6:
            IFC (bindNode, (ckt, fast, 4 + 1, node[4]));
            IFC (bindNode, (ckt, fast, 5 + 1, node[5]));
            fast->GENnode[6] = -1;
            break;
        case 7:
            IFC (bindNode, (ckt, fast, 4 + 1, node[4]));
            IFC (bindNode, (ckt, fast, 5 + 1, node[5]));
            IFC (bindNode, (ckt, fast, 6 + 1, node[6]));
            break;
        default:
            break;
        }
    }

    PARSECALL ((&line, ckt, type, fast, &leadval, &waslead, tab));
    if (waslead)
        LITERR (" error:  no unlabeled parameter permitted on mosfet\n");
}
