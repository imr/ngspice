/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Thomas L. Quarles
Modified: 2001 Paolo Nenzi (Cider Integration)
**********/

#include "ngspice/ngspice.h"
#include "ngspice/ifsim.h"
#include "ngspice/inpdefs.h"
#include "ngspice/inpmacs.h"
#include "ngspice/fteext.h"
#include "inpxx.h"


static int
model_max_numnodes(int type)
{
    if (type == INPtypelook("VBIC") ||
        type == INPtypelook("hicum2"))
        return 5;
#ifdef ADMS
    if (type == INPtypelook("hicum0") ||
        type == INPtypelook("bjt504t"))
        return 5;
#endif
    return 4;
}


void INP2Q(CKTcircuit *ckt, INPtables * tab, struct card *current, CKTnode *gnode)
{

    /* Qname <node> <node> <node> [<node>] <model> [<val>] [OFF]
     *       [IC=<val>,<val>] */

    int type;                   /* the type the model says it is */
    char *line;                 /* the part of the current line left to parse */
    char *name;                 /* the resistor's name */
    const int max_i = 5;
    CKTnode *node[5];
    int error;                  /* error code temporary */
    int numnodes;               /* flag indicating 4 or 5 nodes */
    GENinstance *fast;          /* pointer to the actual instance */
    IFvalue ptemp;              /* a value structure to package resistance into */
    int waslead;                /* flag to indicate that funny unlabeled number was found */
    double leadval;             /* actual value of unlabeled number */
    INPmodel *thismodel;        /* pointer to model description for user's model */
    GENmodel *mdfast;           /* pointer to the actual model */
    int i;

#ifdef TRACE
    printf("INP2Q: Parsing '%s'\n", current->line);
#endif

    line = current->line;

    INPgetNetTok(&line, &name, 1);
    INPinsert(&name, tab);

    for (i = 0; ; i++) {
        char *token;
        INPgetNetTok(&line, &token, 1);
        if (i >= 3 && INPlookMod(token)) {
            INPinsert(&token, tab);
            txfree(INPgetMod(ckt, token, &thismodel, tab));
            if (!thismodel) {
                LITERR ("Unable to find definition of given model");
                return;
            }
            break;
        }
        if (i >= max_i) {
            LITERR ("could not find a valid modelname");
            return;
        }
        INPtermInsert(ckt, &token, tab, &node[i]);
    }

    int model_numnodes = model_max_numnodes(thismodel->INPmodType);
    if (i > model_numnodes) {
        LITERR("Too many nodes for this model type");
        return;
    }

    /* tie missing ports to ground, (substrate and thermal node) */
    while (i < model_numnodes)
        node[i++] = gnode;

    numnodes = i;

#ifdef TRACE
    printf("INP2Q: Looking up model\n");
#endif

    if (thismodel->INPmodType != INPtypelook("BJT") &&
#ifdef CIDER
        thismodel->INPmodType != INPtypelook("NBJT") &&
        thismodel->INPmodType != INPtypelook("NBJT2") &&
#endif
#ifdef ADMS
        thismodel->INPmodType != INPtypelook("hicum0") &&
        thismodel->INPmodType != INPtypelook("bjt504t") &&
#endif
        thismodel->INPmodType != INPtypelook("hicum2") &&
        thismodel->INPmodType != INPtypelook("VBIC"))
    {
        LITERR("incorrect model type");
        return;
    }

    type = thismodel->INPmodType;
    mdfast = thismodel->INPmodfast;

#ifdef TRACE
    printf("INP2Q: Type: %d numnodes: %d instancename: %s\n", type, numnodes, name);
#endif

    IFC(newInstance, (ckt, mdfast, &fast, name));
    for (i = 0; i < numnodes; i++)
        IFC(bindNode, (ckt, fast, i + 1, node[i]));

    PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));

    if (waslead) {
#ifdef CIDER
        if (type == INPtypelook("NBJT2")) {
            LITERR(" error: no unlabeled parameter permitted on NBJT2\n");
            return;
        }
#endif
        ptemp.rValue = leadval;
        GCA(INPpName, ("area", &ptemp, ckt, type, fast));
    }
}
