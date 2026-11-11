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
    if (type == INPtypelook("VBIC"))
        return 5;
#ifdef OSDI
    /* Verilog-A module, e.g. bound to a bipolar level with osdi_bind */
    if (ft_sim->devices[type]->registry_entry)
        return *ft_sim->devices[type]->terms;
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
    const int max_i = 7;
    CKTnode *node[7];
    bool is_osdi = FALSE;
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

#ifdef OSDI
    is_osdi = ft_sim->devices[thismodel->INPmodType]->registry_entry != NULL;
#endif
    if (is_osdi) {
        /* Verilog-A module: tie a missing substrate (4th terminal) to
           ground; further terminals (thermal node) stay internal, so
           that self-heating acts as with N instances */
        if (i == 3 && model_numnodes >= 4)
            node[i++] = gnode;
    } else {
        /* tie missing ports to ground, (substrate and thermal node) */
        while (i < model_numnodes)
            node[i++] = gnode;
    }

    numnodes = i;

#ifdef TRACE
    printf("INP2Q: Looking up model\n");
#endif

    if (thismodel->INPmodType != INPtypelook("BJT") &&
#ifdef CIDER
        thismodel->INPmodType != INPtypelook("NBJT") &&
        thismodel->INPmodType != INPtypelook("NBJT2") &&
#endif
        thismodel->INPmodType != INPtypelook("VBIC") &&
        /* OSDI module, unless a card bound to another device kind */
        !(is_osdi && INPosdiCardFits('q', thismodel->INPmodLine->line)))
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
    /* unconnected terminals of a Verilog-A module */
    for (; is_osdi && i < model_numnodes; i++)
        GENnode(fast)[i] = -1;

#ifdef OSDI
    /* instance parameter names given with 'osdi_bind ... alias=' */
    const struct osdi_bind *bind =
        is_osdi ? INPosdiBindOfCard(thismodel->INPmodLine->line) : NULL;
    char *aliased = INPosdiAliasLine(bind, line);
    if (aliased)
        line = aliased;
#endif
    PARSECALL((&line, ckt, type, fast, &leadval, &waslead, tab));
#ifdef OSDI
    tfree(aliased);
#endif

    if (waslead) {
#ifdef CIDER
        if (type == INPtypelook("NBJT2")) {
            LITERR(" error: no unlabeled parameter permitted on NBJT2\n");
            return;
        }
#endif
        ptemp.rValue = leadval;
        char *areaname = "area";
#ifdef OSDI
        /* the unlabeled value is the parameter area, or its alias */
        if (is_osdi) {
            IFdevice *dev = ft_sim->devices[type];
            areaname = (char *) INPosdiAlias(bind, "area");
            bool found = FALSE;
            for (int k = 0; k < *dev->numInstanceParms && !found; k++)
                found = strcasecmp(dev->instanceParms[k].keyword, areaname) == 0;
            if (!found) {
                char *msg = tprintf("module %s has no instance parameter %s for the "
                                    "unlabeled value; name the parameter, or use "
                                    "'osdi_bind ... alias=area:<par>'",
                                    dev->name, areaname);
                LITERR(msg);
                tfree(msg);
                return;
            }
        }
#endif
        GCA(INPpName, (areaname, &ptemp, ckt, type, fast));
    }
}
