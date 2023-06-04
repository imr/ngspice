/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: AlansFixes
**********/

#include "ngspice/ngspice.h"
#include "ngspice/iferrmsg.h"
#include "ngspice/ifsim.h"
#include "ngspice/inpmacs.h"
#include "ngspice/cktdefs.h"

#include "inppas3.h"

extern IFsimulator *ft_sim;


/* pass 3 - Read all nodeset and IC lines. All circuit nodes will have
 * been created by now, (except for internal device nodes), so any
 * nodeset or IC nodes which have to be created are flagged with a
 * warning.  */

void
INPpas3(CKTcircuit *ckt, struct card *data, INPtables *tab, TSKtask *task,
        IFparm *nodeParms, int numNodeParms)
{

    struct card *current;
    int error;			/* used by the macros defined above */
    char *line;			/* the part of the current line left
                                   to parse */
    char *token=NULL;		/* a token from the line */
    IFparm *prm;		/* pointer to parameter to search
                                   through array */
    IFvalue ptemp;		/* a value structure to package
                                   resistance into */
    int which;			/* which analysis we are performing */
    CKTnode *node1;		/* the first node's node pointer */

    NG_IGNORE(task);

#ifdef TRACE
    /* SDB debug statement */
    printf("In INPpas3 . . . \n");
#endif

    for(current = data; current != NULL; current = current->nextcard) {
        line = current->line;
        FREE(token);
        INPgetTok(&line,&token,1);

        if (strcmp(token,".nodeset")==0) {
            which = -1;

            for(prm = nodeParms; prm < nodeParms + numNodeParms; prm++) {
                if(strcmp(prm->keyword,"nodeset")==0) {
                    which = prm->id;
                    break;
                }
            }

            if(which == -1) {
                LITERR("nodeset unknown to simulator. \n");
                goto quit;
            }

            for(;;) {
                char *name;     /* the node's name */

                /* loop until we run out of data */
                INPgetTok(&line,&name,1);
                if( *name == '\0') {
                    FREE(name);
                    break; /* end of line */
                }

                /* If we have 'all = value' , then set all voltage nodes to 'value',
                   except for ground node at node->number 0 */
                if ( cieq(name, "all")) {
                    ptemp.rValue = INPevaluate(&line,&error,1);
                    for (node1 = ckt->CKTnodes; node1 != NULL; node1 = node1->next) {
                        if ((node1->type == SP_VOLTAGE) && (node1->number > 0))
                            IFC(setNodeParm, (ckt, node1, which, &ptemp, NULL));
                    }
                    FREE(name);
                    break;
                }
                /* check to see if in the form V(xxx) and grab the xxx */
                if( (*name == 'V' || *name == 'v') && !name[1] ) {
                    /* looks like V - must be V(xx) - get xx now*/
                    char *nodename;
                    INPgetNetTok(&line,&nodename,1);
                    /* If node is not found, issue a warning, ignore the defective token */
                    if (INPtermSearch(ckt, &nodename, tab, &node1) != E_EXISTS) {
                        fprintf(stderr,
                            "Warning : Nodeset on non-existent node - %s, ignored\n", nodename);
                        fprintf(stderr,
                            "   Please check line %s\n\n", current->line);
                        FREE(name);
                        /* Gobble the rest of the token */
                        line = nexttok(line);
                        continue;
                    }
                    ptemp.rValue = INPevaluate(&line,&error,1);
                    IFC(setNodeParm, (ckt, node1, which, &ptemp, NULL));
                    FREE(name);
                    continue;
                }
                LITERR(" Error: .nodeset syntax error.\n");
                FREE(name);
                break;
            }
        } else if ((strcmp(token,".ic") == 0)) {
            /* .ic */
            which = -1;
            for(prm = nodeParms; prm < nodeParms + numNodeParms; prm++) {
                if(strcmp(prm->keyword,"ic")==0) {
                    which = prm->id;
                    break;
                }
            }

            if(which==-1) {
                LITERR("ic unknown to simulator. \n");
                goto quit;
            }

            for(;;) {
                char *name;     /* the node's name */

                /* loop until we run out of data */
                INPgetTok(&line,&name,1);
                /* check to see if in the form V(xxx) and grab the xxx */
                if( *name == '\0') {
                    FREE(name);
                    break; /* end of line */
                }
                if( (*name == 'V' || *name == 'v') && !name[1] ) {
                    /* looks like V - must be V(xx) - get xx now*/
                    char *nodename;
                    INPgetNetTok(&line,&nodename,1);
                    /* If node is not found, issue a warning, ignore the defective token */
                    if (INPtermSearch(ckt, &nodename, tab, &node1) != E_EXISTS) {
                        fprintf(stderr,
                            "Warning : IC on non-existent node - %s, ignored\n", nodename);
                        fprintf(stderr,
                            "   Please check line %s\n\n", current->line);
                        FREE(name);
                        /* Gobble the rest of the token */
                        line = nexttok(line);
                        if (!line)
                            break;
                        continue;
                    }
                    ptemp.rValue = INPevaluate(&line,&error,1);
                    IFC(setNodeParm, (ckt, node1, which, &ptemp, NULL));
                    FREE(name);
                    continue;
                }
                LITERR(" Error: .ic syntax error.\n");
                FREE(name);
                break;
            }
        }
    }
quit:
    FREE(token);
    return;
}

