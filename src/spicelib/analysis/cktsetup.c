/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
**********/

    /* CKTsetup(ckt)
     * this is a driver program to iterate through all the various
     * setup functions provided for the circuit elements in the
     * given circuit
     */

#include "ngspice/ngspice.h"
#include "ngspice/smpdefs.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/gendefs.h"
#include "ngspice/sperror.h"
#include "ngspice/fteext.h"
#include "ngspice/cpextern.h"

/* device headers needed by CKTtopologyReduce() to mark dangling passives */
#include "../devices/cap/capdefs.h"
#include "../devices/res/resdefs.h"

#ifdef XSPICE
#include "ngspice/enh.h"
/* XSPICE code models store their terminals in the MIF conn/port tree, not in
 * the generic GENnode() array, and report DEVpublic.terms == 0.  CKTtopologyReduce()
 * walks that tree explicitly so code-model nodes are counted. */
#include "ngspice/mifdefs.h"
#endif

#ifdef USE_OMP
#include <omp.h>
#include "ngspice/cpextern.h"
#endif

#define CKALLOC(var,size,type) \
    if(size && ((var = TMALLOC(type, size)) == NULL)){\
            return(E_NOMEM);\
}

/* Topology reduction for removing dangling capacitors and resistors.
 *
 * A node whose only connection is a single passive terminal (a degree-1
 * "dangling" node, e.g. the dead end of an opamp compensation cap) carries no
 * steady current/charge, but its row becomes ill-conditioned as the timestep
 * shrinks and shows up as a spurious "Timestep too small" abort.  Commercial
 * fast simulators simply remove such elements from the matrix.  This
 * pass does the same: it finds dangling passive leaves and marks the owning
 * capacitor/resistor so that, at load time, the device contributes nothing and
 * its floating node is pinned with a unit diagonal (kept nonsingular).
 *
 * Node degree is counted GENERICALLY over every device type through the
 * GENnode() terminal array, so no device family can be missed (which would
 * wrongly prune a live node).  XSPICE code models are the one exception: they
 * report DEVpublic.terms == 0 and keep their node numbers in the MIF conn/port
 * tree rather than in GENnode(), so they are walked separately below -- without
 * that, a node touched only by a code model would look floating and its passive
 * would be wrongly pruned.  Only capacitors and resistors are ever removed.
 * Disable with `set no_topo_reduce` in .spiceinit. It is disabled as well
 * if option rshunt=xx is selected.*/
static void
CKTtopologyReduce(CKTcircuit *ckt)
{
    int i, t, nterm, maxnode, removed_total = 0, reported = 0;
    int captype = -1, restype = -1;
    int *degree;
    GENmodel *gmod;
    GENinstance *ginst;

    if (cp_getvar("no_topo_reduce", CP_BOOL, NULL, 0))
        return;

#ifdef XSPICE
    if (ckt->enh->rshunt_data.enabled)
        return;
#endif

    maxnode = ckt->CKTmaxEqNum;
    if (maxnode < 1)
        return;

    degree = TMALLOC(int, maxnode + 1);
    if (!degree)
        return;
    for (i = 0; i <= maxnode; i++)
        degree[i] = 0;

    /* complete, type-agnostic node degree over all device terminals */
    for (i = 0; i < DEVmaxnum; i++) {
        if (!DEVices[i] || !ckt->CKThead[i])
            continue;
        nterm = *(DEVices[i]->DEVpublic.terms);
        if (DEVices[i]->DEVpublic.name) {
            if (!strcmp(DEVices[i]->DEVpublic.name, "Capacitor"))
                captype = i;
            else if (!strcmp(DEVices[i]->DEVpublic.name, "Resistor"))
                restype = i;
        }

#ifdef XSPICE
        /* XSPICE code models: DEVpublic.terms is 0 and GENnode() does not hold
         * their nodes, so the generic walk below would count nothing for them.
         * Walk the MIF conn/port tree and count every analog port's pos/neg node
         * instead.  Event-driven (digital) and vsource-current ports leave these
         * at 0 (calloc'd) and are skipped by the nd > 0 test.
         *
         * A code-model instance IS a MIFinstance, so it is identified by its
         * instance size -- a plain compile-time integer (sizeof(MIFinstance))
         * that is identical across the .cm modules.  A function-pointer test
         * (DEVsetup == MIFsetup) is NOT reliable here: code models linked into
         * separate .cm shared objects carry their own copy of MIFsetup, so the
         * pointers differ from this translation unit's. */
        if (DEVices[i]->DEVinstSize &&
            *(DEVices[i]->DEVinstSize) == (int) sizeof(MIFinstance)) {
            for (gmod = ckt->CKThead[i]; gmod; gmod = gmod->GENnextModel)
                for (ginst = gmod->GENinstances; ginst; ginst = ginst->GENnextInstance) {
                    MIFinstance *mif = (MIFinstance *) ginst;
                    int c, p;
                    for (c = 0; c < mif->num_conn; c++) {
                        Mif_Conn_Data_t *conn = mif->conn[c];
                        if (!conn || conn->is_null)
                            continue;
                        for (p = 0; p < conn->size; p++) {
                            Mif_Port_Data_t *port = conn->port[p];
                            int nd;
                            if (!port || port->is_null)
                                continue;
                            nd = port->smp_data.pos_node;
                            if (nd > 0 && nd <= maxnode) degree[nd]++;
                            nd = port->smp_data.neg_node;
                            if (nd > 0 && nd <= maxnode) degree[nd]++;
                        }
                    }
                }
            continue;   /* code model handled; skip the generic terminal walk */
        }
#endif

        if (!DEVices[i]->DEVpublic.terms)
            continue;
        nterm = *(DEVices[i]->DEVpublic.terms);
        for (gmod = ckt->CKThead[i]; gmod; gmod = gmod->GENnextModel)
            for (ginst = gmod->GENinstances; ginst; ginst = ginst->GENnextInstance) {
                int *nodes = GENnode(ginst);
                for (t = 0; t < nterm; t++) {
                    int nd = nodes[t];
                    if (nd > 0 && nd <= maxnode)
                        degree[nd]++;
                }
            }
    }

    if (captype < 0 && restype < 0) {
        FREE(degree);
        return;
    }

    /* Leaf-prune to a fixpoint: removing a dangling passive can drop its other
     * node to degree 1, exposing the next leaf of a dangling chain. */
    for (;;) {
        int removed_this_pass = 0;

        if (captype >= 0) {
            CAPmodel *cm;
            for (cm = (CAPmodel *)ckt->CKThead[captype]; cm; cm = CAPnextModel(cm))
                for (CAPinstance *ci = CAPinstances(cm); ci; ci = CAPnextInstance(ci)) {
                    int pn = ci->CAPposNode, nn = ci->CAPnegNode, mode = 0;
                    if (ci->CAPdangling)
                        continue;
                    if (pn > 0 && degree[pn] == 1)
                        mode |= 1;
                    if (nn > 0 && degree[nn] == 1)
                        mode |= 2;
                    if (!mode)
                        continue;
                    ci->CAPdangling = mode;
                    if (mode & 1)
                        degree[pn] = 0;
                    else if (pn > 0)
                        degree[pn]--;
                    if (mode & 2)
                        degree[nn] = 0;
                    else if (nn > 0)
                        degree[nn]--;
                    removed_this_pass++;
                    if (ft_stricterror) {
                        fprintf(stderr,
                            "Dangling capacitor %s "
                            "(floating node %s) found in netlist.\n", ci->CAPname,
                            (char*)CKTnodName(ckt, (mode & 2) ? nn : pn));
                    }
                    else if (reported++ < 40)
                        fprintf(stdout,
                            "Topology reduction: removed dangling capacitor %s "
                            "(floating node %s)\n", ci->CAPname,
                            (char *)CKTnodName(ckt, (mode & 2) ? nn : pn));
                }
        }

        if (restype >= 0) {
            RESmodel *rm;
            for (rm = (RESmodel *)ckt->CKThead[restype]; rm; rm = RESnextModel(rm))
                for (RESinstance *ri = RESinstances(rm); ri; ri = RESnextInstance(ri)) {
                    int pn = ri->RESposNode, nn = ri->RESnegNode, mode = 0;
                    if (ri->RESdangling)
                        continue;
                    if (pn > 0 && degree[pn] == 1)
                        mode |= 1;
                    if (nn > 0 && degree[nn] == 1)
                        mode |= 2;
                    if (!mode)
                        continue;
                    ri->RESdangling = mode;
                    if (mode & 1)
                        degree[pn] = 0;
                    else if (pn > 0)
                        degree[pn]--;
                    if (mode & 2)
                        degree[nn] = 0;
                    else if (nn > 0)
                        degree[nn]--;
                    removed_this_pass++;
                    if (ft_stricterror) {
                        fprintf(stderr,
                            "Dangling resistor %s "
                            "(floating node %s) found in netlist.\n", ri->RESname,
                            (char*)CKTnodName(ckt, (mode & 2) ? nn : pn));
                    }
                    else if (reported++ < 40)
                        fprintf(stdout,
                            "Topology reduction: removed dangling resistor %s "
                            "(floating node %s)\n", ri->RESname,
                            (char*)CKTnodName(ckt, (mode & 2) ? nn : pn));
                }
        }

        removed_total += removed_this_pass;
        if (!removed_this_pass)
            break;
    }

    if (removed_total) {
        if (ft_stricterror) {
            fprintf(stderr, "\nError: %d dangling passive(s) found.\n "
                "    Please correct the netlist.\n", removed_total);
            FREE(degree);
            controlled_exit(EXIT_BAD);
        }

        fprintf(stdout, "Topology reduction: %d dangling passive(s) removed "
            "from the matrix.\n", removed_total);
    }

    FREE(degree);
}

int
CKTsetup(CKTcircuit *ckt)
{
    int i;
    int error;
#ifdef USE_OMP
    int nthreads = 2;
#endif
#ifdef XSPICE
 /* gtri - begin - Setup for adding rshunt option resistors */
    CKTnode *node;
    int     num_nodes;
 /* gtri - end - Setup for adding rshunt option resistors */

#ifdef KLU
    BindElement BindNode, *matched, *BindStruct ;
    size_t nz ;
#endif
#endif

    SMPmatrix *matrix;

    if (!ckt->CKThead) {
        fprintf(stderr, "Error: No model list found, device setup not possible!\n");
        if (ft_stricterror)
            controlled_exit(EXIT_BAD);
        return E_PANIC;
    }
    if (!DEVices) {
        fprintf(stderr, "Error: No device list found, device setup not possible!\n");
        if (ft_stricterror)
            controlled_exit(EXIT_BAD);
        return E_PANIC;
    }

    ckt->CKTnumStates=0;

#ifdef WANT_SENSE2
    if(ckt->CKTsenInfo){
        error = CKTsenSetup(ckt);
        if (error)
            return(error);
    }
#endif

    if (ckt->CKTisSetup)
        return E_NOCHANGE;

    error = NIinit(ckt);
    if (error) 
        return(error);

    ckt->CKTisSetup = 1;

    matrix = ckt->CKTmatrix;

#ifdef USE_OMP
    if (!cp_getvar("num_threads", CP_NUM, &nthreads, 0))
        nthreads = 2;

    omp_set_num_threads(nthreads);
/*    if (nthreads == 1)
      printf("OpenMP: %d thread is requested in ngspice\n", nthreads);
    else
      printf("OpenMP: %d threads are requested in ngspice\n", nthreads);*/
#endif

#ifdef HAS_PROGREP
    SetAnalyse("Device Setup", 0);
#endif

    /* preserve CKTlastNode before invoking DEVsetup()
     * so we can check for incomplete CKTdltNNum() invocations
     * during DEVunsetup() causing an erronous circuit matrix
     *   when reinvoking CKTsetup()
     */
    ckt->prev_CKTlastNode = ckt->CKTlastNode;

    for (i=0;i<DEVmaxnum;i++) {
        if ( DEVices[i] && DEVices[i]->DEVsetup && ckt->CKThead[i] ) {
            error = DEVices[i]->DEVsetup (matrix, ckt->CKThead[i], ckt,
                    &ckt->CKTnumStates);
            if(error) return(error);
        }
    }

    /* Topology reduction for removing dangling capacitors and resistors:
     * mark dangling (degree-1) passive leaves
     * for removal.  Done after all device setups (so the node degrees are
     * complete) but before the matrix is converted/bound for the linear
     * solver -- it only changes load-time stamping, not the matrix structure. */
    CKTtopologyReduce(ckt);

#ifdef XSPICE
  /* gtri - begin - Setup for adding rshunt option resistors */

    if(ckt->enh->rshunt_data.enabled) {

        /* Count number of voltage nodes in circuit */
        for(num_nodes = 0, node = ckt->CKTnodes; node; node = node->next)
            if((node->type == SP_VOLTAGE) && (node->number != 0))
                num_nodes++;

        /* Allocate space for the matrix diagonal data */
        if(num_nodes > 0) {
            FREE(ckt->enh->rshunt_data.diag);
            ckt->enh->rshunt_data.diag =
                 TMALLOC(double *, num_nodes);
        }

        /* Set the number of nodes in the rshunt data */
        ckt->enh->rshunt_data.num_nodes = num_nodes;

        /* Get/create matrix diagonal entry following what RESsetup does */
        for(i = 0, node = ckt->CKTnodes; node; node = node->next) {
            if((node->type == SP_VOLTAGE) && (node->number != 0)) {
                ckt->enh->rshunt_data.diag[i] =
                      SMPmakeElt(matrix,node->number,node->number);
                i++;
            }
        }
    }

    /* gtri - end - Setup for adding rshunt option resistors */
#endif

#ifdef KLU
    if (ckt->CKTmatrix->CKTkluMODE)
    {
        fprintf (stdout, "Using KLU as Direct Linear Solver\n") ;

        /* Convert the COO Storage to CSC for KLU and Fill the Binding Table */
        SMPconvertCOOtoCSC (matrix) ;

        /* Assign the KLU Pointers */
        for (i = 0 ; i < DEVmaxnum ; i++)
            if (DEVices [i] && DEVices [i]->DEVbindCSC && ckt->CKThead [i])
                DEVices [i]->DEVbindCSC (ckt->CKThead [i], ckt) ;

#ifdef XSPICE
        if (ckt->enh->rshunt_data.num_nodes > 0) {
            BindStruct = ckt->CKTmatrix->SMPkluMatrix->KLUmatrixBindStructCOO ;
            nz = (size_t)ckt->CKTmatrix->SMPkluMatrix->KLUmatrixLinkedListNZ ;
            for(i = 0, node = ckt->CKTnodes; node; node = node->next) {
                if((node->type == SP_VOLTAGE) && (node->number != 0)) {
                    BindNode.COO = ckt->enh->rshunt_data.diag [i] ;
                    BindNode.CSC = NULL ;
                    BindNode.CSC_Complex = NULL ;
                    matched = (BindElement *) bsearch (&BindNode, BindStruct, nz, sizeof (BindElement), BindCompare) ;
                    if (!matched) {
                        fprintf (stderr, "Error: Ptr %p not found in BindStruct Table\n", ckt->enh->rshunt_data.diag [i]) ;
                        ckt->enh->rshunt_data.diag[i] = NULL;
                    }
                    else
                        ckt->enh->rshunt_data.diag [i] = matched->CSC ;
                    i++;
                }
            }
        }
#endif

    } else {
        fprintf (stdout, "Using SPARSE 1.3 as Direct Linear Solver\n") ;
    }
#endif

    for(i=0;i<=MAX(2,ckt->CKTmaxOrder)+1;i++) { /* dctran needs 3 states as minimum */
        CKALLOC(ckt->CKTstates[i],ckt->CKTnumStates,double);
    }
#ifdef WANT_SENSE2
    if(ckt->CKTsenInfo){
        /* to allocate memory to sensitivity structures if
         * it is not done before */

        error = NIsenReinit(ckt);
        if(error) return(error);
    }
#endif
    if(ckt->CKTniState & NIUNINITIALIZED) {
        error = NIreinit(ckt);
        if(error) return(error);
    }

    return(OK);
}

int
CKTunsetup(CKTcircuit *ckt)
{
    int i, error, e2;
    CKTnode *node;

    error = OK;
    if (!ckt->CKTisSetup)
        return OK;

    for(i=0;i<=ckt->CKTmaxOrder+1;i++) {
        tfree(ckt->CKTstates[i]);
    }

    /* added by HT 050802*/
    for(node=ckt->CKTnodes;node;node=node->next){
        if(node->icGiven || node->nsGiven) {
            node->ptr=NULL;
        }
    }

    for (i=0;i<DEVmaxnum;i++) {
        if ( DEVices[i] && DEVices[i]->DEVunsetup && ckt->CKThead[i] ) {
            e2 = DEVices[i]->DEVunsetup (ckt->CKThead[i], ckt);
            if (!error && e2)
                error = e2;
        }
    }

    if (ckt->prev_CKTlastNode != ckt->CKTlastNode) {
        fprintf(stderr, "Internal Error: incomplete CKTunsetup(), this will cause serious problems, please report this issue !\n");
        controlled_exit(EXIT_FAILURE);
    }
    ckt->prev_CKTlastNode = NULL;

    ckt->CKTisSetup = 0;
    if(error) return(error);

    NIdestroy(ckt);
    /*
    if (ckt->CKTmatrix)
        SMPdestroy(ckt->CKTmatrix);
    ckt->CKTmatrix = NULL;
    */

    return OK;
}
