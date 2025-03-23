#ifndef ngspice_EVTPROTO_H
#define ngspice_EVTPROTO_H

/* ===========================================================================
FILE    EVTproto.h

MEMBER OF process XSPICE

Copyright 1991
Georgia Tech Research Corporation
Atlanta, Georgia 30332
All Rights Reserved

PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn

MODIFICATIONS

    <date> <person name> <nature of modifications>

SUMMARY

    This file contains ANSI C function prototypes for functions
    in the event-driven simulation algorithm package.

INTERFACES

    None.

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

=========================================================================== */

#include "ngspice/cktdefs.h"
#include "ngspice/cpstd.h"
#include "ngspice/evt.h"
#include "ngspice/mifdefs.h"
#include "ngspice/ipc.h"


/* ******************* */
/* Function Prototypes */
/* ******************* */


int EVTinit(CKTcircuit *ckt);
/*int EVTinit2(CKTcircuit *ckt);*/

void EVTtermInsert(
    CKTcircuit      *ckt,
    MIFinstance     *fast,
    char            *node_name,
    char            *type_name,
    int             conn_num,
    int             port_num,
    char            **err_msg);

int EVTsetup(CKTcircuit *ckt);
int EVTunsetup(CKTcircuit* ckt);

int EVTdest(Evt_Ckt_Data_t *evt);

int EVTiter(CKTcircuit *ckt);

void EVTbackup(CKTcircuit *ckt, double new_time);

double EVTnext_time(CKTcircuit *ckt);

void EVTqueue_output(
    CKTcircuit *ckt,
    int        output_index,
    int        udn_index,
    Evt_Output_Event_t  *new_event,
    double     posted_time,
    double     event_time);


void EVTqueue_inst(
    CKTcircuit *ckt,
    int        inst_index,
    double     posted_time,
    double     event_time);

void EVTdequeue(CKTcircuit *ckt, double time);

int EVTload(CKTcircuit *ckt, MIFinstance *inst);

int EVTload_with_event(CKTcircuit *ckt, MIFinstance *inst, Mif_Call_Type_t type);

void EVTprint(wordlist *wl);
void EVTprintvcd(wordlist *wl);
void EVTsave(wordlist *wl);
void EVTdisplay(wordlist *wl);

int EVTop(
    CKTcircuit *ckt,
    long       firstmode,
    long       continuemode,
    int        max_iter,
    Mif_Boolean_t  first_call);

void EVTop_save(
    CKTcircuit    *ckt,
    Mif_Boolean_t op,
    double        step);

void EVTnode_copy(
    CKTcircuit    *ckt,
    int           node_index,
    Evt_Node_t    *from,
    Evt_Node_t    **to);

void EVTcall_hybrids(CKTcircuit *ckt);

void EVTdump(
    CKTcircuit    *ckt,
    Ipc_Anal_t    mode,
    double        step);

void EVTaccept(
    CKTcircuit *ckt,    /* main circuit struct */
    double     time);    /* time at which analog soln was accepted */

struct INPtables;
bool Evtcheck_nodes(
    CKTcircuit         *ckt,             /* The circuit structure */
    struct INPtables   *stab);           /* Symbol table. */

struct dvec *EVTfindvec(char *node);

/* Set and remove call-backs on new node values. */

Mif_Boolean_t EVTnew_value_call(const char         *node,
                                Evt_New_Value_Cb_t  fn,
                                Evt_Node_Cb_Type_t  type,
                                void               *ctx);

void EVTcancel_value_call(const char         *node,
                          Evt_New_Value_Cb_t  fn,
                          void               *ctx);

/* Internal utility functions. */

void Evt_purge_free_outputs(void);

/* Parse a node name with member and find the node index. */

struct node_parse {
    char            *node;
    char            *member;
};

int Evt_Parse_Node(const char *node, struct node_parse *result);
#endif
