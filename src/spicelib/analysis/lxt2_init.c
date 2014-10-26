#include "ngspice/ngspice.h"
#include "ngspice/sperror.h"
#include "ngspice/fteext.h"

#include "../frontend/outitf.h"

#ifdef XSPICE
#include "ngspice/evtproto.h"
#endif

#include "ngspice/lxt2_write.h"


#define CKALLOC(var, size, type) \
    if(size) { \
        if(!(var = (type *) MALLOC((size) * sizeof(type)))) \
            return(E_NOMEM); \
    }


static void EVTtraceinit(CKTcircuit *ckt);
static void KVLtraceinit(runDesc *run);


void
lxt2_init(runDesc *run)
{
    if (run && run->circuit) {
        run->circuit->lxt2.file = lxt2_wr_init("waveforms.lxt");
        lxt2_wr_set_timescale(run->circuit->lxt2.file, LXT2_TIME_RESOLUTION_EXPONENT);
        EVTtraceinit(run->circuit);
        KVLtraceinit(run);
    }
}


void
lxt2_end(runDesc *run)
{
    if (run && run->circuit) {
        lxt2_wr_flush(run->circuit->lxt2.file);
        lxt2_wr_close(run->circuit->lxt2.file);
        free(run->circuit->lxt2.evt_table);
        free(run->circuit->lxt2.evt_indexmap);
        free(run->circuit->lxt2.kvl_table);
        free(run->circuit->lxt2.kvl_indexmap);
    }
}


static void
EVTtraceinit(CKTcircuit *ckt)       /* the circuit structure */
{

    struct lxt2_wr_symbol **trace_table = NULL;     /* vector of pointer to traces in lxt2 output file */
    Evt_Node_Info_t       **node_table;
    int i;
    int num_nodes;

    int *evt_indexmap = NULL;

    node_table = ckt->evt->info.node_table;
    num_nodes  = ckt->evt->counts.num_nodes;


    /* Allocate and initialize table of lxt2 evt trace pointers */
    CKALLOC(trace_table, num_nodes, struct lxt2_wr_symbol *);
    CKALLOC(evt_indexmap, num_nodes, int);

    for (i = 0; i < num_nodes; i++) {

        evt_indexmap[i] = i;

        switch (node_table[i]->udn_index)
        {
        case 0:   /* Bit */
            trace_table[i] = lxt2_wr_symbol_add(ckt->lxt2.file, node_table[i]->name, 0, 0, 0, LXT2_WR_SYM_F_BITS);
            break;
        case 1:   /* Integer */
            trace_table[i] = lxt2_wr_symbol_add(ckt->lxt2.file, node_table[i]->name, 0, 0, 0, LXT2_WR_SYM_F_INTEGER);
            break;
        case 2:   /* Double */
            trace_table[i] = lxt2_wr_symbol_add(ckt->lxt2.file, node_table[i]->name, 0, 0, 0, LXT2_WR_SYM_F_DOUBLE);
            break;
        default:  /* Bit */
            trace_table[i] = lxt2_wr_symbol_add(ckt->lxt2.file, node_table[i]->name, 0, 0, 0, LXT2_WR_SYM_F_BITS);
            break;
        }
    }

    ckt->lxt2.evt_indexmap = evt_indexmap;
    ckt->lxt2.evt_table = trace_table;
    ckt->lxt2.evt_num = num_nodes;
}


static void
KVLtraceinit(runDesc *run)   /* Call this after OUTpBeginPlot after all electrical run data is loaded */
{
    struct lxt2_wr_symbol **trace_table = NULL;
    int *kvl_indexmap = NULL;
    int i;

    if (run->circuit) {

        CKALLOC(trace_table, run->numData, struct lxt2_wr_symbol *);
        CKALLOC(kvl_indexmap, run->numData, int);

        for (i = 0; i < run->numData; i++) {
            kvl_indexmap[i] = -1;
            trace_table[i] = NULL;
            if (run->data[i].outIndex != -1) {  /* Skip over the time vector. Once we parse the saves list this check won't be necessary. */
                if (run->data[i].regular) {  /* Check that the data is not "special" like a parameter, might allow this later */
                    if (run->data[i].type == IF_REAL) {  /* As opposed to being complex */
                        kvl_indexmap[i] = i;
                        trace_table[i] = lxt2_wr_symbol_add(run->circuit->lxt2.file, run->data[i].name, 0, 0, 0, LXT2_WR_SYM_F_DOUBLE);
                    }
                }
            }
        }

        run->circuit->lxt2.kvl_indexmap = kvl_indexmap;
        run->circuit->lxt2.kvl_table = trace_table;
        run->circuit->lxt2.kvl_num = run->numData;
    }
}
