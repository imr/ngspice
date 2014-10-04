#include "ngspice/ngspice.h"
#include "ngspice/config.h"
#include "ngspice/cktdefs.h"
//#include "ngspice/cktaccept.h"
#include "ngspice/trandefs.h"
#include "ngspice/sperror.h"
#include "ngspice/fteext.h"
#include "ngspice/missing_math.h"
#include "../frontend/outitf.h"

#ifdef XSPICE
/* gtri - add - wbk - Add headers */
#include "ngspice/miftypes.h"

#include "ngspice/evt.h"
#include "ngspice/mif.h"
#include "ngspice/evtproto.h"
#include "ngspice/ipctiein.h"
/* gtri - end - wbk - Add headers */
#endif

#include "ngspice/lxt2_write.h"

#ifdef CLUSTER
#include "cluster.h"
#endif

#define CKALLOC(var,size,type) \
    if(size) { \
        if(!(var = (type *) MALLOC((size) * sizeof(type)))) \
            return(E_NOMEM); \
    }

static void EVTtraceinit(CKTcircuit *ckt);
static void KVLtraceinit(runDesc *run);

void lxt2_init(runDesc *run) {

    #ifdef LXT2_DEBUG
    printf("LXT2 lxt2_init\n");
    #endif

    if(run && run->circuit) {
      #ifdef LXT2_DEBUG
      printf("LXT2 run and run->circuit exist\n");
      printf("\nLXT2 run->name(%s)\n",run->name);
      printf("\nLXT2 Spice_Path(%s)\n",Spice_Path);
      #endif
      run->circuit->lxt2.file = lxt2_wr_init("waveforms.lxt");
      lxt2_wr_set_timescale(run->circuit->lxt2.file, LXT2_TIME_RESOLUTION_EXPONENT);
      printf("LXT2 time resolution is 1.0e%d seconds.\n\n",LXT2_TIME_RESOLUTION_EXPONENT);
      fflush(stdout);
      EVTtraceinit(run->circuit);
      KVLtraceinit(run);
    }
}

void lxt2_end(runDesc *run)
{
    if(run && run->circuit) {
      #ifdef LXT2_DEBUG
      printf("LXT2 final file flush.\n");
      #endif
      lxt2_wr_flush(run->circuit->lxt2.file);
      #ifdef LXT2_DEBUG
      printf("LXT2 close file.\n");
      #endif
      lxt2_wr_close(run->circuit->lxt2.file);
      free(run->circuit->lxt2.evt_table);
      free(run->circuit->lxt2.evt_indexmap);
      free(run->circuit->lxt2.kvl_table);
      free(run->circuit->lxt2.kvl_indexmap);
    }
}

static void EVTtraceinit(CKTcircuit *ckt)       /* the circuit structure */
{

    struct lxt2_wr_symbol  **trace_table = NULL;     /* holmes: vector of pointer to traces in lxt2 output file */
    Evt_Node_Info_t        **node_table;
    int i;
    int num_nodes;

    node_table = ckt->evt->info.node_table;;
    num_nodes  = ckt->evt->counts.num_nodes;

    int *evt_indexmap = NULL;

    /* holmes: Allocate and initialize table of lxt2 evt trace pointers */
    #ifdef LXT2_DEBUG
    printf("LXT2 Allocate trace_table\n");
    #endif
    CKALLOC(trace_table, num_nodes, struct lxt2_wr_symbol *);
    CKALLOC(evt_indexmap, num_nodes, int);
    for(i = 0; i < num_nodes; i++) {
        evt_indexmap[i] = i;
    	switch(node_table[i]->udn_index)
    	{
    	  case 0:   /* Bit */
    		trace_table[i] = lxt2_wr_symbol_add(ckt->lxt2.file, node_table[i]->name, 0, 0, 0, LXT2_WR_SYM_F_BITS);
		#ifdef LXT2_DEBUG
		printf("LXT2 created EVT BIT trace_table[%d] pointer(%p) name(%s)\n",i,&trace_table[i],node_table[i]->name);
		#endif
		break;
    	  case 1:   /* Integer */
    		trace_table[i] = lxt2_wr_symbol_add(ckt->lxt2.file, node_table[i]->name, 0, 0, 0, LXT2_WR_SYM_F_INTEGER);
		#ifdef LXT2_DEBUG
		printf("LXT2 created EVT INT trace_table[%d] pointer(%p) name(%s)\n",i,&trace_table[i],node_table[i]->name);
		#endif
		break;
    	  case 2:   /* Double */
    		trace_table[i] = lxt2_wr_symbol_add(ckt->lxt2.file, node_table[i]->name, 0, 0, 0, LXT2_WR_SYM_F_DOUBLE);
		#ifdef LXT2_DEBUG
		printf("LXT2 created EVT DUB trace_table[%d] pointer(%p) name(%s)\n",i,&trace_table[i],node_table[i]->name);
		#endif
		break;
    	  default:  /* Bit */
    		trace_table[i] = lxt2_wr_symbol_add(ckt->lxt2.file, node_table[i]->name, 0, 0, 0, LXT2_WR_SYM_F_BITS);
		#ifdef LXT2_DEBUG
		printf("LXT2 created EVT BIT trace_table[%d] pointer(%p) name(%s)\n",i,&trace_table[i],node_table[i]->name);
		#endif
		break;
    	}
    }
    fflush(stdout);
    ckt->lxt2.evt_indexmap = evt_indexmap;
    ckt->lxt2.evt_table = trace_table;
    ckt->lxt2.evt_num = num_nodes;
}

static void KVLtraceinit(runDesc *run)   /* Call this after OUTpBeginPlot after all electrical run data is loaded */
{
    struct lxt2_wr_symbol  **trace_table = NULL;
    int *kvl_indexmap = NULL;
    int i;
    if(run->circuit) {
       CKALLOC(trace_table, run->numData, struct lxt2_wr_symbol *);
       CKALLOC(kvl_indexmap, run->numData, int);
       for(i = 0; i < run->numData; i++) {
    	  kvl_indexmap[i] = -1;
    	  trace_table[i] = NULL;
    	  if (run->data[i].outIndex != -1) {  /* Skip over the time vector. Once we parse the saves list this check won't be necessary. */
    	     if (run->data[i].regular) {	      /* Check that the data is not "special" like a parameter, might allow this later */
    		if (run->data[i].type == IF_REAL) {   /* As opposed to being complex */
 	   	   kvl_indexmap[i] = i;
 	   	   trace_table[i] = lxt2_wr_symbol_add(run->circuit->lxt2.file, run->data[i].name, 0, 0, 0, LXT2_WR_SYM_F_DOUBLE);
 	   	   #ifdef LXT2_DEBUG
		   printf("LXT2 created KVL DUB trace_table[%d] pointer(%p) name(%s)\n",i,&trace_table[i],run->data[i].name);
                   #endif
		   fflush(stdout);
    		}
 	     }
    	  }
       }
       run->circuit->lxt2.kvl_indexmap = kvl_indexmap;
       run->circuit->lxt2.kvl_table = trace_table;
       run->circuit->lxt2.kvl_num = run->numData;
    }
}
