#include "ngspice/ngspice.h"
#include "ngspice/fteext.h"

#include "../frontend/outitf.h"

#include <math.h>

#ifdef XSPICE
#include "ngspice/evtproto.h"
#endif

#include "ngspice/lxt2_write.h"

#define CKALLOC(var,size,type) \
    if(size) { \
        if(!(var = (type *) MALLOC((size) * sizeof(type)))) \
            return(E_NOMEM); \
    }

void CKTemitlxt2(runDesc *run)
{
    struct lxt2_wr_symbol  **trace_table;
    struct lxt2_wr_symbol  *trace;
    char *name;
    double value;
    static unsigned int last_set_time, set_time;
    static double time_resolution=1.0e-9;
    int i,trace_num;
    int *trace_index;
    CKTcircuit *ckt;
    IFvalue valData;

    ckt=run->circuit;
    trace_table = run->circuit->lxt2.kvl_table;
    trace_num = ckt->lxt2.kvl_num;
    trace_index  = ckt->lxt2.kvl_indexmap;

    valData.v.numValue = ckt->CKTmaxEqNum-1;
    valData.v.vec.rVec = ckt->CKTrhsOld+1;

    if(ckt->CKTtime <= 0.0) {
       time_resolution = pow(10.0, LXT2_TIME_RESOLUTION_EXPONENT);
       last_set_time = 0;
       #ifdef LXT2_DEBUG
       printf("LXT2 CKTemitlxt2 set time (%g)  (%d)\n", ckt->CKTtime,last_set_time);
       #endif
       lxt2_wr_set_time(ckt->lxt2.file, last_set_time);
    }
    set_time = (unsigned int)(ckt->CKTtime/time_resolution);
    if(set_time > last_set_time) {
      #ifdef LXT2_DEBUG
      printf("LXT2 CKTemitlxt2 set time (%g)  (%d)\n", ckt->CKTtime,set_time);
      #endif
      lxt2_wr_set_time(ckt->lxt2.file, set_time);
      last_set_time = set_time;
    }
    for(i = 0; i < trace_num; i++) {
       if(trace_index[i] > 0) {
          trace = trace_table[i];
          if(trace!=NULL) {
	      value = valData.v.vec.rVec[run->data[trace_index[i]].outIndex];
 	      name = run->data[trace_index[i]].name;
              #ifdef LXT2_DEBUG
              printf("LXT2 cktlxt2.c:trace(%p) emit double (%s)->(%g)\n",&trace,name,value);
	      #endif
 	      lxt2_wr_emit_value_double(ckt->lxt2.file,trace,0,value);
	  }
       }
    }
    #ifdef LXT2_DEBUG
    printf("LXT2 cktlxt2.c: flush ckt.tracelxt2\n");
    #endif
    lxt2_wr_flush(ckt->lxt2.file);
}
