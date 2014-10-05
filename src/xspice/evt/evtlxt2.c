#include "ngspice/ngspice.h"
#include "ngspice/evtproto.h"

#include <math.h>

static char *EVTbitmap(int i);

/*
EVTemitlxt2


This function emits a evt value into an lxt2 change-on-event file stream. (holmes)

*/

void EVTemitlxt2(
    CKTcircuit    *ckt,        /* The circuit structure */
    int           node_index,  /* The node to copy */
    Evt_Node_t    *from)       /* Location to copy from */
{
    static unsigned int last_set_time;
    static double time_resolution=-1.0;
    unsigned int set_time;
    int type;
    char *name;
    /*  Mif_Boolean_t       invert; */

    Evt_Node_Info_t        **node_table;
    struct lxt2_wr_symbol  **trace_table;
    struct lxt2_wr_symbol  *trace;

    node_table = ckt->evt->info.node_table;
    trace_table = ckt->lxt2.evt_table;

    type = node_table[node_index]->udn_index;
    name = node_table[node_index]->name;

    if(time_resolution<0.0) {
       time_resolution = pow(10.0, LXT2_TIME_RESOLUTION_EXPONENT);
       last_set_time=0;
       #ifdef LXT2_DEBUG
       printf("LXT2 time_resolution (%g) last_time(%d)\n", time_resolution,last_set_time);
       #endif
    }
    set_time = (unsigned int)(ckt->CKTtime/time_resolution);
    if(set_time > last_set_time) {
      #ifdef LXT2_DEBUG
      printf("LXT2 set time (%g)  (%d)\n", ckt->CKTtime,set_time);
      #endif
      lxt2_wr_set_time(ckt->lxt2.file, set_time);
      last_set_time = set_time;
    }

    trace = trace_table[node_index];

    switch(type)
    {
      case 0:	/* Bit */
                #ifdef LXT2_DEBUG
      		printf("LXT2 evtlxt2.c:trace(%p) emit bit (%s)->(%s)\n",trace,name,EVTbitmap(*((int *)(from->node_value))));
		#endif
      		if(lxt2_wr_emit_value_bit_string(ckt->lxt2.file,trace,0,EVTbitmap(*((int *)(from->node_value))))==0) {
		   #ifdef LXT2_DEBUG
		   printf("LXT2 evtlxt2.c:trace(%p) emit bit (%s)->(%s) SUCCESS\n",&trace,name,EVTbitmap(*((int *)(from->node_value))));
		   #endif
		};
		break;
      case 1:	/* Integer */
                #ifdef LXT2_DEBUG
      		printf("LXT2 evtlxt2.c:trace(%p) emit int (%s)->(%d)\n",trace,name,*((int *)(from->node_value)));
		#endif
      		lxt2_wr_emit_value_int(ckt->lxt2.file,trace,0,*((int *)(from->node_value)));
		break;
      case 2:	/* Double */
                #ifdef LXT2_DEBUG
      		printf("LXT2 evtlxt2.c:trace(%p) emit double (%s)->(%g)\n",trace,name,*((double *)(from->node_value)));
		#endif
		lxt2_wr_emit_value_double(ckt->lxt2.file,trace,0,*((double *)(from->node_value)));
		break;
      default:	/* Bit */
                #ifdef LXT2_DEBUG
      		printf("LXT2 evtlxt2.c:trace(%p) emit bit (%s)->(%s)\n",trace,name,EVTbitmap(*((int *)(from->node_value))));
		#endif
      		lxt2_wr_emit_value_bit_string(ckt->lxt2.file,trace,0,EVTbitmap(*((int *)(from->node_value))));
		break;
    }
    #ifdef LXT2_DEBUG
    printf("LXT2 evtlxt2.c: flush ckt.tracelxt2\n");
    #endif
    lxt2_wr_flush(ckt->lxt2.file);

}

static char *EVTbitmap(int i)
{
    static char s[2];
    char *p = s;

    switch(i)
    {
      case 0:
        *(p++) = '0';
	break;
      case 1:
        *(p++) = '1';
	break;
      case 2:
        *(p++) = 'u';
	break;
      case 3:
        *(p++) = 'z';
	break;
      default:
        *(p++) = 'u';
	break;
    }

    *(p) = '\0';
    return(s);
}
