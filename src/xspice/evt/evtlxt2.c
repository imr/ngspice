#include "ngspice/ngspice.h"
#include "ngspice/evtproto.h"

#include <math.h>

static char *EVTbitmap(int i);


/*
 * This function emits an evt value into an lxt2 change-on-event file stream.
 */

void
EVTemitlxt2(
    CKTcircuit    *ckt,        /* The circuit structure */
    int           node_index,  /* The node to copy */
    Evt_Node_t    *from)       /* Location to copy from */
{
    static unsigned int last_set_time;
    static double time_resolution = -1.0;
    unsigned int set_time;
    int type;
    char *name;

    Evt_Node_Info_t        **node_table;
    struct lxt2_wr_symbol  **trace_table;
    struct lxt2_wr_symbol  *trace;

    node_table = ckt->evt->info.node_table;
    trace_table = ckt->lxt2.evt_table;

    type = node_table[node_index]->udn_index;
    name = node_table[node_index]->name;

    if (time_resolution < 0.0) {
        time_resolution = pow(10.0, LXT2_TIME_RESOLUTION_EXPONENT);
        last_set_time = 0;
    }

    set_time = (unsigned int) (ckt->CKTtime / time_resolution);

    if (set_time > last_set_time) {
        lxt2_wr_set_time(ckt->lxt2.file, set_time);
        last_set_time = set_time;
    }

    trace = trace_table[node_index];

    switch (type)
    {
    case 0:   /* Bit */
        lxt2_wr_emit_value_bit_string(ckt->lxt2.file, trace, 0, EVTbitmap(*((int *)(from->node_value))));
        break;
    case 1:   /* Integer */
        lxt2_wr_emit_value_int(ckt->lxt2.file, trace, 0, *((int *)(from->node_value)));
        break;
    case 2:   /* Double */
        lxt2_wr_emit_value_double(ckt->lxt2.file, trace, 0, *((double *)(from->node_value)));
        break;
    default:  /* Bit */
        lxt2_wr_emit_value_bit_string(ckt->lxt2.file, trace, 0, EVTbitmap(*((int *)(from->node_value))));
        break;
    }

    lxt2_wr_flush(ckt->lxt2.file);
}


static char *
EVTbitmap(int i)
{
    static char s[2];
    char *p = s;

    switch (i)
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

    return s;
}
