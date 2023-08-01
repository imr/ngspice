/*============================================================================
FILE    EVTprint.c

MEMBER OF process XSPICE

Copyright 1991
Georgia Tech Research Corporation
Atlanta, Georgia 30332
All Rights Reserved

PROJECT A-8503

AUTHORS

    9/12/91  Bill Kuhn

MODIFICATIONS

    7/11/2012  Holger Vogt   Replace printf by out_printf to allow output redirection
    5/21/2017  Holger Vogt   Update 'edisplay': add node type and number of events
    5/26/2018  Uros Platise  Update 'EVTprintvcd': stepsize based on tstep

SUMMARY

    This file contains function EVTprint which is used to provide a simple
    tabular output of event-driven node data.  This printout is invoked
    through a new nutmeg command called 'eprint' which takes event-driven
    node names as argument.

INTERFACES

    void EVTprint(wordlist *wl)

REFERENCED FILES

    None.

NON-STANDARD FEATURES

    None.

============================================================================*/

#include "ngspice/ngspice.h"

#include "ngspice/cpstd.h"
#include "ngspice/cpextern.h"
#include "ngspice/fteext.h"

#include "ngspice/mif.h"
#include "ngspice/evt.h"
#include "ngspice/evtudn.h"

#include "ngspice/evtproto.h"

#include "ngspice/fteext.h"

#include <time.h>
#include <locale.h>


static int get_index(char *node_name);

static void print_data(
    Mif_Boolean_t dcop,
    double        step,
    char          **node_value,
    int           nargs);

/*
EVTprint

This function implements the 'eprint' command used to print
event-driven node values and messages from the latest simulation.

This is a simple prototype implementation of the
eprint command for testing purposes.  It is currently lacking
in the following areas:

1)  It accepts only up to 93 nodes. (EPRINT_MAXARGS)

2)  It does not support the selected printing of different
    members of a user-defined data struct.

3)  It is dumb in its output formatting - just tries to print
    everything on a line with 4 space separators.

4)  It works only for the latest simulation - i.e. it does not
    use the evt jobs structure to find old results.

5)  It does not allow a range of timesteps to be selected.

*/



#define EPRINT_MAXARGS  93


void EVTprint(
    wordlist *wl)    /* The command line entered by user */
{

    int i;
    int nargs;
    int num_ports;

    wordlist    *w;

    char        *node_name[EPRINT_MAXARGS];
    int         node_index[EPRINT_MAXARGS];
    int         udn_index[EPRINT_MAXARGS];
    Evt_Node_t  *node_data[EPRINT_MAXARGS];
    char        *node_value[EPRINT_MAXARGS];

    CKTcircuit  *ckt;

    Evt_Node_Info_t  **node_table;
    Evt_Port_Info_t  **port_table;

    Mif_Boolean_t    more;
    Mif_Boolean_t    dcop;

    double      step = 0.0;
    double      next_step;
    double      this_step;

    char        *value;

    Evt_Msg_t   *msg_data;
    Evt_Statistic_t  *statistics;


    /* Count the number of arguments to the command */
    nargs = 0;
    w = wl;
    while(w) {
        nargs++;
        w = w->wl_next;
    }

    if(nargs < 1) {
        printf("Usage: eprint <node1> <node2> ...\n");
        return;
    }
    if(nargs > EPRINT_MAXARGS) {
        fprintf(cp_err, "ERROR - eprint currently limited to %d arguments\n", EPRINT_MAXARGS);
        return;
    }

    /* Get needed pointers */
    ckt = g_mif_info.ckt;
    if (!ckt) {
        fprintf(cp_err, "Error: no circuit loaded.\n");
        return;
    }
    node_table = ckt->evt->info.node_table;

    /* Get data for each argument */
    w = wl;
    for(i = 0; i < nargs; i++) {
        node_name[i] = w->wl_word;
        node_index[i] = get_index(node_name[i]);
        if(node_index[i] < 0) {
            fprintf(cp_err, "ERROR - Node %s is not an event node.\n", node_name[i]);
            return;
        }
        udn_index[i] = node_table[node_index[i]]->udn_index;
        if (ckt->evt->data.node)
            node_data[i] = ckt->evt->data.node->head[node_index[i]];
        else  {
            fprintf(cp_err, "ERROR - No node data: simulation not yet run?\n");
            return;
        }
        node_value[i] = "";
        w = w->wl_next;
    }

    out_init();

    /* Print results data */
    out_printf("\n**** Results Data ****\n\n");

    /* Print the column identifiers */
    out_printf("Time or Step\n");
    for(i = 0; i < nargs; i++)
        out_printf("%s\n",node_name[i]);
    out_printf("\n\n");

    /* Scan the node data and determine if the first vector */
    /* is for a DCOP analysis or the first step in a swept DC */
    /* or transient analysis.  Also, determine if there is */
    /* more data following it and if so, what the next step */
    /* is. */
    more = MIF_FALSE;
    dcop = MIF_FALSE;
    next_step = 1e30;
    for(i = 0; i < nargs; i++) {
        if(node_data[i]->op)
            dcop = MIF_TRUE;
        else
            step = node_data[i]->step;
        (*(g_evt_udn_info[udn_index[i]]->print_val))
                (node_data[i]->node_value, "all", &value);
        node_value[i] = value;
        node_data[i] = node_data[i]->next;
        if(node_data[i]) {
            more = MIF_TRUE;
            if(node_data[i]->step < next_step)
                next_step = node_data[i]->step;
        }
    }

    /* Print the data */
    print_data(dcop, step, node_value, nargs);

    /* While there is more data, get the next values and print */
    while(more) {

        more = MIF_FALSE;
        this_step = next_step;
        next_step = 1e30;

        for(i = 0; i < nargs; i++) {

            if(node_data[i]) {
                if(node_data[i]->step == this_step) {
                    (*(g_evt_udn_info[udn_index[i]]->print_val))
                            (node_data[i]->node_value, "all", &value);
                    node_value[i] = value;
                    node_data[i] = node_data[i]->next;
                }
                if(node_data[i]) {
                    more = MIF_TRUE;
                    if(node_data[i]->step < next_step)
                        next_step = node_data[i]->step;
                }

            } /* end if node_data not NULL */

        } /* end for number of args */

        print_data(MIF_FALSE, this_step, node_value, nargs);

    } /* end while there is more data */
    out_printf("\n\n");


    /* Print messages for all ports */
    out_printf("\n**** Messages ****\n\n");

    num_ports = ckt->evt->counts.num_ports;
    port_table = ckt->evt->info.port_table;

    for(i = 0; i < num_ports; i++) {

        /* Get pointer to messages for this port */
        msg_data = ckt->evt->data.msg->head[i];

        /* If no messages on this port, skip */
        if(! msg_data)
            continue;

        /* Print the port description */
        out_printf("Node: %s   Inst: %s   Conn: %s   Port: %d\n\n",
                port_table[i]->node_name,
                port_table[i]->inst_name,
                port_table[i]->conn_name,
                port_table[i]->port_num);

        /* Print the messages on this port */
        while(msg_data) {
            if(msg_data->op)
                printf("DCOP            ");
            else
                printf("%-16.9e", msg_data->step);
            printf("%s\n", msg_data->text);
            msg_data = msg_data->next;
        }
        out_printf("\n\n");

    } /* end for number of ports */


    /* Print statistics */
    out_printf("\n**** Statistics ****\n\n");

    statistics = ckt->evt->data.statistics;
    out_printf("Operating point analog/event alternations:  %d\n",
            statistics->op_alternations);
    out_printf("Operating point load calls:                 %d\n",
            statistics->op_load_calls);
    out_printf("Operating point event passes:               %d\n",
            statistics->op_event_passes);
    out_printf("Transient analysis load calls:              %d\n",
            statistics->tran_load_calls);
    out_printf("Transient analysis timestep backups:        %d\n",
            statistics->tran_time_backups);

    out_printf("\n\n");
}




/*
get_index

This function determines the index of a specified event-driven node.
*/


static int get_index(
    char *node_name      /* The name of the node to search for */
)
{

    /* Get the event-driven node index for the specified name */

    int                 index;

    Mif_Boolean_t       found;
    Evt_Node_Info_t     *node;
    CKTcircuit          *ckt;


    /* Scan list of nodes in event structure to see if there */

    found = MIF_FALSE;
    index = 0;

    ckt = g_mif_info.ckt;
    if (!ckt) {
        fprintf(cp_err, "Error: no circuit loaded.\n");
        return(-1);
    }
    node = ckt->evt->info.node_list;

    while(node) {
        if(strcmp(node_name, node->name) == 0) {
            found = MIF_TRUE;
            break;
        }
        else {
            index++;
            node = node->next;
        }
    }

    /* Return the index or -1 if not found */
    if(! found)
        return(-1);
    else
        return(index);
}




/*
print_data

This function prints the values of one or more nodes to
standard output.
*/


static void print_data(
    Mif_Boolean_t dcop,          /* Is this the operating point data */
    double        step,          /* The analysis step if dcop */
    char          **node_value,  /* The array of values to be printed */
    int           nargs)         /* The size of the value array */
{

    int  i, preci;
    char step_str[100];

    /* If option numdgt is set, use it for printout precision. */
    if (cp_numdgt > 0)
        preci = cp_numdgt;
    else
        preci = 9;

    if(dcop)
        strcpy(step_str, "DCOP            ");
    else
        sprintf(step_str, "%.*e", preci, step);

    out_printf("%s", step_str);
    for(i = 0; i < nargs; i++)
        out_printf("    %s", node_value[i]);
    out_printf("\n");
}


/* print all event node names */
void
EVTdisplay(wordlist *wl)
{
    Evt_Node_Info_t  *node;
    CKTcircuit       *ckt;
    int node_index, udn_index;
    Evt_Node_Info_t  **node_table;

    NG_IGNORE(wl);
    ckt = g_mif_info.ckt;
    if (!ckt) {
        fprintf(cp_err, "Error: no circuit loaded.\n");
        return;
    }
    node = ckt->evt->info.node_list;
    node_table = ckt->evt->info.node_table;

    if (!node || !node_table) {
        out_printf("No event node available!\n");
        return;
    }
    out_init();
    if (ckt->evt->jobs.job_plot) {
        out_printf("\nList of event nodes in plot %s\n",
                   ckt->evt->jobs.job_plot[ckt->evt->jobs.cur_job]);
    } else {
        out_printf("\nList of event nodes\n");
    }
    out_printf("    %-20s: %-5s, %s\n\n",
               "node name", "type", "number of events");

    node_index = 0;
    while (node) {
        Evt_Node_t  *node_data = NULL;
        int count = 0;
        char *type;

        udn_index = node_table[node_index]->udn_index;
        if (ckt->evt->data.node) {
            node_data = ckt->evt->data.node->head[node_index];
            while (node_data) {
                count++;
                node_data = node_data->next;
            }
        }
        type = g_evt_udn_info[udn_index]->name;
        out_printf("    %-20s: %-5s, %5d\n", node->name, type, count);

        node = node->next;
        node_index++;
    }
}


/* xspice valid 12-state values (idndig.c):
 *   0s, 1s, Us, 0r, 1r, Ur, 0z, 1z, Uz, 0u, 1u, Uu
 *   0   1   x   0   1   x   0   1   z   0   1   x
 *
 * tentative vcd translation, return value:
 *   0: digital value, 1: real number, 2: unknown
 */

static int
get_vcdval(char *xspiceval, char **newval)
{
    int i, err;
    double retval;

    static char *map[] = {
        "0s", "1s", "Us",
        "0r", "1r", "Ur",
        "0z", "1z", "Uz",
        "0u", "1u", "Uu"
    };

    static char *returnmap[] = {
        "0", "1", "x",
        "0", "1", "x",
        "z", "z", "z",
        "0", "1", "x"
    };

    for (i = 0; i < 12; i++)
        if (eq(xspiceval, map[i])) {
            *newval = copy(returnmap[i]);
            return 0;
        }

    /* is it a real number ? */
    retval = INPevaluate(&xspiceval, &err, 1);
    if (err) {
        *newval = copy(xspiceval); // Assume the node type is coded for this.
        return 2;
    }
    *newval = tprintf("%.16g", retval);
    return 1;
}


#ifdef _MSC_VER
#define time _time64
#define localtime _localtime64
#endif

/* Function to return a real value to be written to a VCD file. */

struct reals {
    struct dvec *time;                          // Scale vector
    int          v_index, last_i;
    double       factor;
    struct dvec *node_vector[EPRINT_MAXARGS];   // For analog nodes
};

static double get_real(int index, double when, struct reals *ctx)
{
    struct dvec *dv;

    if (index < ctx->last_i) {
        /* Starting a new pass. */

        if (!ctx->time) {
            ctx->v_index = 0;
            ctx->time = vec_get("time");
            if (!ctx->time) {
                if (ctx->last_i == EPRINT_MAXARGS) { // First time
                    fprintf(cp_err,
                            "ERROR - No vector 'time' in current plot\n");
                }
                ctx->node_vector[index] = NULL; // No more calls
                return NAN;
            }
        }

        /* Advance the vector index. */

        while (ctx->v_index < ctx->time->v_length &&
               ctx->time->v_realdata[ctx->v_index++] < when) ;
        ctx->v_index--;

        /* Calculate interpolation factor. */

        if (ctx->v_index + 1 < ctx->time->v_length) {
            ctx->factor = (when - ctx->time->v_realdata[ctx->v_index]);
            ctx->factor /= (ctx->time->v_realdata[ctx->v_index + 1] -
                            ctx->time->v_realdata[ctx->v_index]);
            if (ctx->factor < 0.0 || ctx->factor >= 1.0)
                ctx->factor = 0.0; // Rounding
        } else {
            ctx->factor = 0.0;
        }
    }

    /* Return interpolated value. */

    ctx->last_i = index;
    dv = ctx->node_vector[index];
    if (ctx->v_index < dv->v_length) {
        if (ctx->factor == 0.0) {
            return dv->v_realdata[ctx->v_index];
        } else {
            return dv->v_realdata[ctx->v_index] +
                       ctx->factor *
                          (dv->v_realdata[ctx->v_index + 1] -
                           dv->v_realdata[ctx->v_index]);
        }
    } else {
        ctx->node_vector[index] = NULL; // No more calls
        return dv->v_realdata[dv->v_length - 1];
    }
}

/*
 * A simple vcd file printer.
 * command 'eprvcd a0 a1 a2 b0 b1 b2 clk > myvcd.vcd'
 *   prints the event nodes listed to file myvcd.vcd
 *   which then may be viewed with an vcd viewer,
 *     for example 'gtkwave'
 * Still missing:
 *   hierarchy, vector variables
 */

void
EVTprintvcd(wordlist *wl)
{
    int i;
    int nargs;
    int timesteps = 0, tspower = -1;

    wordlist    *w;

    struct reals ctx;

    double out_time, last_out_time;


    char        *node_name[EPRINT_MAXARGS];
    int          node_index[EPRINT_MAXARGS];
    int          udn_index[EPRINT_MAXARGS];
    Evt_Node_t  *node_data[EPRINT_MAXARGS];
    char        *node_value[EPRINT_MAXARGS];
    char        *old_node_value[EPRINT_MAXARGS];
    char         node_ident[EPRINT_MAXARGS + 1];
    char         vbuf[24][2][EPRINT_MAXARGS];   // Analog value strings

    CKTcircuit  *ckt;

    Evt_Node_Info_t  **node_table;

    Mif_Boolean_t    more;

    double      next_step;
    double      this_step;

    char        *value;

    /* Check for the "-a" option (output analog values at timesteps)
     * and "-t nn": specifies the VCD timestep with range 1fs to 1s */

    while (wl && wl->wl_word[0] == '-') {
        if (wl->wl_word[1] == 'a' && !wl->wl_word[2]) {
            timesteps = 1;
        } else if (wl->wl_word[1] == 't' && !wl->wl_word[2]) {
            wl = wl->wl_next;
            if (wl) {
                double input;
                int error = 0;
                char* inword = wl->wl_word;
                input = INPevaluate(&inword, &error, 0);
                tspower = (int)ceil(- 1. * log10(input));
                if (tspower < 0)
                    tspower = 0;
            }
            else
                break;
        } else {
            break;
        }
        wl = wl->wl_next;
    }

    /* Count the number of arguments to the command */
    nargs = 0;
    for (w = wl; w; w = w->wl_next)
        nargs++;

    if (nargs < 1) {
        printf("Usage: eprvcd [-a] <node1> <node2> ...\n");
        return;
    }
    if (nargs > EPRINT_MAXARGS) {
        fprintf(cp_err, "ERROR - eprvcd currently limited to %d arguments\n", EPRINT_MAXARGS);
        return;
    }

    /* Get needed pointers */
    ckt = g_mif_info.ckt;
    if (!ckt) {
        fprintf(cp_err, "Error: no circuit loaded.\n");
        return;
    }
    if (!ckt->evt->data.node) {
      fprintf(cp_err, "ERROR - No node data: simulation not yet run?\n");
      return;
    }
    node_table = ckt->evt->info.node_table;

    /* Get data for each argument */

    w = wl;
    for (i = 0; i < nargs; i++) {
        node_name[i] = w->wl_word;
        node_index[i] = get_index(node_name[i]);

        if (node_index[i] >= 0) {
            udn_index[i] = node_table[node_index[i]]->udn_index;
            node_data[i] = ckt->evt->data.node->head[node_index[i]];
            ctx.node_vector[i] = NULL;
        } else {
            struct pnode *pn;
            struct dvec  *dv;
            wordlist     *save;

            /* Is it an analog parameter/node expression?
             * The whole expression must be a single word (no spaces).
             */

            save = w->wl_next;
            w->wl_next = NULL;
            pn = ft_getpnames_quotes(w, TRUE);
            w->wl_next = save;
            if (pn) {
                dv = ft_evaluate(pn);
                free_pnode(pn);
            } else {
                dv = NULL;
            }
            if (!dv) {
                fprintf(cp_err, "ERROR - Node %s not parsed.\n", node_name[i]);
                return;
            }
            ctx.node_vector[i] = dv;
        }
        node_value[i] = "";
        w = w->wl_next;
    }

    /* generate the vcd identifier code made of the printable
       ASCII character set from ! to ~ (decimal 33 to 126) */
    for (i = 0; i < nargs; i++)
        node_ident[i] = (char) ('!' + i);
    node_ident[i] = '\0';

    out_init();


    /* for gtkwave, avoid e.g. German Umlaute */
    setlocale(LC_TIME, "en_US");

    /* get actual time */
    time_t ltime;
    char datebuff[80];
    struct tm *my_time;

    time(&ltime);
    /* Obtain the local time: */
    my_time = localtime(&ltime);
    /* time output format according to vcd spec */
    strftime(datebuff, sizeof(datebuff), "%B %d, %Y %H:%M:%S", my_time);
    out_printf("$date %s $end\n", datebuff);

    /* return to what it was before */
    setlocale(LC_TIME, "");    

    out_printf("$version %s %s $end\n", ft_sim->simulator, ft_sim->version);

    /* get the sim time resolution based on tstep */
    char *unit;
    double scale, tick;

    if (tspower >= 0) {
        /* VCD timestep set by "-t" option. */

        if (tspower == 0) {
            unit = "s";
            scale = 1.0;
        } else if (tspower < 4) {
            unit = "ms";
            tspower = 3 - tspower;
            scale = 1e3 * pow(10, (double)-tspower);
        } else if (tspower < 7) {
            unit = "us";
            tspower = 6 - tspower;
            scale = 1e6 * pow(10, (double)-tspower);
        } else if (tspower < 10) {
            unit = "ns";
            tspower = 9 - tspower;
            scale = 1e9 * pow(10, (double)-tspower);
        } else if (tspower < 13) {
            unit = "ps";
            tspower = 12 - tspower;
            scale = 1e12 * pow(10, (double)-tspower);
        } else if (tspower < 16) {
            unit = "fs";
            tspower = 15 - tspower;
            scale = 1e15 * pow(10, (double)-tspower);
        } else {  // 1 fS is the bottom.
            unit = "fs";
            tspower = 0;
            scale = 1e15;
        }
        out_printf("$timescale %g %s $end\n", pow(10, (double)tspower), unit);
    } else {
        double tstep = ckt->CKTstep;

        /* Use the simulation time step. If the selected time step
         * is down to [ms] then report time at [us] etc.,
         * always with one level higher resolution.
         */

        if (tstep >= 1e-3) {
            unit = "us";
            scale = 1e6;
        }
        else if (tstep >= 1e-6) {
            unit = "ns";
            scale = 1e9;
        }
        else if (tstep >= 1e-9) {
            unit = "ps";
            scale = 1e12;
        } else {
            unit = "fs";
            scale = 1e15;
        }
        out_printf("$timescale 1 %s $end\n", unit);
    }
    tick = 1.0 / scale;

    /* Scan the node data. Go for printing using $dumpvars
       for the initial values.  Also, determine if there is
       more data following it and if so, what the next step is. */

    ctx.time = NULL;
    ctx.v_index = 0;
    ctx.last_i = EPRINT_MAXARGS;  // Indicate restart
    more = MIF_FALSE;
    next_step = 1e30;
    for (i = 0; i < nargs; i++) {
        if (ctx.node_vector[i]) {
            /* Analog node or expression. */

            sprintf(vbuf[0][i], "%.16g", get_real(i, 0.0, &ctx));
            node_value[i] = vbuf[0][i];
            old_node_value[i] = vbuf[1][i];
            strcpy(vbuf[1][i], vbuf[0][i]);
        } else {
            /* This must return a pointer to a statically-allocated string. */

            g_evt_udn_info[udn_index[i]]->print_val
                (node_data[i]->node_value, "all", &value);
            node_data[i] = node_data[i]->next;
            old_node_value[i] = node_value[i] = value;
            if (node_data[i]) {
                more = MIF_TRUE;
                if (next_step > node_data[i]->step)
                    next_step = node_data[i]->step;
            }
        }
    }

    for (i = 0; i < nargs; i++) {
        char *buf;
        if (get_vcdval(node_value[i], &buf) == 1)
            /* real number format */
            out_printf("$var real 1 %c %s $end\n", node_ident[i], node_name[i]);
        else
            /* digital data format */
            out_printf("$var wire 1 %c %s $end\n", node_ident[i], node_name[i]);
        tfree(buf);
    }

    out_printf("$enddefinitions $end\n");

    /* first set of data for initialization
       or if only op has been calculated */
    out_printf("$dumpvars\n");
    for (i = 0; i < nargs; i++) {
        char *buf;
        if (get_vcdval(node_value[i], &buf) == 1)
            /* real number format */
            out_printf("r%s %c\n", buf, node_ident[i]);
        else
            /* digital data format */
            out_printf("%s%c\n", buf, node_ident[i]);
        tfree(buf);
    }
    out_printf("$end\n");

    /* While there is more data, get the next values and print */

    last_out_time = 0.0;
    while (more ||
           (timesteps && ctx.time && ctx.v_index + 1 < ctx.time->v_length)) {
        int got_one;

        this_step = next_step;

        if (timesteps && ctx.time && ctx.v_index + 1 < ctx.time->v_length &&
            (ctx.time->v_realdata[ctx.v_index + 1] < this_step ||
             (timesteps && !more))) {

            /* Analogue output at each time step, skipping if they would
             * appear simulataneous in the output.
             */

            out_time = ctx.time->v_realdata[ctx.v_index + 1];
            if (out_time - last_out_time < tick) {
                ++ctx.v_index;
                continue;
            }

            for (i = 0; i < nargs; i++) {
                if (ctx.node_vector[i])
                    sprintf(node_value[i], "%.16g",
                            get_real(i, out_time, &ctx));
            }
        } else {
            /* Process next event. */

            out_time = this_step;
            more = MIF_FALSE;
            next_step = 1e30;
            for (i = 0; i < nargs; i++) {
                if (ctx.node_vector[i]) {
                    /* Analog node or expression. */

                    sprintf(node_value[i], "%.16g",
                            get_real(i, this_step, &ctx));
                } else if (node_data[i]) {
                    if (node_data[i]->step == this_step) {
                        g_evt_udn_info[udn_index[i]]->print_val
                            (node_data[i]->node_value, "all", &value);
                        node_value[i] = value;
                        node_data[i] = node_data[i]->next;
                    }
                    if (node_data[i]) {
                        more = MIF_TRUE;
                        if (next_step > node_data[i]->step)
                            next_step = node_data[i]->step;
                    }
                }
            }
        }

        /* print only values that have changed */

        for (i = 0, got_one = 0; i < nargs; i++) {
            if (!eq(old_node_value[i], node_value[i])) {
                char *buf;

                if (!got_one) {
                    /* timestamp */

                    out_printf("#%lld\n",
                               (unsigned long long)(out_time * scale));
                    last_out_time = out_time;;
                    got_one = 1;
                }

                if (get_vcdval(node_value[i], &buf) == 1)
                    out_printf("r%s %c\n", buf, node_ident[i]);
                else
                    out_printf("%s%c\n", buf, node_ident[i]);

                if (ctx.node_vector[i]) {
                    char *t;

                    /* Swap buffers. */

                    t = old_node_value[i];
                    old_node_value[i] = node_value[i];
                    node_value[i] = t;
                } else {;
                    old_node_value[i] = node_value[i];
                }
                tfree(buf);
            }
        }
    } /* end while there is more data */

    out_printf("\n\n");
}

/* Mark event nodes whose data should be saved for printing.
 * By default, all nodes are saved, so initial "none" clears.
 */

static void set_all(CKTcircuit *ckt, Mif_Boolean_t val)
{
    int               i, count;
    Evt_Node_Info_t **node_table;

    count = ckt->evt->counts.num_nodes;
    node_table = ckt->evt->info.node_table;
    if (!node_table)
        return;
    for (i = 0; i < count; i++)
        node_table[i]->save = val;
}

void
EVTsave(wordlist *wl)
{
    int               i;
    wordlist         *w;
    CKTcircuit       *ckt;
    Evt_Node_Info_t **node_table;

    if (wl == NULL) {
        printf("Usage: esave all | none | <node1> <node2> ...\n");
        return;
    }

    /* Get needed pointers. */

    ckt = g_mif_info.ckt;
    if (!ckt) {
        fprintf(cp_err, "Error: no circuit loaded.\n");
        return;
    }

    node_table = ckt->evt->info.node_table;
    if (!node_table)
        return;

    /* Deal with "all" and "none". */

    if (wl->wl_next == NULL) {
        if (!strcmp("none", wl->wl_word)) {
            set_all(ckt, MIF_FALSE);
            return;
        } else if (!strcmp("all", wl->wl_word)) {
            set_all(ckt, MIF_TRUE);
            return;
        }
    }

    set_all(ckt, MIF_FALSE);    /* Clear previous settings. */

    /* Set save flag for each argument */

    for (w = wl; w; w = w->wl_next) {
        i = get_index(w->wl_word);
        if (i < 0) {
            fprintf(cp_err, "ERROR - Node %s is not an event node.\n",
                    w->wl_word);
            return;
        }
        node_table[i]->save = MIF_TRUE;
    }
}
