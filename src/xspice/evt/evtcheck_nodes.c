/* Automatic insertion of XSPICE devices bridging event and analogue nodes.
 * Giles Atkinson 2022.
 * The contents of this file may belong in src/frontend, except that it is
 * part of XSPICE.
 */

#include <ctype.h>
#include <errno.h>

#include "ngspice/ngspice.h"
#include "ngspice/cktdefs.h"
#include "ngspice/devdefs.h"
#include "ngspice/inpdefs.h"
#include "ngspice/sperror.h"
#include "ngspice/wordlist.h"
#include "ngspice/fteext.h"

#include "ngspice/mif.h"
#include "ngspice/evt.h"
#include "ngspice/evtudn.h"
#include "ngspice/evtproto.h"
#include "ngspice/mif.h"
#include "../../frontend/inpcom.h"
#include "../../frontend/inp.h"
#include "../../frontend/subckt.h"
#include "../../frontend/variable.h"
#include "../../frontend/numparam/numpaif.h"
#include "../../misc/util.h"

/* Automatic insertion of bridge devices.

When using XSPICE to simulate digital devices in a mixed-mode simulation,
bridging devices must be included in the circuit to pass signals between the
analogue simulator and XSPICE.  Such devices may be included in the netlist,
or they may can be inserted automatically.  Different types of automatic
bridge may exist in the same circuit, depending on signal direction and
characteristics of the connected device, for example digital devices
powered by differing voltages.  Non-digital XSPICE nodes are supported.

Bridging may be disabled by setting command variable "auto_bridge" to zero.
Values greater than one enable diagnostic outputs.

Automatic insertion works by checking node names; if the same name occurs in
both sides of the simulator, it indicates that a connection between
an analogue and a digital device has been specified.  At this point no such
connection exists: it is created by adding the bridging device.  Devices are
added by generating netlist lines describing them and then performing some
stages of ngspice's normal input processing.

To add a bridging device, one or two lines of netlist text are created.
Typically, one specifies the device itself (an XSPICE 'A' "device card")
and the other ("setup card") is the required model definition.  Alternatively,
the setup card may use .include to read in a subcircuit definition and the
device card is a subcircuit ('X' card).   Setup cards are processed once,
device cards, once per bridged node.

Inserted or included netlist lines use a restricted set of features.  There
may be further includes and subcircuits, but nothing else apart from device
and model cards has been tested.  Parameter and expression processing are
available, but parameters defined in the main circuit are inaccessible.
Models defined in the circuit may be used, but subcircuits are inaccessible.
These restictions arise from the fact that missing bridges are detected
after the circuit has been built.  Previous parameter and subcircuit
definitions are no longer fully available.

The device card must be a format string for the C library function sprintf()
with one %d and two %s insertion indicators.  The %d insertion makes the
device name unique and the string insertions supply a list of node names
that will be bridged.  Optionally a final substitution for a double may be
included that will receive the value of internal variable, vcc (see below).
The results of formatting should be an XSPICE device card or a subcircuit call.

The setup and device cards to be used are determined by examining the
event node that is to be connected.  The rules are a little complex,
but are intended to be very flexible, while easy to use in common
cases.   If all XSPICE nodes are digital, using 3.3V supply and good
simulation of the device's electrical characteristics is not needed,
then nothing need be done.  Other supply voltages can be handled by:

.param vcc=5

or similar, and the parameter may have different values in subcircuits.
Otherwise, the full rules follow.  The third rule is intended to be useful
for device libraries whose devices are defined by subcircuits.

1: The program looks for a command interpreter variable
   "auto_bridge_parm_TTTT" where TTTT is the node type string.  So
   "auto_bridge_parm_d" for digital nodes.  The value should be the name of
   a circuit parameter (.param card).  If not found, ignore the next step,
   except in the case of a digital node where the parameter name is assumed
   to be "vcc". Initialise an internal variable, vcc, to zero.

2: Given a parameter name, find the most deeply-nested (in subcircuits) XSPICE
   device that is connected to the node.  Search upwards through the enclosing
   subcircuits for a definition of the parameter.  If found, set vcc from
   the parameter.

3: If a command interpreter variable "no_auto_bridge_family": exists, go to
   step 5.  Search the connected device instances for a parameter, "Family"
   with a string value.  The first one found will be used.  If the first
   character of the value is not '*', the setup card will be ".include
   FFFFF_DDD_bridge.cir" where FFFFF is the family and DDD is the signal
   direction for the XSPICE device: "in", "out" or "inout".  The device card
   will be "Xauto_bridge%d %s %s bridge_FFFFF_DDD vcc=%g", so a suitably
   parameterised subcircuit must be defined in the file.

4: If the first character was '*', look for a variable
   "auto_bridge_FFFFF_TTTT_DDD" where FFFFF is the family, TTTT is the node
   type string and DDD is the direction.  So this might be
   "auto_bridge_74HCT_d_inout" for a digital node.  Use the variable's value
   in step 6.

5: Look for a variable "auto_bridge_TTTT_DDD_NNNN" where TTTT and DDD are as
   before and NNNN is the integer part of (vcc * 10).

6: Check and use the variable's value from step 2 or 5.  It must be a list
   of two or three elements: the setup card, device card and an optional
   third value.  If present, the third value in the list must be an integer,
   specifying the maximum number of nodes handled by a single device.  If
   not present, or zero, there is no limit.  If no variable was found, the
   circuit fails, except for digital nodes where defaults are
   supplied. Example:

   set auto_bridge_real_out_0 = ( ".model real_output_bridge d_to_real"
                                 "areal_bridge%d [ %s ] null [%s] delay=1e-6" )

   (This would be on a single line and the spaces are important.)

   For digital the defaults are:

   ( ".model auto_adc adc_bridge(in_low = %g in_high = %g)"
     "auto_adc%d [ %s ] [ %s ] auto_adc" )

   and

   ( ".model auto_dac dac_bridge(out_low = 0 out_high = %g)"
     "auto_dac%d [ %s ] [ %s ] auto_dac" )

  with vcc/2.0 or vcc substituted for %g as appropriate.

  An example of an included subcircuit might be:

  * Subcircuit for digital output buffer with impedance

  .subckt auto_buf dig ana vcc=5
  .model auto_dac dac_bridge(out_low = 0 out_high = {vcc})
  auto_dac [ dig ] [ internal ] auto_dac
  rint internal ana 100
  .ends

   and invoked by:

   set auto_bridge_d_out_30 = ( ".include test_sub.cir"
                                "xauto_buf%d %s %s auto_buf vcc=%g"
                                1 )
*/

/* Working information about a type of bridge. */

struct bridge {
    int             udn_index;          // Node type.
    Mif_Dir_t       direction;          // IN, OUT or INOUT
    double          vcc;                // A .param value used for matching.
    const char     *family;             // Alternative match parameter.
    struct bridge  *next;               // Linked list
    int             max, count;         // How many nodes/device?
    char           *setup;              // For model or inclusion card.
    const char     *format;             // For device card.
    int             end_index;          // Free buffer space
    char            held[256];          // Buffer with node names.
};

/* Enable/verbosity level */

#define AB_OFF    0
#define AB_SILENT 1
#define AB_DECK   2 // Show generated deck

#define BIG 999990000 // For fake linenumbers

/* Expand .include cards and subcircuits in a deck. */

static struct card *expand_deck(struct card *head)
{
    struct card  *card, *next;
    char        **pointers;
    int           i, dico;

    /* Save the current parameter symbol table. */

    dico = nupa_add_dicoslist();

    /* Count the cards, allocate and fill a pointer array. */

    for (i = 0, card = head; card; ++i, card = card->nextcard)
        ;
    pointers = TMALLOC(char *, i + 1);
    for (i = 0, card = head; card; ++i, card = card->nextcard)
        pointers[i] = card->line;
    pointers[i] = NULL;

    /* Free the card headers. */

    for (card = head; card; card = next) {
        next = card->nextcard;
        tfree(card);
    }

    /* The cards are passed to _inp_readall() via a global. */

    circarray = pointers;
    card = inp_readall(NULL, Infile_Path, FALSE, TRUE, NULL);
    card = inp_subcktexpand(card);

    /* Destroy the parameter table that was created in subcircuit/parameter
     * expansion and restore the previous version.
     */

    nupa_del_dicoS();
    nupa_set_dicoslist(dico);
    nupa_rem_dicoslist(dico);
    return card;
}

/* Write a completed bridge instance card. */

static struct card *flush_card(struct bridge *bridge, int ln,
                               struct card *last)
{
    char         buff[2 * sizeof bridge->held + 128];

    bridge->held[bridge->end_index] = '\0';
    snprintf(buff, sizeof buff, bridge->format,
             ln, bridge->held, bridge->held, bridge->vcc);
    bridge->count = 0;
    bridge->end_index = 0;
    return insert_new_line(last, copy(buff), BIG + ln, 0);
}


/* Release memory used. */

static void free_bridges(struct bridge *bridge_list)
{
    struct bridge *bridge;

    while (bridge_list) {
        /* Free the structure.  Setup has gone. */

        bridge = bridge_list;
        bridge_list = bridge->next;
        if (bridge->format)
            tfree(bridge->format);
        tfree(bridge);
    }
}

/* Find an XSPICE device instance given its index number.  Uses link chasing
 * as at this point the indexable inst_table does not yet exist.
 */

static MIFinstance *find_inst(CKTcircuit *ckt, int index)
{
    struct Evt_Inst_Info *chase;
    int                   i;

    for (i = 0, chase = ckt->evt->info.inst_list;
         i < index && chase;
         ++i, chase = chase->next)
        ;
    return chase ? chase->inst_ptr : NULL;
}

/* All ports are declared as outputs, but some may be INOUT. */

static Mif_Dir_t scan_ports(Evt_Node_Info_t *event_node, CKTcircuit *ckt)
{
    Evt_Inst_Index_t  *inst_list;
    Evt_Inst_Info_t  **inst_table;

    inst_table = ckt->evt->info.inst_table;
    for (inst_list = event_node->inst_list;
         inst_list;
         inst_list = inst_list->next) {
        MIFinstance     *inst;
        int              i;

        inst = find_inst(ckt, inst_list->index);
        for (i = 0; i < inst->num_conn; ++i) {
            Mif_Conn_Data_t *conn;
            int              j;

            conn = inst->conn[i];
            if (conn->is_null || !conn->is_input)
                continue;
            for (j = 0; j < conn->size; ++j) {
                Mif_Port_Data_t  *port;

                port = conn->port[j];
                if (!strcmp(port->pos_node_str, event_node->name) ||
                    !strcmp(port->neg_node_str, event_node->name)) {
                    /* An inout connection to this node. */

                    return MIF_INOUT;
                }
            }
        }
    }
    return MIF_OUT;
}

/* Examine a device and return its subcircuit nesting depth
 * and possibly the value of its "family" parameter, if that exists.
 */

static int examine_device(MIFinstance *inst, const char **family)
{
    struct MIFmodel *mp;
    char            *dot;
    int              i;

     if (!*family) {
         Mif_Param_Data_t *pdp;
         IFparm           *pp;
         int               type;

         mp = MIFmodPtr(inst);
         type = mp->MIFmodType;
         pp = ft_sim->devices[type]->modelParms;
         for (i = 0; i < inst->num_param; ++i) {
             pdp = mp->param[i];
             if (!pdp->is_null && pdp->eltype == IF_STRING &&
                 !strcmp(pp[i].keyword, "family")) {
                 /* Return value for "family" parameter. */

                 *family = pdp->element->svalue; // May be NULL
                 if (family && !family[0])
                     family = NULL;     // Ignore empty string.
                 break;
             }
         }
     }
 
     for (i = 0, dot = inst->MIFname;
          (dot = strchr(dot, '.'));
          dot += 1, ++i)
         ;
     return i;
}

/* Scan devices attached to node.  Return the name of the deepest-nested
 * one and the first "Family" parameter found.
 */

static const char *scan_devices(Evt_Node_Info_t   *event_node,
                                CKTcircuit        *ckt,
                                const char       **family)
{
    Evt_Inst_Index_t   *inst_list;
    MIFinstance        *best_inst = NULL, *inst;
    int                 best_depth = -1, depth, left;

    /* Scan input connections. */

    for (inst_list = event_node->inst_list;
         inst_list;
         inst_list = inst_list->next) {
        inst = find_inst(ckt, inst_list->index);
        depth = examine_device(inst, family);
        if (depth > best_depth) {
            best_depth = depth;
            best_inst = inst;
        }
    }

    if ((left = event_node->num_outputs)) {
        Evt_Node_Info_t    *node;
        Evt_Output_Info_t  *oip;
        int                 i, my_index;

        /* Find the index of this node (not stored). */

        for (my_index = 0, node = ckt->evt->info.node_list;
             node != event_node;
             ++my_index, node = node->next)
            ;

        /* Scan output connections. */

        for (i = 0, oip = ckt->evt->info.output_list;
             oip;
             ++i, oip = oip->next) {
            if (oip->node_index == my_index) {
                inst = find_inst(ckt, oip->inst_index);
                depth = examine_device(inst, family);
                if (depth > best_depth) {
                    best_depth = depth;
                    best_inst = inst;
                }
                if (--left == 0)
                    break;
            }
        }
    }
    return best_inst ? best_inst->MIFname : NULL;
}

/* Can a bridge element be inserted? */

static struct bridge *find_bridge(Evt_Node_Info_t  *event_node,
                                  CKTcircuit       *ckt,
                                  struct bridge   **bridge_list_p)
{
    static const char * const   dirs[] = {"in", "out", "inout"};
    struct bridge              *bridge;
    Mif_Dir_t                   direction;
    const char                 *format = NULL;
    const char                 *type_name, *family, *s_family, *deep;
    char                       *setup, *vcc_parm, *dot;
    double                      vcc;
    int                         max = 0, ok;
    struct variable            *cvar = NULL;
    char                        buff[256];

    /* Find the direction for this node. */

    if (event_node->num_outputs == 0) {
        direction = MIF_IN;
    } else if (event_node->num_outputs  < event_node->num_ports) {
        direction = MIF_INOUT;
    } else {
        direction = scan_ports(event_node, ckt); // Ugly
    }

    /* Find the vcc parameter for this node. */

    type_name = g_evt_udn_info[event_node->udn_index]->name;
    snprintf(buff, sizeof buff, "auto_bridge_parm_%s", type_name);
    if (cp_getvar(buff, CP_STRING, buff, sizeof buff))
        vcc_parm = buff;
    else if (event_node->udn_index == 0)
        vcc_parm = "vcc";
    else
        vcc_parm = NULL;

    /* Scan attached XSPICE devices for the deepest nested one and
     * a device with a "family" parameter.
     */

    family = NULL;
    deep = scan_devices(event_node, ckt, &family);
    if (family && cp_getvar("no_auto_bridge_family", CP_BOOL, NULL, 0))
        family = NULL;

    /* Look for a real parameter (.param type) in the device's subcircuit,
     * and those enclosing it.
     */

    snprintf(buff, sizeof buff, "%s", deep);
    while ((dot = strrchr(buff, '.'))) {
        snprintf(dot + 1, sizeof buff - (size_t)(dot - buff), vcc_parm);
        vcc = nupa_get_param(buff, &ok);
        if (ok)
            break;
        *dot = '\0';
    }
    if (dot == NULL) {
        vcc = nupa_get_param(vcc_parm, &ok);
        if (!ok && event_node->udn_index == 0) // Fallback default for digital.
            vcc = 3.3;
    } else {
        vcc = 0;
    }

    if (family && *family == '*') {
        s_family = family + 1;
        family = NULL; // Not used for matching.
    } else {
        s_family = NULL;
    }

    /* Look for an existing entry. */

    for (bridge = *bridge_list_p; bridge; bridge = bridge->next) {
        if (bridge->udn_index == event_node->udn_index &&
            bridge->direction == direction) {
            if (family) {
                if (!strcmp(family, bridge->family)) {
                    /* Return vcc for formatting: requires bridge->max == 1. */

                    bridge->vcc = vcc;
                    break;
                }
            } else if (bridge->vcc == vcc) {            // Match vcc.
                break;
            }
        }
    }
    if (bridge)
        return bridge;

    /* Determine if a bridging element exists, starting with the node type. */

    if (family) {
        /* Use standard pattern for known parts family. */

        snprintf(buff, sizeof buff, ".include %s_%s_bridge.cir",
                 family, dirs[direction]);
        setup = copy(buff);
        snprintf(buff, sizeof buff,
                 "Xauto_bridge%%d %%s %%s bridge_%s_%s vcc=%%g",
                 family, dirs[direction]);
        format = copy(buff);
        max = 1;
    } else if (s_family) {
        /* Family variable lookup. */

        snprintf(buff, sizeof buff, "auto_bridge_%s_%s_%s",
                 s_family, type_name, dirs[direction]);
        cp_getvar(buff, CP_LIST, &cvar, sizeof cvar);
    }

    if (!format && !cvar) {
        /* General variable lookup. */

        snprintf(buff, sizeof buff, "auto_bridge_%s_%s_%d",
                 type_name, dirs[direction], (int)(vcc * 10));
        cp_getvar(buff, CP_LIST, &cvar, sizeof cvar);
    }
    if (!format && cvar) {
        struct variable *v1, *v2, *v3;

        v1 = cvar->va_vlist;
        if (v1 && v1->va_type == CP_STRING) {
            v2 = v1->va_next;
            if (v2->va_type == CP_STRING &&
                v2->va_string && v2->va_string[0]) {
                format = copy(v2->va_string);
                setup = copy(v1->va_string);

                v3 = v2->va_next;
                if (v3 && v3->va_type == CP_NUM)
                    max = v3->va_num;
            } else {
                cvar = NULL;
            }
        } else {
                cvar = NULL;
        }
    }

    /* Last and probably most common case, default digital bridges. */

    if (!format && event_node->udn_index == 0) {
        if (direction == MIF_INOUT) {
            return NULL; // Abandon hope, for now.
        } else {
            if (direction == MIF_IN) {
                snprintf(buff, sizeof buff,
                        ".model auto_adc adc_bridge(in_low = %g in_high = %g)",
                        vcc/2.0, vcc/2.0);
                format = copy("auto_adc%d [ %s ] [ %s ] auto_adc");
            } else {    // MIF_OUT
#if 0
                snprintf(buff, sizeof buff, ".include test_sub.cir");
                format = copy("xauto_buf%d %s %s auto_buf vcc=%g");
                max = 1;
#else
                snprintf(buff, sizeof buff,
                        ".model auto_dac dac_bridge"
                        "(out_low = 0 out_high = %g)",
                        vcc);
                format = copy("auto_dac%d [ %s ] [ %s ] auto_dac");
#endif
            }
            setup = copy(buff);
        }
    }
    if (!format)
        return NULL;

    /* Make a new bridge structure. */

    bridge = TMALLOC(struct bridge, 1);
    bridge->udn_index = event_node->udn_index;
    bridge->direction = direction;
    bridge->vcc = vcc;
    bridge->family = family;
    bridge->next = *bridge_list_p;
    *bridge_list_p = bridge;
    bridge->max = max;
    bridge->count = 0;
    bridge->setup = setup;
    bridge->format = format;
    bridge->end_index = 0;
    return bridge;
}

/* Early detection of node type clashes and attempted fix by
 * automatic insertion of a bridging device.
 */

bool Evtcheck_nodes(
    CKTcircuit  *ckt,             /* The circuit structure */
    INPtables   *stab)            /* Symbol table. */
{
    struct bridge       *bridge_list = NULL, *bridge;
    CKTnode             *analog_node;
    Evt_Node_Info_t     *event_node;
    struct card         *head = NULL, *lastcard = NULL, *card;
    int                  ln = 0, do_expand = 0, show;

    /* Is auto-bridge enabled? */

    if (!cp_getvar("auto_bridge", CP_NUM, &show, sizeof show))
        show = AB_SILENT;

    /* Try to create joining device if any analog node name matches
     * an event node. Failure is fatal.
     */

    for (event_node = ckt->evt->info.node_list;
         event_node;
         event_node = event_node->next) {
         for (analog_node = ckt->CKTnodes;
             analog_node;
             analog_node = analog_node->next) {
             int nl;

             if (strcmp(event_node->name, analog_node->name) == 0) {
                 if (show == AB_OFF)
                     return FALSE;      // Auto-bridge disabled
                 bridge = find_bridge(event_node, ckt, &bridge_list);
                 if (!bridge) {
                    /* Fatal, circuit cannot run. */

                    errMsg = tprintf("Can not insert bridge for mixed-type "
                                     "node %s\n",
                                     analog_node->name);
                    free_bridges(bridge_list);
                    if (head)
                        line_free(head, TRUE);
                    return FALSE;
                }

                if (bridge->setup) {
                    /* Model/include card for bridge.
                     * bridge->setup must be dynamic (for tfree()).
                     */

                    if (!strncmp(".inc", bridge->setup, 4))
                        do_expand = 1;
                    card = insert_new_line(lastcard, bridge->setup,
                                           BIG + ln++, 0);
                    if (!lastcard)
                        head = card;
                    lastcard = card;
                    bridge->setup = NULL;       // Output just once.
                }

                /* Card buffer full? */

                nl = (int)strlen(analog_node->name);
                if ((bridge->max && bridge->count >= bridge->max) ||
                    bridge->end_index + nl + 2 > (int)(sizeof bridge->held)) {
                    /* Buffer full, flush card. */

                    card = flush_card(bridge, ln++, lastcard);
                    if (!lastcard)
                        head = card;
                    lastcard = card;
                }

                /* Copy node name to card contents buffer. */

                bridge->count++;
                if (bridge->end_index)
                    bridge->held[bridge->end_index++] = ' ';
                strcpy(bridge->held + bridge->end_index, analog_node->name);
                bridge->end_index += nl;
            }
        }
    }

    /* Flush cards. */

    for (bridge = bridge_list; bridge; bridge = bridge->next) {
        if (bridge->end_index > 0) {
            card = flush_card(bridge, ln++, lastcard);
            if (!lastcard)
                head = card;
            lastcard = card;
        }
    }

    if (!lastcard)
        return TRUE;  // No success, but also no failures - nothing to bridge.

    if (show >= AB_DECK) {
        for (card = head; card; card = card->nextcard)
            printf("%d: %s\n", card->linenum, card->line);
    }

    /* If there are any .include cards, expand them. */

    if (do_expand) {
        head = expand_deck(head);
        if (!head)
            return FALSE;
    }

    /* Push the cards into the circuit. */

    INPpas1(ckt, head, stab);
    INPpas2(ckt, head, stab, ft_curckt->ci_defTask);

    /* Store them so that they show in "listing e". */

    ft_curckt->ci_auto = head;

    free_bridges(bridge_list);
    return TRUE;
}
