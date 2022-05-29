/*
    udevices.c translate PSPICE U* instances and timing models.

    Notes: To translate Pspice U* devices in a subcircuit containing
    U* instance and Pspice .model cards, two passes through the subcircuit
    are necessary. The first pass is to translate the timing models from
    the .model cards. This timing delay information is stored. The second
    pass is for translating the U* instance cards to generate equivalent
    Xspice digital device instances and their timing delay .model cards
    using the previously stored delays.

    Some limitations are:
        No support for logicexp, pindly, and constraint behavioral primitives.
        Approximations to the Pspice timing delays. Typical values for delays
        are estimated. Pspice has a rich set of timing simulation features,
        such as checks for setup/hold violations, minimum pulse width, and
        hazard detection.
        Only the common logic gates, flip-flops, and latches are supported.

   First pass through a subcircuit. Call initialize_udevice() and read the
   .model cards by calling u_process_model_line() (or similar) for each card,
   The delays for the different types (ugate, utgate, ueff, ugff) are stored
   by get_delays_...() and add_delays_to_model_xlator(). Also, during the
   first pass, check that each U* instance can be translated to Xspice.
   If there are any .model or U* instance cards that cannot be processed
   in the .subckt, then there is no second pass and the .subckt is skipped.

   Second pass through a subcircuit. To translate each U* instance call
   u_process_instance_line() (or similar). This calls translate_...()
   functions for gates, tristate, flip-flops and latches. translate_...()
   calls add_..._inout_timing_model() to parse the U* card, and then calls
   gen_..._instance(). Creating new cards to replace the U* and .model
   cards is done by calling replacement_udevice_cards(), which returns a
   list of new cards. The list of cards is then to be inserted after the
   .subckt card and before the .ends card.  This occurs in the driver
   function u_instances() in frontend/inpcom.c.
   Finally, call cleanup_udevice() before repeating the sequence for
   another subcircuit.

   More explanations are provided below in comments with NOTE.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <ctype.h>
#include "ngspice/ngspice.h"
#include "ngspice/memory.h"
#include "ngspice/bool.h"
#include "ngspice/stringskip.h"
#include "ngspice/stringutil.h"
#include "ngspice/inpdefs.h"
#include "ngspice/cpextern.h"
#include "ngspice/macros.h"
#include "ngspice/udevices.h"

extern struct card* insert_new_line(
    struct card* card, char* line, int linenum, int linenum_orig);

static void print_all_pin_lists(BOOL just_ports);

/* #define TRACE */

/* device types */
#define D_AND    0
#define D_AO     1
#define D_AOI    2
#define D_BUF    3
#define D_INV    4
#define D_NAND   5
#define D_NOR    6
#define D_NXOR   7
#define D_OA     8
#define D_OAI    9
#define D_OR     10
#define D_XOR    11
#define D_DFF    12
#define D_JKFF   13
#define D_DLTCH  14
#define D_SRFF   15
#define D_UP     16
#define D_DOWN   17
#define D_TRI    18
#define XSPICESZ  19

/* structs for parsed gate U... instances */
struct instance_hdr {
    char *instance_name;
    char *instance_type;
    int num1;
    int num2;
};

struct gate_instance {
    struct instance_hdr *hdrp;
    int num_gates;
    int width;
    int num_ins;
    char **inputs;
    char *enable;
    int num_outs;
    char **outputs;
    char *tmodel;
};

struct compound_instance {
    struct instance_hdr *hdrp;
    int num_gates;
    int width;
    int num_ins;
    char **inputs;
    char *output;
    char *tmodel;
};

struct dff_instance {
    struct instance_hdr *hdrp;
    char *prebar;
    char *clrbar;
    char *clk;
    int num_gates;
    char **d_in;
    char **q_out;
    char **qb_out;
    char *tmodel;
};

struct jkff_instance {
    struct instance_hdr *hdrp;
    char *prebar;
    char *clrbar;
    char *clkbar;
    int num_gates;
    char **j_in;
    char **k_in;
    char **q_out;
    char **qb_out;
    char *tmodel;
};

struct dltch_instance {
    struct instance_hdr *hdrp;
    char *prebar;
    char *clrbar;
    char *gate;
    int num_gates;
    char **d_in;
    char **q_out;
    char **qb_out;
    char *tmodel;
};

/* structs for instances and timing models which have been translated */
typedef struct s_xlate *Xlatep;
typedef struct s_xlate {
    Xlatep next;
    char *translated;  // the translated instance line
    char *delays;      // the delays from the pspice timing model
    char *utype;       // pspice model type ugate, utgate, ueff, ugff
    char *xspice;      // xspice device type such as d_and, d_dff, etc.
    char *tmodel;      // timing model name of pspice instance or model
    char *mname;       // name of the xspice timing model of the instance
} Xlate;

typedef struct s_xlator *Xlatorp;
typedef struct s_xlator {
    Xlatep head;
    Xlatep tail;
    Xlatep iter;
} Xlator;

#ifdef TRACE
static void interpret_xlator(Xlatorp xp, BOOL brief);
#endif

/* For timing model extraction */
#define EST_UNK -1
#define EST_MIN 0
#define EST_TYP 1
#define EST_MAX 2
#define EST_AVE 3

struct timing_data {
    char *min;
    char *typ;
    char *max;
    char *ave;
    int estimate;
};

/* NOTE
  Pin lists contain the names of instance inputs, outputs, tristate outputs.
  These pin_lists are created by add_..._pin() calls within the various
  gen_..._instance() calls. When a .subckt ... .ends sequence is completed,
*/
#define DIR_UNKNOWN  0
#define DIR_IN       1
#define DIR_OUT      2
#define DIR_TRI      3
#define DIR_INOUT    4

struct pin_entry {
    char *name;
    struct pin_entry *next;
    int pin_type;
};

/*
  static data cleared and reset by initialize_udevice(),
  cleared by cleanup_udevice().
*/
static char *subckt_saved = NULL;
static int pins_and_ports = 0; // If non-zero then generate pins and ports
static struct pin_entry *subckt_ports = NULL;
static struct pin_entry *input_pins = NULL;
static struct pin_entry *output_pins = NULL;
static struct pin_entry *tristate_pins = NULL;
/* .model d_zero_inv99 d_inverter just once per subckt */
static BOOL add_zero_delay_inverter_model = FALSE;

static struct pin_entry *add_pin(char *name, int pin_type, BOOL is_port)
{
    struct pin_entry **plist = NULL, *x = NULL, *last = NULL;
    char *tmp;
    size_t sz;

    if (!pins_and_ports || !subckt_saved) {
        /* subckt_saved is necessary for the port list */
        return NULL;
    }
    if (is_port) {
        plist = &subckt_ports;
    } else {
        switch (pin_type) {
        case DIR_IN:
            plist = &input_pins;
            break;
        case DIR_OUT:
            plist = &output_pins;
            break;
        case DIR_TRI:
            plist = &tristate_pins;
            break;
        default:
            return NULL;
        }
    }
    if (!(*plist)) {
        *plist = TMALLOC(struct pin_entry, 1);
        sz = strlen(name) + 1;
        tmp = TMALLOC(char, sz);
        (void) memcpy(tmp, name, sz);
        (*plist)->name = tmp;
        (*plist)->next = NULL;
        (*plist)->pin_type = pin_type;
        return *plist;
    }
    for (x = *plist; x; x = x->next) {
        /* No duplicates */
        if (eq(x->name, name)) {
            return *plist;
        }
        last = x;
    }
    x = TMALLOC(struct pin_entry, 1);
    sz = strlen(name) + 1;
    tmp = TMALLOC(char, sz);
    (void) memcpy(tmp, name, sz);
    x->name = tmp;
    x->next = NULL;
    x->pin_type = pin_type;
    last->next = x;
    return *plist;
}

static struct pin_entry *find_pin(char *name, int pin_type, BOOL is_port)
{
    struct pin_entry *plist = NULL, *x = NULL;
    if (is_port) {
        plist = subckt_ports;
    } else {
        switch (pin_type) {
        case DIR_IN:
            plist = input_pins;
            break;
        case DIR_OUT:
            plist = output_pins;
            break;
        case DIR_TRI:
            plist = tristate_pins;
            break;
        default:
            return NULL;
        }
    }
    if (!plist) {
        return NULL;
    }
    for (x = plist; x; x = x->next) {
        /* No duplicates */
        if (eq(x->name, name)) {
            return x;
        }
    }
    return NULL;
}

static void add_input_pin(char *name)
{
    if (pins_and_ports && strncmp(name, "$d_", 3) != 0) {
        (void) add_pin(name, DIR_IN, FALSE);
    }
}

static void add_output_pin(char *name)
{
    if (pins_and_ports && strncmp(name, "$d_", 3) != 0) {
        (void) add_pin(name, DIR_OUT, FALSE);
    }
}

static void add_tristate_pin(char *name)
{
    if (pins_and_ports && strncmp(name, "$d_", 3) != 0) {
        (void) add_pin(name, DIR_TRI, FALSE);
    }
}

static void add_port(char *name, int pin_type)
{
    if (pins_and_ports) {
        add_pin(name, pin_type, TRUE);
    }
}

static struct pin_entry *find_input_pin(char *name)
{
    return find_pin(name, DIR_IN, FALSE);
}

static struct pin_entry *find_output_pin(char *name)
{
    return find_pin(name, DIR_OUT, FALSE);
}

static struct pin_entry *find_tristate_pin(char *name)
{
    return find_pin(name, DIR_TRI, FALSE);
}

static void clear_pin_list(struct pin_entry *plist)
{
    struct pin_entry *x = NULL, *next = NULL;
    if (!plist) { return; }
    for (x = plist; x; x = next) {
        next = x->next;
        if (x->name) { tfree(x->name); }
        tfree(x);
    }
}

static void clear_all_pin_lists(void)
{
    print_all_pin_lists(TRUE);

    clear_pin_list(input_pins);
    clear_pin_list(output_pins);
    clear_pin_list(tristate_pins);
    clear_pin_list(subckt_ports);
    input_pins = NULL;
    output_pins = NULL;
    tristate_pins = NULL;
    subckt_ports = NULL;
}

static void print_pin_list(struct pin_entry *plist)
{
    struct pin_entry *x = NULL;
    char *type, *porp;
    if (!plist) { return; }
    if (plist == subckt_ports) {
        porp = "port";
    } else {
        porp = "pin";
    }
    for (x = plist; x; x = x->next) {
        switch (x->pin_type) {
        case DIR_UNKNOWN:
            type = "unk";
            break;
        case DIR_IN:
            type = "in";
            break;
        case DIR_OUT:
            type = "out";
            break;
        case DIR_TRI:
            type = "tri";
            break;
        case DIR_INOUT:
            type = "inout";
            break;
        default:
            type = "???";
            break;
        }
        printf("%s: %s %s\n", porp, x->name, type);
    }
}

static void print_all_pin_lists(BOOL just_ports)
{
    if (!pins_and_ports) {
        return;
    }
    if (subckt_saved) {
        printf("%s\n", subckt_saved);
    }
    if (!just_ports) {
        print_pin_list(input_pins);
        print_pin_list(output_pins);
        print_pin_list(tristate_pins);
    }
    print_pin_list(subckt_ports);
}

/*
  create_ports_list() can be called to determine the IN/OUT/INOUT directions
  of the subckt ports.
*/
static void create_ports_list(void)
{
    char *copy_line, *tok, *pos;
    BOOL inp = FALSE, outp = FALSE, tri = FALSE;
    int port_type;

    if (!pins_and_ports || !subckt_saved) {
        return;
    }
    copy_line = tprintf("%s", subckt_saved);
    pos = strstr(copy_line, "optional:");
    if (pos) {
        *pos = '\0';
    } else {
        pos = strstr(copy_line, "params:");
        if (pos) {
            *pos = '\0';
        } else {
            pos = strstr(copy_line, "text:");
            if (pos) {
                *pos = '\0';
            }
        }
    }
    /* skip past .subckt and its name */
    tok = strtok(copy_line, " \t");
    tok = strtok(NULL, " \t");
    while ((tok = strtok(NULL, " \t")) != NULL) {
        port_type = DIR_UNKNOWN;
        inp = (find_input_pin(tok) ? TRUE : FALSE);
        outp = (find_output_pin(tok) ? TRUE : FALSE);
        tri = (find_tristate_pin(tok) ? TRUE : FALSE);
        if (tri) {
            if (outp && inp) {
                port_type = DIR_INOUT;
            } else if (outp) {
                port_type = DIR_OUT;
            }
        } else {
            if (outp && inp) {
                port_type = DIR_OUT;
            } else if (outp) {
                port_type = DIR_OUT;
            } else if (inp) {
                port_type = DIR_IN;
            }
        }
        add_port(tok, port_type);
    }
    tfree(copy_line);
}

/*
  Compound gates have delays added to the final gate.
*/
static char *xspice_tab[XSPICESZ] = {
    "d_and",        // D_AND
    "d_or",         // D_AO
    "d nor",        // D_AOI
    "d_buffer",     // D_BUF
    "d_inverter",   // D_INV
    "d_nand",       // D_NAND
    "d_nor",        // D_NOR
    "d_xnor",       // D_NXOR
    "d_and",        // D_OA
    "d_nand",       // D_OAI
    "d_or",         // D_OR
    "d_xor",        // D_XOR
    "d_dff",        // D_DFF
    "d_jkff",       // D_JKFF
    "d_dlatch",     // D_DLTCH
    "d_srlatch",    // D_SRFF
    "d_pullup",     // D_UP
    "d_pulldown",   // D_DOWN
    "d_tristate",   // D_TRI
};

static char *find_xspice_for_delay(char *itype)
{
    /*
      Returns xspice device name of the timing model corresponding to itype
    */
    switch (itype[0]) {
    case 'a': {
        /* and anda and3 and3a */
        if (eq(itype, "and"))   { return xspice_tab[D_AND]; }
        if (eq(itype, "anda"))  { return xspice_tab[D_AND]; }
        if (eq(itype, "and3"))  { return xspice_tab[D_AND]; }
        if (eq(itype, "and3a")) { return xspice_tab[D_AND]; }

        if (eq(itype, "ao"))  { return xspice_tab[D_AO]; }
        if (eq(itype, "aoi")) { return xspice_tab[D_AOI]; }
        break;
    }
    case 'b': {
        /* buf3 buf3a */
        if (eq(itype, "buf3a")) { return xspice_tab[D_TRI]; }
        if (eq(itype, "buf"))   { return xspice_tab[D_BUF]; }
        if (eq(itype, "bufa"))  { return xspice_tab[D_BUF]; }
        if (eq(itype, "buf3"))  { return xspice_tab[D_TRI]; }
        break;
    }
    case 'd': {
        if (eq(itype, "dff"))   { return xspice_tab[D_DFF]; }
        if (eq(itype, "dltch")) { return xspice_tab[D_DLTCH]; }
        break;
    }
    case 'i': {
        /* inv inva inv3 inv3a */
        if (eq(itype, "inv"))   { return xspice_tab[D_INV]; }
        if (eq(itype, "inv3a")) { return xspice_tab[D_INV]; }
        if (eq(itype, "inva"))  { return xspice_tab[D_INV]; }
        if (eq(itype, "inv3"))  { return xspice_tab[D_INV]; }
        break;
    }
    case 'j': {
        if (eq(itype, "jkff")) { return xspice_tab[D_JKFF]; }
        break;
    }
    case 'n': {
        /* nand nanda nand3 nand3a */
        if (eq(itype, "nand"))   { return xspice_tab[D_NAND]; }
        if (eq(itype, "nanda"))  { return xspice_tab[D_NAND]; }
        if (eq(itype, "nand3"))  { return xspice_tab[D_NAND]; }
        if (eq(itype, "nand3a")) { return xspice_tab[D_NAND]; }

        /* nor nora nor3 nor3a */
        if (eq(itype, "nor"))   { return xspice_tab[D_NOR]; }
        if (eq(itype, "nora"))  { return xspice_tab[D_NOR]; }
        if (eq(itype, "nor3"))  { return xspice_tab[D_NOR]; }
        if (eq(itype, "nor3a")) { return xspice_tab[D_NOR]; }

        /* nxor nxora nxor3 nxor3a */
        if (eq(itype, "nxor"))   { return xspice_tab[D_NXOR]; }
        if (eq(itype, "nxora"))  { return xspice_tab[D_NXOR]; }
        if (eq(itype, "nxor3"))  { return xspice_tab[D_NXOR]; }
        if (eq(itype, "nxor3a")) { return xspice_tab[D_NXOR]; }
        break;
    }
    case 'o': {
        /* or ora or3 or3a */
        if (eq(itype, "or"))   { return xspice_tab[D_OR]; }
        if (eq(itype, "ora"))  { return xspice_tab[D_OR]; }
        if (eq(itype, "or3"))  { return xspice_tab[D_OR]; }
        if (eq(itype, "or3a")) { return xspice_tab[D_OR]; }

        if (eq(itype, "oa"))  { return xspice_tab[D_OA]; }
        if (eq(itype, "oai")) { return xspice_tab[D_OAI]; }
        break;
    }
    case 'p': {
        if (eq(itype, "pulldn")) { return xspice_tab[D_DOWN]; }
        if (eq(itype, "pullup")) { return xspice_tab[D_UP]; }
        break;
    }
    case 's': {
/*  Not implemented
        if (eq(itype, "srff")) { return xspice_tab[D_SRFF]; }
*/
        break;
    }
    case 'x': {
        /* xor xora xor3 xor3a */
        if (eq(itype, "xor"))   { return xspice_tab[D_XOR]; }
        if (eq(itype, "xora"))  { return xspice_tab[D_XOR]; }
        if (eq(itype, "xor3"))  { return xspice_tab[D_XOR]; }
        if (eq(itype, "xor3a")) { return xspice_tab[D_XOR]; }
        break;
    }
    default:
        break;
    };
    return NULL;
}

/* NOTE
  Xlator and Xlate
  Xlate struct data is stored in an Xlator list struct.
  Used to save translated instance and model statements.
  After an Xlator has been created, Xlate data is generated via
  create_xlate() and entered into the Xlator by add_xlator().
  A first_xlator() ... next_xlator() iteration loop will retrieve
  the Xlate data entries in an Xlator.
*/
static void delete_xlate(Xlatep p)
{
    if (p) {
        if (p->translated) { tfree(p->translated); }
        if (p->delays) { tfree(p->delays); }
        if (p->utype) { tfree(p->utype); }
        if (p->xspice) { tfree(p->xspice); }
        if (p->tmodel) { tfree(p->tmodel); }
        if (p->mname) { tfree(p->mname); }
        tfree(p);
    }
    return;
}

static Xlatep create_xlate(char *translated, char *delays, char *utype,
    char *xspice, char *tmodel, char *mname)
{
    /* Any unused parameter is called with an empty string "" */
    Xlatep xlp;

    xlp = TMALLOC(Xlate, 1);
    xlp->next = NULL;
    xlp->translated = TMALLOC(char, strlen(translated) + 1);
    strcpy(xlp->translated, translated);
    xlp->delays = TMALLOC(char, strlen(delays) + 1);
    strcpy(xlp->delays, delays);
    xlp->utype = TMALLOC(char, strlen(utype) + 1);
    strcpy(xlp->utype, utype);
    xlp->xspice = TMALLOC(char, strlen(xspice) + 1);
    strcpy(xlp->xspice, xspice);
    xlp->tmodel = TMALLOC(char, strlen(tmodel) + 1);
    strcpy(xlp->tmodel, tmodel);
    xlp->mname = TMALLOC(char, strlen(mname) + 1);
    strcpy(xlp->mname, mname);
    return xlp;
}

static Xlatep create_xlate_translated(char *translated)
{
    return create_xlate(translated, "", "", "", "", "");
}

static Xlatep create_xlate_instance(
    char *translated, char *xspice, char *tmodel, char *mname)
{
    return create_xlate(translated, "", "", xspice, tmodel, mname);
}

static void print_xlate(Xlatep xp)
{
    if (xp) { return; }
    printf("translated %s\n", xp->translated);
    printf("delays %s\n", xp->delays);
    printf("utype %s ", xp->utype);
    printf("xspice %s ", xp->xspice);
    printf("tmodel %s ", xp->tmodel);
    printf("mname %s\n", xp->mname);
}

static Xlatorp create_xlator(void)
{
    Xlatorp xp;

    xp = TMALLOC(Xlator, 1);
    xp->head = NULL;
    xp->tail = NULL;
    xp->iter = NULL;
    return xp;
}

static void delete_xlator(Xlatorp xp)
{
    Xlatep x, next;

    if (xp) {
        if (xp->head) {
            x = xp->head;
            next = x->next;
            delete_xlate(x);
            while (next) {
                x = next;
                next = x->next;
                delete_xlate(x);
            }
        }
        tfree(xp);
    }
    return;
}

static Xlatorp add_xlator(Xlatorp xp, Xlatep x)
{
    Xlatep prev;

    if (!xp || !x) { return NULL; }
    if (!xp->head) {
        xp->head = x;
        xp->tail = x;
        xp->iter = x;
        x->next = NULL;
    } else {
        prev = xp->tail;
        prev->next = x;
        x->next = NULL;
        xp->tail = x;
    }
    return xp;
}

static Xlatep first_xlator(Xlatorp xp)
{
    Xlatep xret;

    if (!xp) { return NULL; }
    xp->iter = xp->head;
    if (!xp->iter) {
        return NULL;
    } else {
        xret = xp->iter;
        xp->iter = xret->next;
        return xret;
    }
}

static Xlatep next_xlator(Xlatorp xp)
{
    Xlatep ret;

    if (!xp) { return NULL; }
    ret = xp->iter;
    if (!ret) {
        return ret;
    }
    xp->iter = ret->next;
    return ret;
}

static Xlatorp append_xlator(Xlatorp dest, Xlatorp src)
{
    Xlatep x1, copy;

    if (!dest || !src) { return NULL; }
    for (x1 = first_xlator(src); x1; x1 = next_xlator(src)) {
        copy = create_xlate(x1->translated, x1->delays, x1->utype,
            x1->xspice, x1->tmodel, x1->mname);
        dest = add_xlator(dest, copy);
    }
    return dest;
}

#ifdef TRACE
static void interpret_xlator(Xlatorp xp, BOOL brief)
{
    Xlatep x1;

    if (!xp) { return; }
    for (x1 = first_xlator(xp); x1; x1 = next_xlator(xp)) {
        if (brief) {
            if (strlen(x1->translated) > 0) {
                printf("  %s\n", x1->translated);
            }
        } else {
            if (strncmp(x1->translated, ".model", strlen(".model")) == 0) {
                printf("MODEL %s\n", x1->translated);
            } else if (strlen(x1->translated) > 0) {
                printf("INSTANCE %s\n", x1->translated);
            }
            if (strlen(x1->delays) > 0) {
                printf("DELAYS %s\n", x1->delays);
            }
            printf("==> utype: %s xspice: %s tmodel: %s mname: %s\n",
                x1->utype, x1->xspice, x1->tmodel, x1->mname);
        }
    }
    return;
}
#endif

/* static Xlatorp for collecting timing model delays */
static Xlatorp model_xlatorp = NULL;

/* static Xlatorp for default zero delay models */
static Xlatorp default_models = NULL;

/* static Xlatorp for storing translated instance and model statements */
static Xlatorp translated_p = NULL;

static void create_translated_xlator(void)
{
    translated_p = create_xlator();
}

static void cleanup_translated_xlator(void)
{
#ifdef TRACE
    printf("\nStart cleanup\n");
    interpret_xlator(translated_p, TRUE);
    printf("End cleanup\n");
#endif
    delete_xlator(translated_p);
    translated_p = NULL;
}

/* NOTE
  replacement_udevice_cards() is called by the driver function u_instances()
  in frontend/inpcom.c so that the new Xspice instance and model statements
  can be spliced into the card deck.
*/
struct card *replacement_udevice_cards(void)
{
    struct card *newcard = NULL, *nextcard = NULL;
    char *new_str = NULL;
    Xlatep x;
    int count = 0;

    if (!translated_p) { return NULL; }
    if (add_zero_delay_inverter_model) {
        x = create_xlate_translated(".model d_zero_inv99 d_inverter");
        translated_p = add_xlator(translated_p, x);
    }
    for (x = first_xlator(translated_p); x; x = next_xlator(translated_p)) {
        new_str = copy(x->translated);
        if (count == 0) {
            count++;
            newcard = insert_new_line(NULL, new_str, 0, 0);
        } else if (count == 1) {
            count++;
            nextcard = insert_new_line(newcard, new_str, 0, 0);
        } else {
            count++;
            nextcard = insert_new_line(nextcard, new_str, 0, 0);
        }
    }
    return newcard;
}

void initialize_udevice(char *subckt_line)
{
    Xlatep xdata;

    if (subckt_saved) {
        tfree(subckt_saved);
    } else {
        if (subckt_line && strncmp(subckt_line, ".subckt", 7) == 0) {
            size_t sz = strlen(subckt_line) + 1;
            subckt_saved = TMALLOC(char, sz);
            (void) memcpy(subckt_saved, subckt_line, sz);
        } else {
            subckt_saved = NULL;
        }
    }
    create_translated_xlator();
    model_xlatorp = create_xlator();
    default_models = create_xlator();
    /*  .model d0_gate ugate ()  */
    xdata = create_xlate("", "", "ugate", "", "d0_gate", "");
    (void) add_xlator(default_models, xdata);
    /*  .model d0_gff ugff ()  */
    xdata = create_xlate("", "", "ugff", "", "d0_gff", "");
    (void) add_xlator(default_models, xdata);
    /*  .model d0_eff ueff ()  */
    xdata = create_xlate("", "", "ueff", "", "d0_eff", "");
    (void) add_xlator(default_models, xdata);
    /*  .model d0_tgate utgate ()  */
    xdata = create_xlate("", "", "utgate", "", "d0_tgate", "");
    (void) add_xlator(default_models, xdata);
    /* Variable ps_pins_and_ports != 0 to turn on pins and ports */
    if (!cp_getvar("ps_pins_and_ports", CP_NUM, &pins_and_ports, 0)) {
        pins_and_ports = 0;
    }
    /* reset for the new subckt */
    add_zero_delay_inverter_model = FALSE;
}

void cleanup_udevice(void)
{
    create_ports_list();
    cleanup_translated_xlator();
    delete_xlator(model_xlatorp);
    model_xlatorp = NULL;
    delete_xlator(default_models);
    default_models = NULL;
    clear_all_pin_lists();
    if (subckt_saved) {
        tfree(subckt_saved);
        subckt_saved = NULL;
    }
    add_zero_delay_inverter_model = FALSE;
}

static Xlatep create_xlate_model(char *delays,
    char *utype, char *xspice, char *tmodel)
{
    return create_xlate("", delays, utype, xspice, tmodel, "");
}

static Xlatep find_tmodel_in_xlator(Xlatep x, Xlatorp xlp)
{
    /* Only for timing model xlators */
    Xlatep x1;

    if (!x) { return NULL; }
    if (!xlp) { return NULL; }
    for (x1 = first_xlator(xlp); x1; x1 = next_xlator(xlp)) {
        if (eq(x1->tmodel, x->tmodel) && eq(x1->utype, x->utype)) {
            if (eq(x1->xspice, x->xspice)) {
                return x1;
            }
        }
    }
    return NULL;
}

static Xlatep find_in_model_xlator(Xlatep x)
{
    Xlatep x1;

    x1 = find_tmodel_in_xlator(x, model_xlatorp);
    if (x1) { return x1; }
    x1 = find_tmodel_in_xlator(x, default_models);
    return x1;
}

static void add_delays_to_model_xlator(char *delays,
    char *utype, char *xspice, char *tmodel)
{
    /*
      Specify xspice as "d_dlatch" or "d_srlatch" for ugff
      otherwise xspice == ""
    */
    Xlatep x = NULL, x1 = NULL;

    if (!model_xlatorp) { return; }
    x = create_xlate_model(delays, utype, xspice, tmodel);
    x1 = find_in_model_xlator(x);
    if (x1) {
/*
        printf("Already found timing model %s utype %s\n",
             x1->tmodel, x1->utype);
*/
        delete_xlate(x);
        return;
    }
    (void) add_xlator(model_xlatorp, x);
}

/* classify gate variants */
static BOOL is_tristate_buf_array(char *itype)
{
    if (eq(itype, "buf3a")) { return TRUE; }
    if (eq(itype, "inv3a")) { return TRUE; }
    return FALSE;
}

static BOOL is_tristate_xor_array(char *itype)
{
    /* xor/nxor have vector inputs */
    if (eq(itype, "xor3a")) { return TRUE; }
    if (eq(itype, "nxor3a")) { return TRUE; }
    return FALSE;
}

static BOOL is_tristate_vector_array(char *itype)
{
    if (eq(itype, "and3a")) { return TRUE; }
    if (eq(itype, "nand3a")) { return TRUE; }
    if (eq(itype, "or3a")) { return TRUE; }
    if (eq(itype, "nor3a")) { return TRUE; }
    return FALSE;
}

static BOOL is_tristate_array(char *itype)
{
    if (is_tristate_buf_array(itype)) { return TRUE; }
    if (is_tristate_vector_array(itype)) { return TRUE; }
    if (is_tristate_xor_array(itype)) { return TRUE; }
    return FALSE;
}

static BOOL is_buf_tristate(char *itype)
{
    if (eq(itype, "buf3")) { return TRUE; }
    if (eq(itype, "inv3")) { return TRUE; }
    return FALSE;
}

static BOOL is_xor_tristate(char *itype)
{
    /* xor/nxor have vector inputs */
    if (eq(itype, "xor3")) { return TRUE; }
    if (eq(itype, "nxor3")) { return TRUE; }
    return FALSE;
}

static BOOL is_vector_tristate(char *itype)
{
    if (eq(itype, "and3")) { return TRUE; }
    if (eq(itype, "nand3")) { return TRUE; }
    if (eq(itype, "or3")) { return TRUE; }
    if (eq(itype, "nor3")) { return TRUE; }
    return FALSE;
}

static BOOL is_tristate(char *itype)
{
    if (is_buf_tristate(itype)) { return TRUE; }
    if (is_vector_tristate(itype)) { return TRUE; }
    if (is_xor_tristate(itype)) { return TRUE; }
    return FALSE;
}

static BOOL is_vector_gate_array(char *itype)
{
    if (eq(itype, "anda")) { return TRUE; }
    if (eq(itype, "nanda")) { return TRUE; }
    if (eq(itype, "ora")) { return TRUE; }
    if (eq(itype, "nora")) { return TRUE; }
    return FALSE;
}

static BOOL is_buf_gate_array(char *itype)
{
    if (eq(itype, "bufa")) { return TRUE; }
    if (eq(itype, "inva")) { return TRUE; }
    return FALSE;
}

static BOOL is_xor_gate_array(char *itype)
{
    /* xor/nxor have vector inputs */
    if (eq(itype, "xora")) { return TRUE; }
    if (eq(itype, "nxora")) { return TRUE; }
    return FALSE;
}

static BOOL is_gate_array(char *itype)
{
    if (is_vector_gate_array(itype)) { return TRUE; }
    if (is_buf_gate_array(itype)) { return TRUE; }
    if (is_xor_gate_array(itype)) { return TRUE; }
    return FALSE;
}

static BOOL is_vector_gate(char *itype)
{
    if (eq(itype, "nand")) { return TRUE; }
    if (eq(itype, "and")) { return TRUE; }
    if (eq(itype, "nor")) { return TRUE; }
    if (eq(itype, "or")) { return TRUE; }
    return FALSE;
}

static BOOL is_buf_gate(char *itype)
{
    if (eq(itype, "inv")) { return TRUE; }
    if (eq(itype, "buf")) { return TRUE; }
    return FALSE;
}

static BOOL is_xor_gate(char *itype)
{
    /* xor/nxor have vector inputs */
    if (eq(itype, "xor")) { return TRUE; }
    if (eq(itype, "nxor")) { return TRUE; }
    return FALSE;
}

static BOOL is_gate(char *itype)
{
    if (is_vector_gate(itype)) { return TRUE; }
    if (is_buf_gate(itype)) { return TRUE; }
    if (is_xor_gate(itype)) { return TRUE; }
    return FALSE;
}

static BOOL is_compound_gate(char *itype)
{
    if (eq(itype, "aoi")) { return TRUE; }
    if (eq(itype, "ao")) { return TRUE; }
    if (eq(itype, "oa")) { return TRUE; }
    if (eq(itype, "oai")) { return TRUE; }
    return FALSE;
}

static BOOL has_vector_inputs(char *itype)
{
    switch (itype[0]) {
    case 'a': {
        if (strncmp(itype, "and", 3) == 0) { return TRUE; }
        break;
    }
    case 'n': {
        if (strncmp(itype, "nand", 4) == 0) { return TRUE; }
        if (strncmp(itype, "nor", 3) == 0) { return TRUE; }
        if (strncmp(itype, "nxor", 4) == 0) { return TRUE; }
        break;
    }
    case 'o': {
        if (strncmp(itype, "or", 2) == 0) { return TRUE; }
        break;
    }
    case 'x': {
        if (strncmp(itype, "xor", 3) == 0) { return TRUE; }
        break;
    }
    default:
        break;
    };
    return FALSE;
}

static void delete_instance_hdr(struct instance_hdr *hdr)
{
    if (!hdr) { return; }
    if (hdr->instance_name) { tfree(hdr->instance_name); }
    if (hdr->instance_type) { tfree(hdr->instance_type); }
    tfree(hdr);
    return;
}

/* NOTE
  There are translate_...() functions for each instance type (gates, arrays
  of gates, compound gates, dff, jkff, dltch).
  For each type (except pullup/down) an instance struct is created by calling
  the corresponding create_..._instance() then add_..._inout_timing_model()
  functions. The instance struct returned by add_..._inout_timing_model()
  is passed to the corresponding gen_..._instance() function, and the Xspice
  translations are stored in Xlator translated_p.
*/
static struct dff_instance *create_dff_instance(struct instance_hdr *hdrp)
{
    struct dff_instance *dffip;

    dffip = TMALLOC(struct dff_instance, 1);
    dffip->hdrp = hdrp;
    dffip->prebar = NULL;
    dffip->clrbar = NULL;
    dffip->clk = NULL;
    dffip->num_gates = 0;
    dffip->d_in = NULL;
    dffip->q_out = NULL;
    dffip->qb_out = NULL;
    dffip->tmodel = NULL;
    return dffip;
}

static void delete_dff_instance(struct dff_instance *dp)
{
    char **arr;
    int i;

    if (!dp) { return; }
    if (dp->hdrp) { delete_instance_hdr(dp->hdrp); }
    if (dp->prebar) { tfree(dp->prebar); }
    if (dp->clrbar) { tfree(dp->clrbar); }
    if (dp->clk) { tfree(dp->clk); }
    if (dp->tmodel) { tfree(dp->tmodel); }
    if (dp->num_gates > 0) {
        if (dp->d_in) {
            arr = dp->d_in;
            for (i = 0; i < dp->num_gates; i++) {
                tfree(arr[i]);
            }
            tfree(dp->d_in);
        }
        if (dp->q_out) {
            arr = dp->q_out;
            for (i = 0; i < dp->num_gates; i++) {
                tfree(arr[i]);
            }
            tfree(dp->q_out);
        }
        if (dp->qb_out) {
            arr = dp->qb_out;
            for (i = 0; i < dp->num_gates; i++) {
                tfree(arr[i]);
            }
            tfree(dp->qb_out);
        }
    }
    tfree(dp);
    return;
}

static struct dltch_instance *create_dltch_instance(struct instance_hdr *hdrp)
{
    struct dltch_instance *dlp;

    dlp = TMALLOC(struct dltch_instance, 1);
    dlp->hdrp = hdrp;
    dlp->prebar = NULL;
    dlp->clrbar = NULL;
    dlp->gate = NULL;
    dlp->num_gates = 0;
    dlp->d_in = NULL;
    dlp->q_out = NULL;
    dlp->qb_out = NULL;
    dlp->tmodel = NULL;
    return dlp;
}

static void delete_dltch_instance(struct dltch_instance *dlp)
{
    char **arr;
    int i;

    if (!dlp) { return; }
    if (dlp->hdrp) { delete_instance_hdr(dlp->hdrp); }
    if (dlp->prebar) { tfree(dlp->prebar); }
    if (dlp->clrbar) { tfree(dlp->clrbar); }
    if (dlp->gate) { tfree(dlp->gate); }
    if (dlp->tmodel) { tfree(dlp->tmodel); }
    if (dlp->num_gates > 0) {
        if (dlp->d_in) {
            arr = dlp->d_in;
            for (i = 0; i < dlp->num_gates; i++) {
                tfree(arr[i]);
            }
            tfree(dlp->d_in);
        }
        if (dlp->q_out) {
            arr = dlp->q_out;
            for (i = 0; i < dlp->num_gates; i++) {
                tfree(arr[i]);
            }
            tfree(dlp->q_out);
        }
        if (dlp->qb_out) {
            arr = dlp->qb_out;
            for (i = 0; i < dlp->num_gates; i++) {
                tfree(arr[i]);
            }
            tfree(dlp->qb_out);
        }
    }
    tfree(dlp);
    return;
}


static struct jkff_instance *create_jkff_instance(struct instance_hdr *hdrp)
{
    struct jkff_instance *jkffip;

    jkffip = TMALLOC(struct jkff_instance, 1);
    jkffip->hdrp = hdrp;
    jkffip->prebar = NULL;
    jkffip->clrbar = NULL;
    jkffip->clkbar = NULL;
    jkffip->num_gates = 0;
    jkffip->j_in = NULL;
    jkffip->k_in = NULL;
    jkffip->q_out = NULL;
    jkffip->qb_out = NULL;
    jkffip->tmodel = NULL;
    return jkffip;
}

static void delete_jkff_instance(struct jkff_instance *jkp)
{
    char **arr;
    int i;

    if (!jkp) { return; }
    if (jkp->hdrp) { delete_instance_hdr(jkp->hdrp); }
    if (jkp->prebar) { tfree(jkp->prebar); }
    if (jkp->clrbar) { tfree(jkp->clrbar); }
    if (jkp->clkbar) { tfree(jkp->clkbar); }
    if (jkp->tmodel) { tfree(jkp->tmodel); }
    if (jkp->num_gates > 0) {
        if (jkp->j_in) {
            arr = jkp->j_in;
            for (i = 0; i < jkp->num_gates; i++) {
                tfree(arr[i]);
            }
            tfree(jkp->j_in);
        }
        if (jkp->k_in) {
            arr = jkp->k_in;
            for (i = 0; i < jkp->num_gates; i++) {
                tfree(arr[i]);
            }
            tfree(jkp->k_in);
        }
        if (jkp->q_out) {
            arr = jkp->q_out;
            for (i = 0; i < jkp->num_gates; i++) {
                tfree(arr[i]);
            }
            tfree(jkp->q_out);
        }
        if (jkp->qb_out) {
            arr = jkp->qb_out;
            for (i = 0; i < jkp->num_gates; i++) {
                tfree(arr[i]);
            }
            tfree(jkp->qb_out);
        }
    }
    tfree(jkp);
    return;
}

static struct gate_instance *create_gate_instance(struct instance_hdr *hdrp)
{
    struct gate_instance *gip;

    gip = TMALLOC(struct gate_instance, 1);
    gip->hdrp = hdrp;
    gip->num_gates = 0;
    gip->width = 0;
    gip->num_ins = 0;
    gip->inputs = NULL;
    gip->enable = NULL;
    gip->num_outs = 0;
    gip->outputs = NULL;
    gip->tmodel = NULL;
    return gip;
}

static struct compound_instance *create_compound_instance(
    struct instance_hdr *hdrp)
{
    struct compound_instance *ci;

    ci = TMALLOC(struct compound_instance, 1);
    ci->hdrp = hdrp;
    ci->num_gates = 0;
    ci->width = 0;
    ci->num_ins = 0;
    ci->inputs = NULL;
    ci->output = NULL;
    ci->tmodel = NULL;
    return ci;
}

static void delete_compound_instance(struct compound_instance *ci)
{
    char **namearr;
    int i;

    if (!ci) { return; }
    if (ci->hdrp) { delete_instance_hdr(ci->hdrp); }
    if (ci->num_ins > 0 && ci->inputs) {
        namearr = ci->inputs;
        for (i = 0; i < ci->num_ins; i++) {
            tfree(namearr[i]);
        }
        tfree(ci->inputs);
    }
    if (ci->output) { tfree(ci->output); }
    if (ci->tmodel) { tfree(ci->tmodel); }
    tfree(ci);
    return;
}

static void delete_gate_instance(struct gate_instance *gip)
{
    char **namearr;
    int i;

    if (!gip) { return; }
    if (gip->hdrp) { delete_instance_hdr(gip->hdrp); }
    if (gip->enable) { tfree(gip->enable); }
    if (gip->num_ins > 0 && gip->inputs) {
        namearr = gip->inputs;
        for (i = 0; i < gip->num_ins; i++) {
            tfree(namearr[i]);
        }
        tfree(gip->inputs);
    }
    if (gip->num_outs > 0 && gip->outputs) {
        namearr = gip->outputs;
        for (i = 0; i < gip->num_outs; i++) {
            tfree(namearr[i]);
        }
        tfree(gip->outputs);
    }
    if (gip->tmodel) { tfree(gip->tmodel); }
    tfree(gip);
    return;
}

static struct instance_hdr *create_instance_header(char *line)
{
    char *tok, *p1, *p3, *p4, *endp, *newline, *tmp, *tmp1;
    struct instance_hdr *hdr = NULL;

    newline = TMALLOC(char, strlen(line) + 1);
    (void) memcpy(newline, line, strlen(line) + 1);
    hdr = TMALLOC(struct instance_hdr, 1);
    hdr->num1 = -1;
    hdr->num2 = -1;
    /* instance name */
    tok = strtok(newline, " \t");
    tmp = TMALLOC(char, strlen(tok) + 1);
    (void) memcpy(tmp, tok, strlen(tok) + 1);
    hdr->instance_name = tmp;
    /* instance type */
    tok = strtok(NULL, " \t");
    p1 = strchr(tok, '(');
    if (p1) {
        /* ...(n1,n2) or ...(n1) */
        tmp = TMALLOC(char, strlen(tok) + 1);
        strcpy(tmp, tok);
        p4 = strchr(tmp, '(');
        *p4 = '\0';
        tmp1 = TMALLOC(char, strlen(tmp) + 1);
        (void) memcpy(tmp1, tmp, strlen(tmp) + 1);
        hdr->instance_type = tmp1;
        tfree(tmp);

        p3 = strchr(tok, ',');
        if (p3) {
            hdr->num1 = (int) strtol(p1 + 1, &endp, 10);
            hdr->num2 = (int) strtol(p3 + 1, &endp, 10);
        } else {
            hdr->num1 = (int) strtol(p1 + 1, &endp, 10);
        }
    } else {
        tmp = TMALLOC(char, strlen(tok) + 1);
        (void) memcpy(tmp, tok, strlen(tok) + 1);
        hdr->instance_type = tmp;
    }
    tfree(newline);
    return hdr;
}

char *new_inverter(char *iname, char *node, Xlatorp xlp)
{
    /* Return the name of the output of the new inverter */
    /* tfree the returned string after it has been used by the caller */
    char *tmp = NULL;
    Xlatep xdata = NULL;

    tmp = tprintf("a%s_%s  %s  not_a%s_%s  d_zero_inv99",
        iname, node, node, iname, node);
    /* instantiate the new inverter */
    /* e.g. au5_s1bar  s1bar  not_au5_s1bar  d_zero_inv99 */
    xdata = create_xlate_translated(tmp);
    (void) add_xlator(xlp, xdata);
    tfree(tmp);
    /* the name of the inverter output */
    tmp = tprintf("not_a%s_%s", iname, node);
    return tmp;
}

static BOOL gen_timing_model(
    char *tmodel, char *utype, char *xspice, char *newname, Xlatorp xlp)
{
    /*
      tmodel is the name of the pspice model of type utype.
      If the model is found in the model_xlatorp list, the delays are used.
      A new xspice model is created using newname and xspice(d_and etc.):
        .model <newname> <xspice>[<delays>]  where <delays> maybe zero/blank.
      The new .model statement is added to the Xlatorp of the translated
      xspice instance and model lines (not to be confused with model_xlatorp.
    */
    Xlatep xin = NULL, xout = NULL, newdata;
    char *s1;
    BOOL retval;

    if (eq(utype, "ugff")) {
        xin = create_xlate_model("", utype, xspice, tmodel);
    } else {
        xin = create_xlate_model("", utype, "", tmodel);
    }
    xout = find_in_model_xlator(xin);
    if (xout) {
        /* Don't delete xout or the model_xlatorp will be corrupted */
        print_xlate(xout);
        if (xout->delays && strlen(xout->delays) > 0) {
            s1 = tprintf(".model %s %s%s", newname, xspice, xout->delays);
        } else {
            s1 = tprintf(".model %s %s", newname, xspice);
        }
        newdata = create_xlate_translated(s1);
        tfree(s1);
        (void) add_xlator(xlp, newdata);
        retval = TRUE;
    } else {
        retval = FALSE;
    }
    delete_xlate(xin);
    return retval;
}


static Xlatorp gen_dff_instance(struct dff_instance *ip)
{
    char *itype, *iname, **darr, **qarr, **qbarr;
    char *preb, *clrb, *clk, *tmodel, *qout, *qbout;
    int i, num_gates;
    char *modelnm, *s1;
    Xlatorp xxp = NULL;
    Xlatep xdata = NULL;
    BOOL need_preb_inv = FALSE, need_clrb_inv = FALSE;

    if (!ip) { return NULL; }
    itype = ip->hdrp->instance_type;
    iname = ip->hdrp->instance_name;
    num_gates = ip->num_gates;
    darr = ip->d_in;
    qarr = ip->q_out;
    qbarr = ip->qb_out;
    preb = ip->prebar;
    clrb = ip->clrbar;

    xxp = create_xlator();
    add_input_pin(preb);
    if (eq(preb, "$d_hi")) {
        preb = "NULL";
    } else {
        need_preb_inv = TRUE;
        preb = new_inverter(iname, preb, xxp);
    }

    add_input_pin(clrb);
    if (eq(clrb, "$d_hi")) {
        clrb = "NULL";
    } else {
        need_clrb_inv = TRUE;
        clrb = new_inverter(iname, clrb, xxp);
    }

    clk = ip->clk;
    add_input_pin(clk);
    tmodel = ip->tmodel;
    /* model name, same for each dff */
    modelnm = tprintf("d_a%s_%s", iname, itype);
    for (i = 0; i < num_gates; i++) {
        qout = qarr[i];
        add_output_pin(qout);
        if (eq(qout, "$d_nc")) {
            qout = "NULL";
        }
        qbout = qbarr[i];
        add_output_pin(qbout);
        if (eq(qbout, "$d_nc")) {
            qbout = "NULL";
        }
        add_input_pin(darr[i]);
        s1 = tprintf( "a%s_%d  %s  %s  %s  %s  %s  %s  %s",
            iname, i, darr[i], clk, preb, clrb, qout, qbout, modelnm
        );
        xdata = create_xlate_instance(s1, " d_dff", tmodel, modelnm);
        xxp = add_xlator(xxp, xdata);
        tfree(s1);
    }
    if (!gen_timing_model(tmodel, "ueff", "d_dff", modelnm, xxp)) {
        printf("WARNING unable to find tmodel %s for %s d_dff\n",
            tmodel, modelnm);
    }
    if (need_preb_inv || need_clrb_inv) {
        add_zero_delay_inverter_model = TRUE;
    }
    if (need_preb_inv) { tfree(preb); }
    if (need_clrb_inv) { tfree(clrb); }
    tfree(modelnm);

    return xxp;
}

static Xlatorp gen_jkff_instance(struct jkff_instance *ip)
{
    char *itype, *iname, **jarr, **karr, **qarr, **qbarr;
    char *preb, *clrb, *clkb, *tmodel, *qout, *qbout;
    int i, num_gates;
    char *modelnm, *s1;
    Xlatorp xxp = NULL;
    Xlatep xdata = NULL;
    BOOL need_preb_inv = FALSE, need_clrb_inv = FALSE;

    if (!ip) { return NULL; }
    itype = ip->hdrp->instance_type;
    iname = ip->hdrp->instance_name;
    num_gates = ip->num_gates;
    jarr = ip->j_in;
    karr = ip->k_in;
    qarr = ip->q_out;
    qbarr = ip->qb_out;
    preb = ip->prebar;
    clrb = ip->clrbar;

    xxp = create_xlator();
    add_input_pin(preb);
    if (eq(preb, "$d_hi")) {
        preb = "NULL";
    } else {
        need_preb_inv = TRUE;
        preb = new_inverter(iname, preb, xxp);
    }

    add_input_pin(clrb);
    if (eq(clrb, "$d_hi")) {
        clrb = "NULL";
    } else {
        need_clrb_inv = TRUE;
        clrb = new_inverter(iname, clrb, xxp);
    }

    /* require a positive edge clock */
    clkb = ip->clkbar;
    add_input_pin(clkb);
    clkb = new_inverter(iname, clkb, xxp);

    tmodel = ip->tmodel;
    /* model name, same for each jkff */
    modelnm = tprintf("d_a%s_%s", iname, itype);
    for (i = 0; i < num_gates; i++) {
        qout = qarr[i];
        add_output_pin(qout);
        if (eq(qout, "$d_nc")) {
            qout = "NULL";
        }
        qbout = qbarr[i];
        add_output_pin(qbout);
        if (eq(qbout, "$d_nc")) {
            qbout = "NULL";
        }
        add_input_pin(jarr[i]);
        add_input_pin(karr[i]);
        s1 = tprintf("a%s_%d  %s  %s  %s  %s  %s  %s  %s  %s",
            iname, i, jarr[i], karr[i], clkb, preb, clrb, qout, qbout, modelnm
        );
        xdata = create_xlate_instance(s1, " d_jkff", tmodel, modelnm);
        xxp = add_xlator(xxp, xdata);
        tfree(s1);
    }
    if (!gen_timing_model(tmodel, "ueff", "d_jkff", modelnm, xxp)) {
        printf("WARNING unable to find tmodel %s for %s d_jkff\n",
            tmodel, modelnm);
    }
    add_zero_delay_inverter_model = TRUE;
    tfree(clkb);
    if (need_preb_inv) { tfree(preb); }
    if (need_clrb_inv) { tfree(clrb); }
    tfree(modelnm);

    return xxp;
}

static Xlatorp gen_dltch_instance(struct dltch_instance *ip)
{
    char *itype, *iname, **darr, **qarr, **qbarr;
    char *preb, *clrb, *gate, *tmodel, *qout, *qbout;
    int i, num_gates;
    char *modelnm, *s1, *s2, *s3;
    Xlatorp xxp = NULL;
    Xlatep xdata = NULL;
    BOOL need_preb_inv = FALSE, need_clrb_inv = FALSE;

    if (!ip) { return NULL; }
    itype = ip->hdrp->instance_type;
    iname = ip->hdrp->instance_name;
    num_gates = ip->num_gates;
    darr = ip->d_in;
    qarr = ip->q_out;
    qbarr = ip->qb_out;
    preb = ip->prebar;
    clrb = ip->clrbar;

    xxp = create_xlator();
    add_input_pin(preb);
    if (eq(preb, "$d_hi")) {
        preb = "NULL";
    } else {
        need_preb_inv = TRUE;
        preb = new_inverter(iname, preb, xxp);
    }

    add_input_pin(clrb);
    if (eq(clrb, "$d_hi")) {
        clrb = "NULL";
    } else {
        need_clrb_inv = TRUE;
        clrb = new_inverter(iname, clrb, xxp);
    }
    gate = ip->gate;
    add_input_pin(gate);
    tmodel = ip->tmodel;
    /* model name, same for each latch */
    modelnm = tprintf("d_a%s_%s", iname, itype);
    for (i = 0; i < num_gates; i++) {
        qout = qarr[i];
        add_output_pin(qout);
        if (eq(qout, "$d_nc")) {
            /* NULL not allowed??? */
            s1 = tprintf("a%s_%d  %s  %s  %s  %s  nco%s_%d",
                iname, i, darr[i], gate, preb, clrb, iname, i);
        } else {
            s1 = tprintf("a%s_%d  %s  %s  %s  %s  %s",
                iname, i, darr[i], gate, preb, clrb, qout);
        }
        add_input_pin(darr[i]);
        qbout = qbarr[i];
        add_output_pin(qbout);
        if (eq(qbout, "$d_nc")) {
            /* NULL not allowed??? */
            s2 = tprintf(" ncn%s_%d  %s", iname, i, modelnm);
        } else {
            s2 = tprintf("  %s  %s", qbout, modelnm);
        }
        s3 = tprintf("%s%s", s1, s2);
        xdata = create_xlate_instance(s3, " d_dlatch", tmodel, modelnm);
        xxp = add_xlator(xxp, xdata);
        tfree(s1);
        tfree(s2);
        tfree(s3);
    }
    if (!gen_timing_model(tmodel, "ugff", "d_dlatch", modelnm, xxp)) {
        printf("WARNING unable to find tmodel %s for %s d_dlatch\n",
            tmodel, modelnm);
    }
    if (need_preb_inv || need_clrb_inv) {
        add_zero_delay_inverter_model = TRUE;
    }
    if (need_preb_inv) { tfree(preb); }
    if (need_clrb_inv) { tfree(clrb); }
    tfree(modelnm);

    return xxp;
}

static Xlatorp gen_compound_instance(struct compound_instance *compi)
{
    char **inarr, *itype, *output, *tmodel;
    char *outgate = NULL, *ingates = NULL, *logic_val = NULL;
    int i, j, k, width, num_gates;
    int num_ins_kept = 0;
    char *model_name = NULL, *inst = NULL, **connector = NULL;
    char *new_inst = NULL, *model_stmt = NULL, *final_model_name = NULL;
    char *new_stmt = NULL;
    char *tmp;
    size_t sz = 0;
    Xlatorp xxp = NULL;
    Xlatep xdata = NULL;

    if (!compi) { return NULL; }
    itype = compi->hdrp->instance_type;
    inst = compi->hdrp->instance_name;
    if (eq(itype, "aoi")) {
        outgate = "d_nor";
        ingates = "d_and";
        logic_val = "$d_hi";
    } else if (eq(itype, "ao")) {
        outgate = "d_or";
        ingates = "d_and";
        logic_val = "$d_hi";
    } else if (eq(itype, "oai")) {
        outgate = "d_nand";
        ingates = "d_or";
        logic_val = "$d_lo";
    } else if (eq(itype, "oa")) {
        outgate = "d_and";
        ingates = "d_or";
        logic_val = "$d_lo";
    } else {
        return NULL;
    }
    inarr = compi->inputs;
    width = compi->width;
    num_gates = compi->num_gates;
    output = compi->output;
    tmodel = compi->tmodel;
    model_name = tprintf("d_%s_%s", inst, itype);
    connector = TMALLOC(char *, num_gates);
    xxp = create_xlator();
    k = 0;
    for (i = 0; i < num_gates; i++) {
        for (j = 0; j < width; j++) {
            sz += strlen(inarr[k]) + 8; // Room for space between each name
            k++;
        }
    }
    tmp = TMALLOC(char, sz);
    tmp[0] = '\0';
    k = 0;
    for (i = 0; i < num_gates; i++) {
        connector[i] = tprintf("con_%s_%d", inst, i);
        num_ins_kept = 0;
        tmp[0] = '\0';
        /* $d_hi AND gate inputs are ignored */
        /* $d_lo OR gate inputs are ignored */
        for (j = 0; j < width; j++) {
            if (!eq(inarr[k], logic_val)) {
                num_ins_kept++;
                sprintf(tmp + strlen(tmp), " %s", inarr[k]);
                add_input_pin(inarr[k]);
            }
            k++;
        }
        if (num_ins_kept >= 2) {
            new_inst = tprintf("a%s_%d [%s ] %s %s", inst, i,
                tmp, connector[i], model_name);
            xdata = create_xlate_translated(new_inst);
            xxp = add_xlator(xxp, xdata);
            tfree(new_inst);
        } else if (num_ins_kept == 1) {
            /*
              connector[i] is the remaining input connected
              directly to the OR/NOR, AND/NAND final gate.
            */
            tfree(connector[i]);
            connector[i] = tprintf("%s", tmp);
        }
    }
    /* .model statement for the input gates */
    model_stmt = tprintf(".model %s %s", model_name, ingates);
    xdata = create_xlate_translated(model_stmt);
    xxp = add_xlator(xxp, xdata);
    tfree(model_stmt);

    /* Final OR/NOR, AND/NAND gate */
    final_model_name = tprintf("%s_out", model_name);
    tfree(tmp);

    sz = 0;
    for (i = 0; i < num_gates; i++) {
        sz += strlen(connector[i]) + 8; // Room for space between each name
    }
    tmp = TMALLOC(char, sz);
    tmp[0] = '\0';
    for (i = 0; i < num_gates; i++) {
        sprintf(tmp + strlen(tmp), " %s", connector[i]);
    }
    /* instance statement for the final gate */
    new_stmt = tprintf("a%s_out [%s ] %s %s",
        inst, tmp, output, final_model_name);
    add_output_pin(output);
    xdata = create_xlate_translated(new_stmt);
    xxp = add_xlator(xxp, xdata);
    tfree(new_stmt);
    tfree(tmp);
    /* timing model for output gate */
    if (!gen_timing_model(tmodel, "ugate", outgate,
            final_model_name, xxp)) {
        printf("WARNING unable to find tmodel %s for %s %s\n",
            tmodel, final_model_name, outgate);
    }

    tfree(final_model_name);
    for (i = 0; i < num_gates; i++) {
        if (connector[i]) { tfree(connector[i]); }
    }
    if (connector) { tfree(connector); }
    tfree(model_name);
    return xxp;
}

static Xlatorp gen_gate_instance(struct gate_instance *gip)
{
    char **inarr, **outarr, *itype, *iname, *enable, *tmodel;
    char *xspice = NULL, *connector = NULL;
    BOOL vector = FALSE, tristate_gate = FALSE, simple_gate = FALSE;
    BOOL tristate_array = FALSE, simple_array = FALSE;
    BOOL add_tristate = FALSE;
    char *modelnm = NULL, *startvec = NULL, *endvec = NULL;
    char *input_buf = NULL;
    int i, j, k, width, num_gates, num_ins, num_outs;
    size_t sz;
    Xlatorp xxp = NULL;
    Xlatep xdata = NULL;

    if (!gip) { return NULL; }
    itype = gip->hdrp->instance_type;
    iname = gip->hdrp->instance_name;
    inarr = gip->inputs;
    outarr = gip->outputs;
    width = gip->width;
    num_gates = gip->num_gates;
    num_ins = gip->num_ins;
    num_outs = gip->num_outs;
    enable = gip->enable;
    tmodel = gip->tmodel;
    vector = has_vector_inputs(itype);
    for (i = 0; i < num_ins; i++) {
        add_input_pin(inarr[i]);
    }
    if (enable) { add_input_pin(enable); }
    if (is_tristate(itype) || is_tristate_array(itype)) {
        for (i = 0; i < num_outs; i++) {
            add_tristate_pin(outarr[i]);
            add_output_pin(outarr[i]);
        }
    } else {
        for (i = 0; i < num_outs; i++) {
            add_output_pin(outarr[i]);
        }
    }

    if (num_gates == 1) {
        char *inst_begin = NULL;
        simple_gate = is_gate(itype);
        tristate_gate = is_tristate(itype);
        if (!simple_gate && !tristate_gate) { return NULL; }
        add_tristate = FALSE;
        if (simple_gate) {
            xspice = find_xspice_for_delay(itype);
        } else if (tristate_gate) {
            xspice = find_xspice_for_delay(itype);
            if (!eq(itype, "buf3")) {
                add_tristate = TRUE;
            }
        }
        xxp = create_xlator();
        /* Now build the instance name and inputs section */
        if (vector) {
            startvec = "[";
            endvec = " ]";
        } else {
            startvec = "";
            endvec = "";
        }
        /* inputs */
        /* First calculate the space */
        sz = 0;
        for (i = 0; i < width; i++) {
            sz += strlen(inarr[i]) + 4; // Extra 4 spaces separating
        }
        input_buf = TMALLOC(char, sz);
        input_buf[0] = '\0';
        for (i = 0; i < width; i++) {
            sprintf(input_buf + strlen(input_buf), " %s", inarr[i]);
        }
        /* instance name and inputs */
        /* add the tristate enable if required on original */
        if (enable) {
            if (!add_tristate) {
                /* Warning: changing the format string affects input_buf sz */
                inst_begin = tprintf("a%s %s%s%s  %s",
                    iname, startvec, input_buf, endvec, enable);
            } else {
                /* Warning: changing the format string affects input_buf sz */
                inst_begin = tprintf("a%s %s%s%s",
                    iname, startvec, input_buf, endvec);
            }
        } else {
            /* Warning: changing the format string affects input_buf sz */
            inst_begin = tprintf("a%s %s%s%s",
                iname, startvec, input_buf, endvec);
        }
        tfree(input_buf);

        /* connector if required for tristate */
        connector = tprintf("a%s_%s", iname, outarr[0]);

        /* keep a copy of the model name of original gate */
        modelnm = tprintf("d_a%s_%s", iname, itype);

        if (!add_tristate) {
            char *instance_stmt = NULL;
            /* add output + model name  => translated instance */
            instance_stmt = tprintf("%s %s %s",
                                    inst_begin, outarr[0], modelnm);
            xdata = create_xlate_instance(instance_stmt,
                                          xspice, tmodel, modelnm);
            xxp = add_xlator(xxp, xdata);
            tfree(instance_stmt);
            if (simple_gate) {
                if (!gen_timing_model(tmodel, "ugate", xspice,
                    modelnm, xxp)) {
                    printf("WARNING unable to find tmodel %s for %s %s\n",
                        tmodel, modelnm, xspice);
                }
            } else { /* must be trstate gate buf3 */
                if (!gen_timing_model(tmodel, "utgate", xspice,
                    modelnm, xxp)) {
                    printf("WARNING unable to find tmodel %s for %s %s\n",
                        tmodel, modelnm, xspice);
                }
            }
        } else {
            char *new_model_nm = NULL;
            char *new_stmt = NULL;
            /*
             Use connector as original gate output and tristate input;
             tristate has original gate output and utgate delay;
             original gate has zero delay timing model.
             Complete the translation of the original gate adding
             the connector as output + model name.
            */
            new_stmt = tprintf("%s %s %s", inst_begin, connector, modelnm);
            xdata = create_xlate_instance(new_stmt, xspice, "", modelnm);
            xxp = add_xlator(xxp, xdata);
            tfree(new_stmt);
            /* new model statement e.g.   .model d_au2nand3 d_nand */
            new_stmt = tprintf(".model %s %s", modelnm, xspice);
            xdata = create_xlate_translated(new_stmt);
            xxp = add_xlator(xxp, xdata);
            tfree(new_stmt);
            /* now the added tristate */
            /* model name of added tristate */
            new_model_nm = tprintf("d_a%s_tribuf", iname);
            new_stmt = tprintf("a%s_tri %s %s %s %s",
                iname, connector, enable, outarr[0], new_model_nm);
            xdata = create_xlate_instance(new_stmt, "d_tristate",
                tmodel, new_model_nm);
            xxp = add_xlator(xxp, xdata);
            tfree(new_stmt);
            if (!gen_timing_model(tmodel, "utgate", "d_tristate",
                new_model_nm, xxp)) {
                printf("WARNING unable to find tmodel %s for %s %s\n",
                    tmodel, new_model_nm, xspice);
            }
            tfree(new_model_nm);
        }
        tfree(connector);
        tfree(modelnm);
        tfree(inst_begin);
        return xxp;

    } else {
        char *primary_model = NULL, *s1 = NULL, *s2 = NULL, *s3 = NULL;
        int ksave;
        /* arrays of gates */
        /* NOTE (n)and3a, (n)or3a, (n)xor3a types are not supported */
        simple_array = is_gate_array(itype);
        tristate_array = is_tristate_array(itype);
        add_tristate = FALSE;
        if (simple_array) {
            xspice = find_xspice_for_delay(itype);
        } else if (tristate_array) {
            xspice = find_xspice_for_delay(itype);
            if (eq("inv3a", itype)) {
                add_tristate = TRUE;
            } else if (!eq(itype, "buf3a")) {
                return NULL;
            }
        }
        xxp = create_xlator();
        k = 0;
        connector = NULL;
        if (vector) {
            startvec = "[";
            endvec = " ]";
        } else {
            startvec = "";
            endvec = "";
        }
        /* model name, same for all primary gates */
        primary_model = tprintf("d_a%s_%s", iname, itype); 
        for (i = 0; i < num_gates; i++) {
            /* inputs */
            /* First calculate the space */
            ksave = k;
            sz = 0;
            for (j = 0; j < width; j++) {
                /* inputs for primary gate */
                sz += strlen(inarr[k]) + 4; // Extra 4 spaces separating
                k++;
            }
            k = ksave;
            input_buf = TMALLOC(char, sz);
            input_buf[0] = '\0';
            for (j = 0; j < width; j++) {
                /* inputs for primary gate */
                /* Warning: changing the format string affects input_buf sz */
                sprintf(input_buf + strlen(input_buf), " %s", inarr[k]);
                k++;
            }
            /* create new instance name for primary gate */
            if (enable) {
                if (!add_tristate) {
                    s1 = tprintf("a%s_%d %s%s%s  %s",
                        iname, i, startvec, input_buf, endvec, enable);
                } else {
                    s1 = tprintf("a%s_%d %s%s%s",
                        iname, i, startvec, input_buf, endvec);
                    /* connector if required for tristate */
                    connector = tprintf("a%s_%d_%s", iname, i, outarr[i]);
                }
            } else {
                s1 = tprintf("a%s_%d %s%s%s",
                    iname, i, startvec, input_buf, endvec);
            }
            tfree(input_buf);
            /* output of primary gate */
            if (add_tristate) {
                s2 = tprintf(" %s %s", connector, primary_model);
            } else {
                s2 = tprintf(" %s %s", outarr[i], primary_model);
            }
            /* translated instance */
            s3 = tprintf("%s%s", s1, s2);

            if (add_tristate) {
                xdata = create_xlate_instance(s3, xspice, "", primary_model);
                xxp = add_xlator(xxp, xdata);
            } else {
                xdata = create_xlate_instance(s3, xspice, tmodel,
                    primary_model);
                xxp = add_xlator(xxp, xdata);
            }
            tfree(s1);
            tfree(s2);
            tfree(s3);

            if (!add_tristate) {
                if (tristate_array) {
                    if (i == 0 && !gen_timing_model(tmodel, "utgate",
                                            xspice, primary_model, xxp)) {
                        printf("WARNING unable to find tmodel %s for %s %s\n",
                            tmodel, primary_model, xspice);
                    }
                } else {
                    if (i == 0 && !gen_timing_model(tmodel, "ugate",
                                            xspice, primary_model, xxp)) {
                        printf("WARNING unable to find tmodel %s for %s %s\n",
                            tmodel, primary_model, xspice);
                    }
                }
            }

            if (add_tristate) {
                char *s1 = NULL, *modelnm = NULL;
                if (i == 0) {
                    /* Zero delay model for all original array instances */
                    s1 = tprintf(".model %s %s", primary_model, xspice); 
                    xdata = create_xlate_translated(s1);
                    xxp = add_xlator(xxp, xdata);
                    tfree(s1);
                }
                /* model name of added tristate */
                modelnm = tprintf("d_a%s_tribuf", iname);
                /*
                 instance name of added tristate, connector,
                 enable, original primary gate output, timing model.
                */
                s1 = tprintf("a%s_%d_tri %s %s %s %s", iname, i, connector,
                    enable, outarr[i], modelnm);
                xdata = create_xlate_instance(s1, "d_tristate",
                    tmodel, modelnm);
                xxp = add_xlator(xxp, xdata);
                tfree(s1);
                if (i == 0 && !gen_timing_model(tmodel, "utgate",
                                        "d_tristate", modelnm, xxp)) {
                    printf("WARNING unable to find tmodel %s for %s %s\n",
                        tmodel, modelnm, "d_tristate");
                }
                tfree(modelnm);
                tfree(connector);
            }
        }
        tfree(primary_model);
        return xxp;
    }
    return NULL;
}

static void extract_model_param(char *rem, char *pname, char *buf)
{
    char *p1, *p2, *p3;

    p1 = strstr(rem, pname);
    if (p1) {
        p2 = strchr(p1, '=');
        if (isspace(p2[1])) {
            p3 = skip_ws(&p2[1]);
        } else {
            p3 = &p2[1];
        }
        while (!isspace(p3[0]) && p3[0] != ')') {
            *buf = p3[0];
            buf++;
            p3++;
        }
        *buf = '\0';
    } else {
        buf[0] = '\0';
    }
}

static void delete_timing_data(struct timing_data *tdp)
{
    if (!tdp) { return; }
    if (tdp->min) { tfree(tdp->min); }
    if (tdp->typ) { tfree(tdp->typ); }
    if (tdp->max) { tfree(tdp->max); }
    if (tdp->ave) { tfree(tdp->ave); }
    tfree(tdp);
    return;
}

static struct timing_data *create_min_typ_max(char *prefix, char *rem)
{
    char *mntymxstr;
    char *buf, *bufsave;
    size_t n = strlen(prefix) + 4;
    struct timing_data *tdp = NULL;

    tdp = TMALLOC(struct timing_data, 1);
    mntymxstr = TMALLOC(char, n);
    buf = TMALLOC(char, strlen(rem) + 1);
    bufsave = buf;

    tdp->ave = NULL;
    tdp->estimate = EST_UNK;

    strcpy(mntymxstr, prefix);
    strcat(mntymxstr, "mn=");
    extract_model_param(rem, mntymxstr, buf);
    tdp->min = NULL;
    if (bufsave[0]) {
        tdp->min = TMALLOC(char, strlen(bufsave) + 1);
        (void) memcpy(tdp->min, bufsave, strlen(bufsave) + 1);
    }

    buf = bufsave;
    strcpy(mntymxstr, prefix);
    strcat(mntymxstr, "ty=");
    extract_model_param(rem, mntymxstr, buf);
    tdp->typ = NULL;
    if (bufsave[0]) {
        tdp->typ = TMALLOC(char, strlen(bufsave) + 1);
        (void) memcpy(tdp->typ, bufsave, strlen(bufsave) + 1);
    }

    buf = bufsave;
    strcpy(mntymxstr, prefix);
    strcat(mntymxstr, "mx=");
    extract_model_param(rem, mntymxstr, buf);
    tdp->max = NULL;
    if (bufsave[0]) {
        tdp->max = TMALLOC(char, strlen(bufsave) + 1);
        (void) memcpy(tdp->max, bufsave, strlen(bufsave) + 1);
    }

    tfree(bufsave);
    tfree(mntymxstr);
    return tdp;
}

static void estimate_typ(struct timing_data *tdp)
{
    char *tmpmax = NULL, *tmpmin = NULL;
    char *min, *typ, *max;
    float valmin, valmax, average;
    char *units1, *units2;

    if (!tdp) { return; }
    min = tdp->min;
    typ = tdp->typ;
    max = tdp->max;

    if (typ && strlen(typ) > 0 && typ[0] != '-') {
        tdp->estimate = EST_TYP;
        return;
    }
    if (max && strlen(max) > 0 && max[0] != '-') {
        tmpmax = max;
    }
    if (min && strlen(min) > 0 && min[0] != '-') {
        tmpmin = min;
    }
    if (tmpmin && tmpmax) {
        if (strlen(tmpmin) > 0 && strlen(tmpmax) > 0) {
            valmin = strtof(tmpmin, &units1);
            valmax = strtof(tmpmax, &units2);
            average = (valmin + valmax) / 2.0;
            tdp->ave = tprintf("%.2f%s", average, units2);
            if (!eq(units1, units2)) {
                printf("WARNING units do not match\n");
            }
            tdp->estimate = EST_AVE;
            return;
        }
    } else if (tmpmax && strlen(tmpmax) > 0) {
        tdp->estimate = EST_MAX;
        return;
    } else if (tmpmin && strlen(tmpmin) > 0) {
        tdp->estimate = EST_MIN;
        return;
    } else {
        tdp->estimate = EST_UNK;
        return;
    }
    tdp->estimate = EST_UNK;
    return;
}

static char *get_estimate(struct timing_data *tdp)
{
    /*
      Call after estimate_typ.
      Don't call delete_timing_data until you have copied
      or finished with this return value.
    */
    if (!tdp) { return NULL; }
    switch (tdp->estimate) {
        case EST_MIN: return tdp->min;
        case EST_TYP: return tdp->typ;
        case EST_MAX: return tdp->max;
        case EST_AVE: return tdp->ave;
        default: break;
    }
    return NULL;
}

/* NOTE
  The get_delays_...() functions calculate estimates of typical delays
  from the Pspice ugate, utgate, ueff, and ugff timing models.
  These functions are called from u_process_model(), and the delay strings
  are added to the timing model Xlator by add_delays_to_model_xlator().
*/
static char *get_delays_ugate(char *rem, char *d_name)
{
    char *rising, *falling, *delays = NULL;
    struct timing_data *tdp1, *tdp2;

    tdp1 = create_min_typ_max("tplh", rem);
    estimate_typ(tdp1);
    rising = get_estimate(tdp1);
    tdp2 = create_min_typ_max("tphl", rem);
    estimate_typ(tdp2);
    falling = get_estimate(tdp2);
    if (rising && falling) {
        if (strlen(rising) > 0 && strlen(falling) > 0) {
            delays = tprintf("(rise_delay = %s fall_delay = %s)",
                            rising, falling);
        }
    }
    delete_timing_data(tdp1);
    delete_timing_data(tdp2);
    return delays;
}

static char *get_delays_utgate(char *rem, char *d_name)
{
    /* Return estimate of tristate delay (delay = val3) */
    char *rising, *falling, *delays = NULL;
    struct timing_data *tdp1, *tdp2;

    tdp1 = create_min_typ_max("tplh", rem);
    estimate_typ(tdp1);
    rising = get_estimate(tdp1);
    tdp2 = create_min_typ_max("tphl", rem);
    estimate_typ(tdp2);
    falling = get_estimate(tdp2);
    if (rising && falling) {
        if (strlen(rising) > 0 && strlen(falling) > 0) {
            delays = tprintf("(delay = %s)", rising);
        }
    }
    delete_timing_data(tdp1);
    delete_timing_data(tdp2);
    return delays;
}

static char *get_delays_ueff(char *rem, char *d_name)
{
    char *delays = NULL;
    char *clkqrise, *clkqfall, *pcqrise, *pcqfall;
    char *clkd, *setd, *resetd;
    struct timing_data *tdp1, *tdp2, *tdp3, *tdp4;

    tdp1 = create_min_typ_max("tpclkqlh", rem);
    estimate_typ(tdp1);
    clkqrise = get_estimate(tdp1);
    tdp2 = create_min_typ_max("tpclkqhl", rem);
    estimate_typ(tdp2);
    clkqfall = get_estimate(tdp2);
    tdp3 = create_min_typ_max("tppcqlh", rem);
    estimate_typ(tdp3);
    pcqrise = get_estimate(tdp3);
    tdp4 = create_min_typ_max("tppcqhl", rem);
    estimate_typ(tdp4);
    pcqfall = get_estimate(tdp4);
    clkd = NULL;
    if (clkqrise && strlen(clkqrise) > 0) {
        clkd = clkqrise;
    } else if (clkqfall && strlen(clkqfall) > 0) {
        clkd = clkqfall;
    }
    setd = NULL;
    resetd = NULL;
    if (pcqrise && strlen(pcqrise) > 0) {
        setd = resetd = pcqrise;
    } else if (pcqfall && strlen(pcqfall) > 0) {
        setd = resetd = pcqfall;
    }
    if (clkd && setd) {
        delays = tprintf("(clk_delay = %s "
            "set_delay = %s reset_delay = %s "
            "rise_delay = 1.0ns fall_delay = 2.0ns)",
            clkd, setd, resetd);
    } else if (clkd) {
        delays = tprintf("(clk_delay = %s "
            "rise_delay = 1.0ns fall_delay = 2.0ns)",
            clkd);
    } else if (setd) {
        delays = tprintf("(set_delay = %s reset_delay = %s "
            "rise_delay = 1.0ns fall_delay = 2.0ns)",
            setd, resetd);
    } else {
        delays = tprintf("(rise_delay = 1.0ns fall_delay = 2.0ns)");
    }
    delete_timing_data(tdp1);
    delete_timing_data(tdp2);
    delete_timing_data(tdp3);
    delete_timing_data(tdp4);
    return delays;
}

static char *get_delays_ugff(char *rem, char *d_name)
{
    char *delays = NULL, *dname;
    char *tpdqlh, *tpdqhl, *tpgqlh, *tpgqhl, *tppcqlh, *tppcqhl;
    char *d_delay, *enab, *setd, *resetd;
    char *s1, *s2;
    struct timing_data *tdp1, *tdp2, *tdp3, *tdp4, *tdp5, *tdp6;

    if (eq(d_name, "d_dlatch")) {
        dname = "data_delay";
    } else if (eq(d_name, "d_srlatch")) {
        dname = "sr_delay";
    } else {
        return NULL;
    }
    tdp1 = create_min_typ_max("tpdqlh", rem);
    estimate_typ(tdp1);
    tpdqlh = get_estimate(tdp1);
    tdp2 = create_min_typ_max("tpdqhl", rem);
    estimate_typ(tdp2);
    tpdqhl = get_estimate(tdp2);
    tdp3 = create_min_typ_max("tpgqlh", rem);
    estimate_typ(tdp3);
    tpgqlh = get_estimate(tdp3);
    tdp4 = create_min_typ_max("tpgqhl", rem);
    estimate_typ(tdp4);
    tpgqhl = get_estimate(tdp4);
    tdp5 = create_min_typ_max("tppcqlh", rem);
    estimate_typ(tdp5);
    tppcqlh = get_estimate(tdp5);
    tdp6 = create_min_typ_max("tppcqhl", rem);
    estimate_typ(tdp6);
    tppcqhl = get_estimate(tdp6);
    d_delay = NULL;
    if (tpdqlh && strlen(tpdqlh) > 0) {
        d_delay = tpdqlh;
    } else if (tpdqhl && strlen(tpdqhl) > 0) {
        d_delay = tpdqhl;
    }
    enab = NULL;
    if (tpgqlh && strlen(tpgqlh) > 0) {
        enab = tpgqlh;
    } else if (tpgqhl && strlen(tpgqhl) > 0) {
        enab = tpgqhl;
    }
    s1 = NULL;
    if (enab) {
        if (d_delay) {
            s1 = tprintf("%s = %s enable_delay = %s ",
                dname, d_delay, enab);
        } else {
            s1 = tprintf("enable_delay = %s ", enab);
        }
    } else {
        if (d_delay) {
            s1 = tprintf("%s = %s ", dname, d_delay);
        }
    }
    setd = NULL;
    resetd = NULL;
    if (tppcqlh && strlen(tppcqlh) > 0) {
        setd = resetd = tppcqlh;
    } else if (tppcqhl && strlen(tppcqhl) > 0) {
        setd = resetd = tppcqhl;
    }
    if (setd) {
        s2 = tprintf("set_delay = %s reset_delay = %s "
            "rise_delay = 1.0ns fall_delay = 2.0ns)",
            setd, resetd);
    } else {
        s2 = tprintf("rise_delay = 1.0ns fall_delay = 2.0ns)");
    }
    if (s1) {
        delays = tprintf("(%s%s", s1, s2);
        tfree(s1);
    } else {
        delays = tprintf("(%s", s2);
    }
    tfree(s2);
    delete_timing_data(tdp1);
    delete_timing_data(tdp2);
    delete_timing_data(tdp3);
    delete_timing_data(tdp4);
    delete_timing_data(tdp5);
    delete_timing_data(tdp6);
    return delays;
}

static BOOL u_process_model(char *nline, char *original,
                          char *newname, char *xspice)
{
    char *tok, *remainder, *delays = NULL, *utype, *tmodel;
    BOOL retval = TRUE;

    /* .model */
    tok = strtok(nline, " \t");
    /* model name */
    tok = strtok(NULL, " \t");
    tmodel = TMALLOC(char, strlen(tok) + 1);
    memcpy(tmodel, tok, strlen(tok) + 1);
    /* model utype */
    tok = strtok(NULL, " \t(");
    utype = TMALLOC(char, strlen(tok) + 1);
    memcpy(utype, tok, strlen(tok) + 1);

    /* delay info */
    remainder = strchr(original, '(');
    if (remainder) {
        if (eq(utype, "ugate")) {
            delays = get_delays_ugate(remainder, xspice);
            add_delays_to_model_xlator((delays ? delays : ""),
                utype, "", tmodel);
            if (delays) { tfree(delays); }
        } else if (eq(utype, "utgate")) {
            delays = get_delays_utgate(remainder, xspice);
            add_delays_to_model_xlator((delays ? delays : ""),
                utype, "", tmodel);
            if (delays) { tfree(delays); }
        } else if (eq(utype, "ueff")) {
            delays = get_delays_ueff(remainder, xspice);
            add_delays_to_model_xlator((delays ? delays : ""),
                utype, "", tmodel);
            if (delays) { tfree(delays); }
        } else if (eq(utype, "ugff")) {
            delays = get_delays_ugff(remainder, "d_dlatch");
            add_delays_to_model_xlator((delays ? delays : ""),
                utype, "d_dlatch", tmodel);
            if (delays) { tfree(delays); }
            delays = get_delays_ugff(remainder, "d_srlatch");
            add_delays_to_model_xlator((delays ? delays : ""),
                utype, "d_srlatch", tmodel);
            if (delays) { tfree(delays); }
        } else {
            retval = FALSE;
            delays = NULL;
        }
    } else {
            retval = FALSE;
    }
    tfree(tmodel);
    tfree(utype);
    return retval;
}

static struct dff_instance *add_dff_inout_timing_model(
    struct instance_hdr *hdr, char *start)
{
    char *tok, *copyline;
    char *name, **arrp;
    int i, num_gates = hdr->num1;
    struct dff_instance *dffip = NULL;

    dffip = create_dff_instance(hdr);
    dffip->num_gates = num_gates;
    copyline = TMALLOC(char, strlen(start) + 1);
    (void) memcpy(copyline, start, strlen(start) + 1);
    /* prebar, clrbar, clk */
    tok = strtok(copyline, " \t");
    dffip->prebar = TMALLOC(char, strlen(tok) + 1);
    (void) memcpy(dffip->prebar, tok, strlen(tok) + 1);
    tok = strtok(NULL, " \t");
    dffip->clrbar = TMALLOC(char, strlen(tok) + 1);
    (void) memcpy(dffip->clrbar, tok, strlen(tok) + 1);
    tok = strtok(NULL, " \t");
    dffip->clk = TMALLOC(char, strlen(tok) + 1);
    (void) memcpy(dffip->clk, tok, strlen(tok) + 1);
    /* d inputs */
    dffip->d_in = TMALLOC(char *, num_gates);
    arrp = dffip->d_in;
    for (i = 0; i < num_gates; i++) {
        tok = strtok(NULL, " \t");
        name = TMALLOC(char, strlen(tok) + 1);
        (void) memcpy(name, tok, strlen(tok) + 1);
        arrp[i] = name;
    }
    /* q_out outputs */
    dffip->q_out = TMALLOC(char *, num_gates);
    arrp = dffip->q_out;
    for (i = 0; i < num_gates; i++) {
        tok = strtok(NULL, " \t");
        name = TMALLOC(char, strlen(tok) + 1);
        (void) memcpy(name, tok, strlen(tok) + 1);
        arrp[i] = name;
    }
    /* qb_out outputs */
    dffip->qb_out = TMALLOC(char *, num_gates);
    arrp = dffip->qb_out;
    for (i = 0; i < num_gates; i++) {
        tok = strtok(NULL, " \t");
        name = TMALLOC(char, strlen(tok) + 1);
        (void) memcpy(name, tok, strlen(tok) + 1);
        arrp[i] = name;
    }
    /* timing model */
    tok = strtok(NULL, " \t");
    dffip->tmodel = TMALLOC(char, strlen(tok) + 1);
    (void) memcpy(dffip->tmodel, tok, strlen(tok) + 1);
    tfree(copyline);

    /* Reject incompatible inputs */
    arrp = dffip->d_in;
    for (i = 0; i < num_gates; i++) {
        if (strncmp(arrp[i], "$d_", 3) == 0) {
            tfree(dffip);
            return NULL;
        }
    }
    if (eq(dffip->prebar, "$d_lo") || eq(dffip->prebar, "$d_nc")) {
        tfree(dffip);
        return NULL;
    }
    if (eq(dffip->clrbar, "$d_lo") || eq(dffip->clrbar, "$d_nc")) {
        tfree(dffip);
        return NULL;
    }

    return dffip;
}

static struct dltch_instance *add_dltch_inout_timing_model(
    struct instance_hdr *hdr, char *start)
{
    char *tok, *copyline;
    char *name, **arrp;
    int i, num_gates = hdr->num1;
    struct dltch_instance *dlp = NULL;

    dlp = create_dltch_instance(hdr);
    dlp->num_gates = num_gates;
    copyline = TMALLOC(char, strlen(start) + 1);
    (void) memcpy(copyline, start, strlen(start) + 1);
    /* prebar, clrbar, clk */
    tok = strtok(copyline, " \t");
    dlp->prebar = TMALLOC(char, strlen(tok) + 1);
    (void) memcpy(dlp->prebar, tok, strlen(tok) + 1);
    tok = strtok(NULL, " \t");
    dlp->clrbar = TMALLOC(char, strlen(tok) + 1);
    (void) memcpy(dlp->clrbar, tok, strlen(tok) + 1);
    tok = strtok(NULL, " \t");
    dlp->gate = TMALLOC(char, strlen(tok) + 1);
    (void) memcpy(dlp->gate, tok, strlen(tok) + 1);
    /* d inputs */
    dlp->d_in = TMALLOC(char *, num_gates);
    arrp = dlp->d_in;
    for (i = 0; i < num_gates; i++) {
        tok = strtok(NULL, " \t");
        name = TMALLOC(char, strlen(tok) + 1);
        (void) memcpy(name, tok, strlen(tok) + 1);
        arrp[i] = name;
    }
    /* q_out outputs */
    dlp->q_out = TMALLOC(char *, num_gates);
    arrp = dlp->q_out;
    for (i = 0; i < num_gates; i++) {
        tok = strtok(NULL, " \t");
        name = TMALLOC(char, strlen(tok) + 1);
        (void) memcpy(name, tok, strlen(tok) + 1);
        arrp[i] = name;
    }
    /* qb_out outputs */
    dlp->qb_out = TMALLOC(char *, num_gates);
    arrp = dlp->qb_out;
    for (i = 0; i < num_gates; i++) {
        tok = strtok(NULL, " \t");
        name = TMALLOC(char, strlen(tok) + 1);
        (void) memcpy(name, tok, strlen(tok) + 1);
        arrp[i] = name;
    }
    /* timing model */
    tok = strtok(NULL, " \t");
    dlp->tmodel = TMALLOC(char, strlen(tok) + 1);
    (void) memcpy(dlp->tmodel, tok, strlen(tok) + 1);
    tfree(copyline);

    /* Reject incompatible inputs */
    arrp = dlp->d_in;
    for (i = 0; i < num_gates; i++) {
        if (strncmp(arrp[i], "$d_", 3) == 0) {
            tfree(dlp);
            return NULL;
        }
    }
    if (eq(dlp->prebar, "$d_lo") || eq(dlp->prebar, "$d_nc")) {
        tfree(dlp);
        return NULL;
    }
    if (eq(dlp->clrbar, "$d_lo") || eq(dlp->clrbar, "$d_nc")) {
        tfree(dlp);
        return NULL;
    }
    return dlp;
}

static struct jkff_instance *add_jkff_inout_timing_model(
    struct instance_hdr *hdr, char *start)
{
    char *tok, *copyline;
    char *name, **arrp, **arrpk;
    int i, num_gates = hdr->num1;
    struct jkff_instance *jkffip = NULL;

    jkffip = create_jkff_instance(hdr);
    jkffip->num_gates = num_gates;
    copyline = TMALLOC(char, strlen(start) + 1);
    (void) memcpy(copyline, start, strlen(start) + 1);
    /* prebar, clrbar, clkbar */
    tok = strtok(copyline, " \t");
    jkffip->prebar = TMALLOC(char, strlen(tok) + 1);
    (void) memcpy(jkffip->prebar, tok, strlen(tok) + 1);
    tok = strtok(NULL, " \t");
    jkffip->clrbar = TMALLOC(char, strlen(tok) + 1);
    (void) memcpy(jkffip->clrbar, tok, strlen(tok) + 1);
    tok = strtok(NULL, " \t");
    jkffip->clkbar = TMALLOC(char, strlen(tok) + 1);
    (void) memcpy(jkffip->clkbar, tok, strlen(tok) + 1);
    /* j inputs */
    jkffip->j_in = TMALLOC(char *, num_gates);
    arrp = jkffip->j_in;
    for (i = 0; i < num_gates; i++) {
        tok = strtok(NULL, " \t");
        name = TMALLOC(char, strlen(tok) + 1);
        (void) memcpy(name, tok, strlen(tok) + 1);
        arrp[i] = name;
    }
    /* k inputs */
    jkffip->k_in = TMALLOC(char *, num_gates);
    arrp = jkffip->k_in;
    for (i = 0; i < num_gates; i++) {
        tok = strtok(NULL, " \t");
        name = TMALLOC(char, strlen(tok) + 1);
        (void) memcpy(name, tok, strlen(tok) + 1);
        arrp[i] = name;
    }
    /* q_out outputs */
    jkffip->q_out = TMALLOC(char *, num_gates);
    arrp = jkffip->q_out;
    for (i = 0; i < num_gates; i++) {
        tok = strtok(NULL, " \t");
        name = TMALLOC(char, strlen(tok) + 1);
        (void) memcpy(name, tok, strlen(tok) + 1);
        arrp[i] = name;
    }
    /* qb_out outputs */
    jkffip->qb_out = TMALLOC(char *, num_gates);
    arrp = jkffip->qb_out;
    for (i = 0; i < num_gates; i++) {
        tok = strtok(NULL, " \t");
        name = TMALLOC(char, strlen(tok) + 1);
        (void) memcpy(name, tok, strlen(tok) + 1);
        arrp[i] = name;
    }
    /* timing model */
    tok = strtok(NULL, " \t");
    jkffip->tmodel = TMALLOC(char, strlen(tok) + 1);
    (void) memcpy(jkffip->tmodel, tok, strlen(tok) + 1);
    tfree(copyline);

    /* Reject incompatible inputs */
    arrp = jkffip->j_in;
    arrpk = jkffip->k_in;
    for (i = 0; i < num_gates; i++) {
        if (strncmp(arrp[i], "$d_", 3) == 0 ||
            strncmp(arrpk[i], "$d_", 3) == 0) {
            tfree(jkffip);
            return NULL;
        }
    }
    if (eq(jkffip->prebar, "$d_lo") || eq(jkffip->prebar, "$d_nc")) {
        tfree(jkffip);
        return NULL;
    }
    if (eq(jkffip->clrbar, "$d_lo") || eq(jkffip->clrbar, "$d_nc")) {
        tfree(jkffip);
        return NULL;
    }
    return jkffip;
}

static struct compound_instance *add_compound_inout_timing_model(
    struct instance_hdr *hdr, char *start)
{
    char *tok, *copyline, *itype  = hdr->instance_type, *name;
    int i, j, k, n1 =hdr->num1, n2 = hdr->num2, inwidth, numgates;
    struct compound_instance *compi;
    char **inarr;
    BOOL first = TRUE;

    if (is_compound_gate(itype)) {
         inwidth = n1;
         numgates = n2;
    } else {
        return NULL;
    }
    compi = create_compound_instance(hdr);
    compi->num_gates = numgates;
    compi->width = inwidth;
    compi->num_ins = numgates * inwidth;
    copyline = TMALLOC(char, strlen(start) + 1);
    (void) memcpy(copyline, start, strlen(start) + 1);
    inarr = TMALLOC(char *, compi->num_ins);
    compi->inputs = inarr;
    /* all the inputs, inwidth inputs per gate */
    k = 0;
    for (i = 0; i < numgates; i++) {
        for (j = 0; j < inwidth; j++) {
            if (first) {
                tok = strtok(copyline, " \t");
                first = FALSE;
            } else {
                tok = strtok(NULL, " \t");
            }
            name = TMALLOC(char, strlen(tok) + 1);
            (void) memcpy(name, tok, strlen(tok) + 1);
            inarr[k] = name;
            k++;
        }
    }
    /* one output */
    tok = strtok(NULL, " \t");
    name = TMALLOC(char, strlen(tok) + 1);
    (void) memcpy(name, tok, strlen(tok) + 1);
    compi->output = name;
    /* timing model */
    tok = strtok(NULL, " \t");
    name = TMALLOC(char, strlen(tok) + 1);
    (void) memcpy(name, tok, strlen(tok) + 1);
    compi->tmodel = name;
    tfree(copyline);
    return compi;
}

static struct gate_instance *add_array_inout_timing_model(
    struct instance_hdr *hdr, char *start)
{
    char *tok, *copyline, *itype  = hdr->instance_type;
    BOOL first = TRUE, tristate = FALSE;
    int i, j, k, n1 =hdr->num1, n2 = hdr->num2, inwidth, numgates;
    struct gate_instance *gip = NULL;
    char **inarr = NULL, **outarr = NULL, *name;

    if (is_tristate_buf_array(itype)) {
        inwidth = 1;
        numgates = n1;
        tristate = TRUE;
    } else if (is_buf_gate_array(itype)) {
        inwidth = 1;
        numgates = n1;
    } else if (is_vector_gate_array(itype)) {
        inwidth = n1;
        numgates = n2;
    } else if (is_tristate_vector_array(itype)) {
        inwidth = n1;
        numgates = n2;
        tristate = TRUE;
    } else if (is_xor_gate_array(itype)) {
        inwidth = 2;
        numgates = n1;
    } else if (is_tristate_xor_array(itype)) {
        inwidth = 2;
        numgates = n1;
        tristate = TRUE;
    } else {
        return NULL;
    }
    gip = create_gate_instance(hdr);
    gip->num_gates = numgates;
    gip->width = inwidth;
    gip->num_ins = numgates * inwidth;
    gip->num_outs = numgates;
    copyline = TMALLOC(char, strlen(start) + 1);
    (void) memcpy(copyline, start, strlen(start) + 1);
    /*
     numgates gates, each gate has inwidth inputs and 1 output
     inputs first
    */
    inarr = TMALLOC(char *, gip->num_ins);
    gip->inputs = inarr;
    k = 0;
    for (i = 0; i < numgates; i++) {
        for (j = 0; j < inwidth; j++) {
            if (first) {
                tok = strtok(copyline, " \t");
                first = FALSE;
            } else {
                tok = strtok(NULL, " \t");
            }
            name = TMALLOC(char, strlen(tok) + 1);
            (void) memcpy(name, tok, strlen(tok) + 1);
            inarr[k] = name;
            k++;
        }
    }
    /* enable for tristate */
    if (tristate) {
        tok = strtok(NULL, " \t");
        name = TMALLOC(char, strlen(tok) + 1);
        (void) memcpy(name, tok, strlen(tok) + 1);
        gip->enable = name;
    }
    /* outputs next */
    outarr = TMALLOC(char *, numgates);
    gip->outputs = outarr;
    for (i = 0; i < numgates; i++) {
        tok = strtok(NULL, " \t");
        name = TMALLOC(char, strlen(tok) + 1);
        (void) memcpy(name, tok, strlen(tok) + 1);
        outarr[i] = name;
    }
    /* timing model last */
    tok = strtok(NULL, " \t");
    name = TMALLOC(char, strlen(tok) + 1);
    (void) memcpy(name, tok, strlen(tok) + 1);
    gip->tmodel = name;
    tfree(copyline);
    return gip;
}

static struct gate_instance *add_gate_inout_timing_model(
    struct instance_hdr *hdr, char *start)
{
    char *tok, *copyline, *itype  = hdr->instance_type;
    int i, n1 = hdr->num1, inwidth;
    BOOL first = TRUE, tristate = FALSE;
    struct gate_instance *gip = NULL;
    char **inarr = NULL, **outarr = NULL, *name;

    if (is_vector_gate(itype)) {
        inwidth = n1;
    } else if (is_vector_tristate(itype)) {
        inwidth = n1;
        tristate = TRUE;
    } else if (is_buf_gate(itype)) {
        inwidth = 1;
    } else if (is_buf_tristate(itype)) {
        inwidth = 1;
        tristate = TRUE;
    } else if (is_xor_gate(itype)) {
        inwidth = 2;
    } else if (is_xor_tristate(itype)) {
        inwidth = 2;
        tristate = TRUE;
    } else {
        return NULL;
    }
    gip = create_gate_instance(hdr);
    gip->num_gates = 1;
    gip->width = inwidth;
    gip->num_ins = inwidth;
    gip->num_outs = 1;
    copyline = TMALLOC(char, strlen(start) + 1);
    (void) memcpy(copyline, start, strlen(start) + 1);
    /* inputs */
    inarr = TMALLOC(char *, gip->num_ins);
    gip->inputs = inarr;
    for (i = 0; i < inwidth; i++) {
        if (first) {
            tok = strtok(copyline, " \t");
            first = FALSE;
        } else {
            tok = strtok(NULL, " \t");
        }
        name = TMALLOC(char, strlen(tok) + 1);
        (void) memcpy(name, tok, strlen(tok) + 1);
        inarr[i] = name;
    }
    /* enable for tristate */
    if (tristate) {
        tok = strtok(NULL, " \t");
        name = TMALLOC(char, strlen(tok) + 1);
        (void) memcpy(name, tok, strlen(tok) + 1);
        gip->enable = name;
    }
    /* output */
    outarr = TMALLOC(char *, gip->num_outs);
    gip->outputs = outarr;
    tok = strtok(NULL, " \t");
    name = TMALLOC(char, strlen(tok) + 1);
    (void) memcpy(name, tok, strlen(tok) + 1);
    outarr[0] = name;
    /* timing model last */
    tok = strtok(NULL, " \t");
    name = TMALLOC(char, strlen(tok) + 1);
    (void) memcpy(name, tok, strlen(tok) + 1);
    gip->tmodel = name;
    tfree(copyline);
    return gip;
}

static char *skip_past_words(char *start, int count)
{
    char *p1;
    int i;

    if (count < 1) { return start; }
    p1 = start;
    p1 = skip_ws(p1);
    for (i = 0; i < count; i++) {
        p1 = skip_non_ws(p1);
        p1 = skip_ws(p1);
    }
    return p1;
}

static Xlatorp translate_pull(struct instance_hdr *hdr, char *start)
{
    char *itype, *xspice, *iname, *newline = NULL, *tok;
    char *model_name, *inst_stmt, *model_stmt;
    int i, numpulls;
    Xlatorp xp = NULL;
    Xlatep xdata = NULL;

    itype = hdr->instance_type;
    iname = hdr->instance_name;
    numpulls = hdr->num1;
    xp = create_xlator();
    /* pull devices do not have a timing model, just need the xspice name */
    xspice = find_xspice_for_delay(itype);
    newline = TMALLOC(char, strlen(start) + 1);
    (void) memcpy(newline, start, strlen(start) + 1);
    model_name = tprintf("d_%s_%s", iname, itype);
    for (i = 0; i < numpulls; i++) {
        if (i == 0) {
            tok = strtok(newline, " \t");
        } else {
            tok = strtok(NULL, " \t");
        }
        inst_stmt = tprintf("a%s_%d %s %s", iname, i, tok, model_name);
        xdata = create_xlate_translated(inst_stmt);
        xp = add_xlator(xp, xdata);
        tfree(inst_stmt);
    }
    model_stmt = tprintf(".model %s %s(load = 1pf)", model_name, xspice);
    xdata = create_xlate_translated(model_stmt);
    xp = add_xlator(xp, xdata);
    tfree(model_stmt);
    tfree(model_name);
    tfree(newline);
    delete_instance_hdr(hdr);
    return xp;
}

static Xlatorp translate_ff_latch(struct instance_hdr *hdr, char *start)
{
    /* If OK return Xlatorp else return NULL */
    char *itype;
    struct dff_instance *dffp = NULL;
    struct jkff_instance *jkffp = NULL;
    struct dltch_instance *dltchp = NULL;
    Xlatorp xp;

    itype = hdr->instance_type;
    if (eq(itype, "dff")) {
        dffp = add_dff_inout_timing_model(hdr, start);
        if (dffp) {
            xp = gen_dff_instance(dffp);
            delete_dff_instance(dffp);
            return xp;
        }
    } else if (eq(itype, "jkff")) {
        jkffp = add_jkff_inout_timing_model(hdr, start);
        if (jkffp) {
            xp = gen_jkff_instance(jkffp);
            delete_jkff_instance(jkffp);
            return xp;
        }
    } else if (eq(itype, "dltch")) {
        dltchp = add_dltch_inout_timing_model(hdr, start);
        if (dltchp) {
            xp = gen_dltch_instance(dltchp);
            delete_dltch_instance(dltchp);
            return xp;
        }
    } else {
        return NULL;
    }
    return NULL;
}

static Xlatorp translate_gate(struct instance_hdr *hdr, char *start)
{
    /* If OK return Xlatorp else return NULL */
    char *itype;
    struct gate_instance *igatep;
    struct compound_instance *compi;
    Xlatorp xp;

    itype = hdr->instance_type;
    if (is_gate(itype) || is_tristate(itype)) {
        igatep = add_gate_inout_timing_model(hdr, start);
        if (igatep) {
            xp = gen_gate_instance(igatep);
            delete_gate_instance(igatep);
            return xp;
        }
    } else if (is_gate_array(itype) || is_tristate_array(itype)) {
        igatep = add_array_inout_timing_model(hdr, start);
        if (igatep) {
            xp = gen_gate_instance(igatep);
            delete_gate_instance(igatep);
            return xp;
        }
    } else if (is_compound_gate(itype)) {
        compi = add_compound_inout_timing_model(hdr, start);
        if (compi) {
            xp = gen_compound_instance(compi);
            delete_compound_instance(compi);
            return xp;
        }
    } else {
        return NULL;
    }
    return NULL;
}

BOOL u_check_instance(char *line)
{
    /*
     Check to see if the U* instance is a type which can be translated.
     Return TRUE if it can be translated
    */
    char *xspice, *itype;
    struct instance_hdr *hdr = create_instance_header(line);

    itype = hdr->instance_type;
    xspice = find_xspice_for_delay(itype);
    delete_instance_hdr(hdr);
    if (!xspice) {
        return FALSE;
    } else {
        return TRUE;
    }
}

/* NOTE
  The input nline is expected to be a card in the deck which contains
  a Pspice u* instance statement with all the '+' continuations added
  minus the '+'.
*/
BOOL u_process_instance(char *nline)
{
    /* Return TRUE if ok */
    char *p1, *itype, *xspice;
    struct instance_hdr *hdr = create_instance_header(nline);
    Xlatorp xp = NULL;

    itype = hdr->instance_type;
    xspice = find_xspice_for_delay(itype);
    if (!xspice) {
        delete_instance_hdr(hdr);
        return FALSE;
    }
    // printf("iname %s itype %s\n", hdr->instance_name, itype);
    /* Skip past instance name, type, pwr, gnd */
    p1 = skip_past_words(nline, 4);
    if (is_gate(itype) || is_gate_array(itype)) {
        xp = translate_gate(hdr, p1);
    } else if (is_tristate(itype) || is_tristate_array(itype)) {
        xp = translate_gate(hdr, p1);
    } else if (is_compound_gate(itype)) {
        xp = translate_gate(hdr, p1);
    } else if (eq(itype, "dff") || eq(itype, "jkff") || eq(itype, "dltch")) {
        xp = translate_ff_latch(hdr, p1);
    } else if (eq(itype, "pullup") || eq(itype, "pulldn")) {
        xp = translate_pull(hdr, p1);
    } else {
        delete_instance_hdr(hdr);
        return FALSE;
    }
    if (xp) {
        append_xlator(translated_p, xp);
#ifdef TRACE
        interpret_xlator(xp, TRUE);
#endif
        delete_xlator(xp);
        return TRUE;
    } else {
        return FALSE;
    }
}

/* NOTE
  The input line is expected to be a card in the deck which contains
  a Pspice .model timing model statement with all the '+' continuations
  added minus the '+'.
*/
BOOL u_process_model_line(char *line)
{
    /* Translate a .model line to find the delays */
    /* Return TRUE if ok */
    char *newline;
    BOOL retval;
    size_t n = strlen(line) - 1;

    if (n > 0 && line[n] == '\n') line[n] = '\0';
    if (strncmp(line, ".model ", strlen(".model ")) == 0) {
        newline = TMALLOC(char, strlen(line) + 1);
        (void) memcpy(newline, line, strlen(line) + 1);
        retval = u_process_model(newline, line, "model_new_name", "d_xspice");
        tfree(newline);
        return retval;
    } else {
        return FALSE;
    }
}

