/* udevices.c translate PSPICE U* instances and timing models */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <assert.h>
#include "ngspice/ngspice.h"
#include "ngspice/memory.h"
#include "ngspice/bool.h"
#include "ngspice/stringskip.h"
#include "ngspice/udevices.h"
/*
 TODO check for name collisions with new names
 TODO add support for compound gates
 TODO add support for srff, pullup/down
 TODO investigate the method for generating model names
*/

#define BUFSIZE 2048
#define SMBUFSIZE 256
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
        if (strcmp(itype, "and") == 0)   { return xspice_tab[D_AND]; }
        if (strcmp(itype, "anda") == 0)  { return xspice_tab[D_AND]; }
        if (strcmp(itype, "and3") == 0)  { return xspice_tab[D_AND]; }
        if (strcmp(itype, "and3a") == 0) { return xspice_tab[D_AND]; }

        if (strcmp(itype, "ao") == 0)  { return xspice_tab[D_AO]; }
        if (strcmp(itype, "aoi") == 0) { return xspice_tab[D_AOI]; }
        break;
    }
    case 'b': {
        /* buf3 buf3a */
        if (strcmp(itype, "buf3a") == 0) { return xspice_tab[D_TRI]; }
        if (strcmp(itype, "buf") == 0)   { return xspice_tab[D_BUF]; }
        if (strcmp(itype, "bufa") == 0)  { return xspice_tab[D_BUF]; }
        if (strcmp(itype, "buf3") == 0)  { return xspice_tab[D_TRI]; }
        break;
    }
    case 'd': {
        if (strcmp(itype, "dff") == 0)   { return xspice_tab[D_DFF]; }
        if (strcmp(itype, "dltch") == 0) { return xspice_tab[D_DLTCH]; }
        break;
    }
    case 'i': {
        /* inv inva inv3 inv3a */
        if (strcmp(itype, "inv") == 0)   { return xspice_tab[D_INV]; }
        if (strcmp(itype, "inv3a") == 0) { return xspice_tab[D_INV]; }
        if (strcmp(itype, "inva") == 0)  { return xspice_tab[D_INV]; }
        if (strcmp(itype, "inv3") == 0)  { return xspice_tab[D_INV]; }
        break;
    }
    case 'j': {
        if (strcmp(itype, "jkff") == 0) { return xspice_tab[D_JKFF]; }
        break;
    }
    case 'n': {
        /* nand nanda nand3 nand3a */
        if (strcmp(itype, "nand") == 0)   { return xspice_tab[D_NAND]; }
        if (strcmp(itype, "nanda") == 0)  { return xspice_tab[D_NAND]; }
        if (strcmp(itype, "nand3") == 0)  { return xspice_tab[D_NAND]; }
        if (strcmp(itype, "nand3a") == 0) { return xspice_tab[D_NAND]; }

        /* nor nora nor3 nor3a */
        if (strcmp(itype, "nor") == 0)   { return xspice_tab[D_NOR]; }
        if (strcmp(itype, "nora") == 0)  { return xspice_tab[D_NOR]; }
        if (strcmp(itype, "nor3") == 0)  { return xspice_tab[D_NOR]; }
        if (strcmp(itype, "nor3a") == 0) { return xspice_tab[D_NOR]; }

        /* nxor nxora nxor3 nxor3a */
        if (strcmp(itype, "nxor") == 0)   { return xspice_tab[D_NXOR]; }
        if (strcmp(itype, "nxora") == 0)  { return xspice_tab[D_NXOR]; }
        if (strcmp(itype, "nxor3") == 0)  { return xspice_tab[D_NXOR]; }
        if (strcmp(itype, "nxor3a") == 0) { return xspice_tab[D_NXOR]; }
        break;
    }
    case 'o': {
        /* or ora or3 or3a */
        if (strcmp(itype, "or") == 0)   { return xspice_tab[D_OR]; }
        if (strcmp(itype, "ora") == 0)  { return xspice_tab[D_OR]; }
        if (strcmp(itype, "or3") == 0)  { return xspice_tab[D_OR]; }
        if (strcmp(itype, "or3a") == 0) { return xspice_tab[D_OR]; }

        if (strcmp(itype, "oa") == 0)  { return xspice_tab[D_OA]; }
        if (strcmp(itype, "oai") == 0) { return xspice_tab[D_OAI]; }
        break;
    }
    case 'p': {
        if (strcmp(itype, "pulldn") == 0) { return xspice_tab[D_DOWN]; }
        if (strcmp(itype, "pullup") == 0) { return xspice_tab[D_UP]; }
        break;
    }
    case 's': {
        if (strcmp(itype, "srff") == 0) { return xspice_tab[D_SRFF]; }
        break;
    }
    case 'x': {
        /* xor xora xor3 xor3a */
        if (strcmp(itype, "xor") == 0)   { return xspice_tab[D_XOR]; }
        if (strcmp(itype, "xora") == 0)  { return xspice_tab[D_XOR]; }
        if (strcmp(itype, "xor3") == 0)  { return xspice_tab[D_XOR]; }
        if (strcmp(itype, "xor3a") == 0) { return xspice_tab[D_XOR]; }
        break;
    }
    default:
        break;
    };
    return NULL;
}

/* Xlator and Xlate */
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
    printf("xspixe %s ", xp->xspice);
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

void create_model_xlator(void)
{
    model_xlatorp = create_xlator();
}

void cleanup_model_xlator(void)
{
    delete_xlator(model_xlatorp);
    model_xlatorp = NULL;
}

static Xlatep create_xlate_model(char *delays,
    char *utype, char *xspice, char *tmodel)
{
    return create_xlate("", delays, utype, xspice, tmodel, "");
}

static Xlatep find_in_model_xlator(Xlatep x)
{
    Xlatep x1;

    if (!x) { return NULL; }
    if (!model_xlatorp) { return NULL; }
    for (x1 = first_xlator(model_xlatorp); x1;
        x1 = next_xlator(model_xlatorp)) {
        if (strcmp(x1->tmodel, x->tmodel) == 0 &&
            strcmp(x1->utype, x->utype) == 0) {
            if (strcmp(x1->xspice, x->xspice) == 0) {
                return x1;
            }
        }
    }
    return NULL;
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
        printf("Already found timing model %s utype %s\n",
             x1->tmodel, x1->utype);
        delete_xlate(x);
        return;
    }
    (void) add_xlator(model_xlatorp, x);
}

/* classify gate variants */
static BOOL is_tristate_buf_array(char *itype)
{
    if (strcmp(itype, "buf3a") == 0) { return TRUE; }
    if (strcmp(itype, "inv3a") == 0) { return TRUE; }
    return FALSE;
}

static BOOL is_tristate_xor_array(char *itype)
{
    /* xor/nxor have vector inputs */
    if (strcmp(itype, "xor3a") == 0) { return TRUE; }
    if (strcmp(itype, "nxor3a") == 0) { return TRUE; }
    return FALSE;
}

static BOOL is_tristate_vector_array(char *itype)
{
    if (strcmp(itype, "and3a") == 0) { return TRUE; }
    if (strcmp(itype, "nand3a") == 0) { return TRUE; }
    if (strcmp(itype, "or3a") == 0) { return TRUE; }
    if (strcmp(itype, "nor3a") == 0) { return TRUE; }
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
    if (strcmp(itype, "buf3") == 0) { return TRUE; }
    if (strcmp(itype, "inv3") == 0) { return TRUE; }
    return FALSE;
}

static BOOL is_xor_tristate(char *itype)
{
    /* xor/nxor have vector inputs */
    if (strcmp(itype, "xor3") == 0) { return TRUE; }
    if (strcmp(itype, "nxor3") == 0) { return TRUE; }
    return FALSE;
}

static BOOL is_vector_tristate(char *itype)
{
    if (strcmp(itype, "and3") == 0) { return TRUE; }
    if (strcmp(itype, "nand3") == 0) { return TRUE; }
    if (strcmp(itype, "or3") == 0) { return TRUE; }
    if (strcmp(itype, "nor3") == 0) { return TRUE; }
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
    if (strcmp(itype, "anda") == 0) { return TRUE; }
    if (strcmp(itype, "nanda") == 0) { return TRUE; }
    if (strcmp(itype, "ora") == 0) { return TRUE; }
    if (strcmp(itype, "nora") == 0) { return TRUE; }
    return FALSE;
}

static BOOL is_buf_gate_array(char *itype)
{
    if (strcmp(itype, "bufa") == 0) { return TRUE; }
    if (strcmp(itype, "inva") == 0) { return TRUE; }
    return FALSE;
}

static BOOL is_xor_gate_array(char *itype)
{
    /* xor/nxor have vector inputs */
    if (strcmp(itype, "xora") == 0) { return TRUE; }
    if (strcmp(itype, "nxora") == 0) { return TRUE; }
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
    if (strcmp(itype, "nand") == 0) { return TRUE; }
    if (strcmp(itype, "and") == 0) { return TRUE; }
    if (strcmp(itype, "nor") == 0) { return TRUE; }
    if (strcmp(itype, "or") == 0) { return TRUE; }
    return FALSE;
}

static BOOL is_buf_gate(char *itype)
{
    if (strcmp(itype, "inv") == 0) { return TRUE; }
    if (strcmp(itype, "buf") == 0) { return TRUE; }
    return FALSE;
}

static BOOL is_xor_gate(char *itype)
{
    /* xor/nxor have vector inputs */
    if (strcmp(itype, "xor") == 0) { return TRUE; }
    if (strcmp(itype, "nxor") == 0) { return TRUE; }
    return FALSE;
}

static BOOL is_gate(char *itype)
{
    if (is_vector_gate(itype)) { return TRUE; }
    if (is_buf_gate(itype)) { return TRUE; }
    if (is_xor_gate(itype)) { return TRUE; }
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
    char *tok, *p1, *p2, *p3, *p4, *endp, *newline, *tmp, *tmp1;
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
        assert(p4 != NULL);
        *p4 = '\0';
        tmp1 = TMALLOC(char, strlen(tmp) + 1);
        (void) memcpy(tmp1, tmp, strlen(tmp) + 1);
        hdr->instance_type = tmp1;
        tfree(tmp);

        p2 = strchr(tok, ')');
        assert(p2 != NULL);
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
    /* tfree the returned string after it has been used ny the caller */
    char *tmp = NULL;
    size_t sz;
    Xlatep xdata = NULL;

    sz = 2*strlen(iname) + 3*strlen(node) + strlen("d_zero_inv99");
    sz += 100; /* that should be enough */
    tmp = TMALLOC(char, sz);
    tmp[0] = '\0';
    sprintf(tmp, "a%s_%s  %s", iname, node, node);
    sprintf(tmp + strlen(tmp), "  not_%s_%s", iname, node);
    sprintf(tmp + strlen(tmp), "  d_zero_inv99");
    /* instantiate the new inverter */
    /* e.g. au5_s1bar  s1bar  not_u5_s1bar  d_zero_inv99 */
    xdata = create_xlate_translated(tmp);
    (void) add_xlator(xlp, xdata);
    tmp[0] = '\0';
    /* the name of the inverter output */
    sprintf(tmp, "not_%s_%s", iname, node);
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
    char work[BUFSIZE];
    BOOL retval;

    printf("Need model %s %s for %s\n", tmodel, xspice, newname);
    if (strcmp(utype, "ugff") == 0) {
        xin = create_xlate_model("", utype, xspice, tmodel);
    } else {
        xin = create_xlate_model("", utype, "", tmodel);
    }
    xout = find_in_model_xlator(xin);
    if (xout) {
        /* Don't delete xout or the model_xlatorp will be corrupted */
        //printf("Found tming model %s %s\n", xout->tmodel, xout->utype);
        print_xlate(xout);
        work[0] = '\0';
        sprintf(work, ".model %s %s", newname, xspice);
        if (xout->delays && strlen(xout->delays) > 0) {
            sprintf(work + strlen(work), "%s", xout->delays);
        }
        newdata = create_xlate_translated(work);
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
    char work[BUFSIZE];
    char smallbuf[SMBUFSIZE];
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
    xxp = create_xlator();

    preb = ip->prebar;
    if (strcmp(preb, "$d_hi") == 0) {
        preb = "NULL";
    } else {
        need_preb_inv = TRUE;
        preb = new_inverter(iname, preb, xxp);
    }

    clrb = ip->clrbar;
    if (strcmp(clrb, "$d_hi") == 0) {
        clrb = "NULL";
    } else {
        need_clrb_inv = TRUE;
        clrb = new_inverter(iname, clrb, xxp);
    }

    clk = ip->clk;
    tmodel = ip->tmodel;
    smallbuf[0] = '\0';
    /* model name, same for each dff */
    sprintf(smallbuf, "d_a%s%s", iname, itype);
    for (i = 0; i < num_gates; i++) {
        work[0] = '\0';
        sprintf(work, "a%s_%d", iname, i);
        sprintf(work + strlen(work), "  %s  %s  %s  %s",
            darr[i], clk, preb, clrb);
        qout = qarr[i];
        if (strcmp(qout, "$d_nc") == 0) {
            qout = "NULL";
        }
        sprintf(work + strlen(work), "  %s", qout);
        qbout = qbarr[i];
        if (strcmp(qbout, "$d_nc") == 0) {
            qbout = "NULL";
        }
        sprintf(work + strlen(work), "  %s", qbout);
        sprintf(work + strlen(work), "  %s", smallbuf);
        xdata = create_xlate_instance(work, " d_dff", tmodel, smallbuf);
        xxp = add_xlator(xxp, xdata);
    }
    if (!gen_timing_model(tmodel, "ueff", "d_dff", smallbuf, xxp)) {
        printf("WARNING unable to find tmodel %s for %s d_dff\n",
            tmodel, smallbuf);
    }
    if (need_preb_inv || need_clrb_inv) {
        xdata = create_xlate_translated(".model d_zero_inv99 d_inverter");
        xxp = add_xlator(xxp, xdata);
    }
    if (need_preb_inv) { tfree(preb); }
    if (need_clrb_inv) { tfree(clrb); }

    return xxp;
    return NULL;
}

static Xlatorp gen_jkff_instance(struct jkff_instance *ip)
{
    char *itype, *iname, **jarr, **karr, **qarr, **qbarr;
    char *preb, *clrb, *clkb, *tmodel, *qout, *qbout;
    int i, num_gates;
    char work[BUFSIZE];
    char smallbuf[SMBUFSIZE];
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
    xxp = create_xlator();

    preb = ip->prebar;
    if (strcmp(preb, "$d_hi") == 0) {
        preb = "NULL";
    } else {
        need_preb_inv = TRUE;
        preb = new_inverter(iname, preb, xxp);
    }

    clrb = ip->clrbar;
    if (strcmp(clrb, "$d_hi") == 0) {
        clrb = "NULL";
    } else {
        need_clrb_inv = TRUE;
        clrb = new_inverter(iname, clrb, xxp);
    }

    /* require a positive edge clock */
    clkb = ip->clkbar;
    clkb = new_inverter(iname, clkb, xxp);

    tmodel = ip->tmodel;
    smallbuf[0] = '\0';
    /* model name, same for each latch */
    sprintf(smallbuf, "d_a%s%s", iname, itype);
    for (i = 0; i < num_gates; i++) {
        work[0] = '\0';
        sprintf(work, "a%s_%d", iname, i);
        sprintf(work + strlen(work), "  %s  %s  %s  %s  %s",
            jarr[i], karr[i], clkb, preb, clrb);
        qout = qarr[i];
        if (strcmp(qout, "$d_nc") == 0) {
            qout = "NULL";
        }
        sprintf(work + strlen(work), "  %s", qout);
        qbout = qbarr[i];
        if (strcmp(qbout, "$d_nc") == 0) {
            qbout = "NULL";
        }
        sprintf(work + strlen(work), "  %s  %s", qbout, smallbuf);
        xdata = create_xlate_instance(work, " d_jkff", tmodel, smallbuf);
        xxp = add_xlator(xxp, xdata);
    }
    if (!gen_timing_model(tmodel, "ueff", "d_jkff", smallbuf, xxp)) {
        printf("WARNING unable to find tmodel %s for %s d_jkff\n",
            tmodel, smallbuf);
    }
    xdata = create_xlate_translated(".model d_zero_inv99 d_inverter");
    xxp = add_xlator(xxp, xdata);
    tfree(clkb);
    if (need_preb_inv) { tfree(preb); }
    if (need_clrb_inv) { tfree(clrb); }

    return xxp;
    return NULL;
}

static Xlatorp gen_dltch_instance(struct dltch_instance *ip)
{
    char *itype, *iname, **darr, **qarr, **qbarr;
    char *preb, *clrb, *gate, *tmodel, *qout, *qbout;
    int i, num_gates;
    char work[BUFSIZE];
    char smallbuf[SMBUFSIZE];
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
    xxp = create_xlator();
    preb = ip->prebar;
    if (strcmp(preb, "$d_hi") == 0) {
        preb = "NULL";
    } else {
        need_preb_inv = TRUE;
        preb = new_inverter(iname, preb, xxp);
    }

    clrb = ip->clrbar;
    if (strcmp(clrb, "$d_hi") == 0) {
        clrb = "NULL";
    } else {
        need_clrb_inv = TRUE;
        clrb = new_inverter(iname, clrb, xxp);
    }
    gate = ip->gate;
    tmodel = ip->tmodel;
    smallbuf[0] = '\0';
    /* model name, same for each latch */
    sprintf(smallbuf, "d_a%s%s", iname, itype);
    for (i = 0; i < num_gates; i++) {
        work[0] = '\0';
        sprintf(work, "a%s_%d", iname, i);
        sprintf(work + strlen(work), "  %s  %s  %s  %s",
            darr[i], gate, preb, clrb);
        qout = qarr[i];
        if (strcmp(qout, "$d_nc") == 0) {
            /* NULL not allowed??? */
            sprintf(work + strlen(work), "  nco%s_%d", iname, i);
        } else {
            sprintf(work + strlen(work), "  %s", qout);
        }
        qbout = qbarr[i];
        if (strcmp(qbout, "$d_nc") == 0) {
            /* NULL not allowed??? */
            sprintf(work + strlen(work), " ncn%s_%d", iname, i);
        } else {
            sprintf(work + strlen(work), "  %s", qbout);
        }
        sprintf(work + strlen(work), "  %s", smallbuf);
        xdata = create_xlate_instance(work, " d_dlatch", tmodel, smallbuf);
        xxp = add_xlator(xxp, xdata);
    }
    if (!gen_timing_model(tmodel, "ugff", "d_dlatch", smallbuf, xxp)) {
        printf("WARNING unable to find tmodel %s for %s d_dlatch\n",
            tmodel, smallbuf);
    }
    if (need_preb_inv || need_clrb_inv) {
        xdata = create_xlate_translated(".model d_zero_inv99 d_inverter");
        xxp = add_xlator(xxp, xdata);
    }
    if (need_preb_inv) { tfree(preb); }
    if (need_clrb_inv) { tfree(clrb); }

    return xxp;
}

static Xlatorp gen_gate_instance(struct gate_instance *gip)
{
    char **inarr, **outarr, *itype, *iname, *enable, *tmodel;
    char *work, *xspice = NULL, *connector = NULL;
    BOOL vector = FALSE, tristate_gate = FALSE, simple_gate = FALSE;
    BOOL tristate_array = FALSE, simple_array = FALSE;
    BOOL add_tristate = FALSE;
    char smallbuf[SMBUFSIZE], modelbuf[SMBUFSIZE];
    int i, j, k, width, num_gates, num_ins, num_outs;
    size_t sz;
    Xlatorp xxp = NULL;
    Xlatep xdata = NULL;

    if (!gip) { return NULL; }
    smallbuf[0] = '\0';
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
    assert(num_gates >= 1);
    vector = has_vector_inputs(itype);

    if (num_gates == 1) {
        assert(num_outs == 1);
        simple_gate = is_gate(itype);
        tristate_gate = is_tristate(itype);
        if (!simple_gate && !tristate_gate) { return NULL; }
        if (simple_gate && tristate_gate) { assert(FALSE); }
        add_tristate = FALSE;
        if (simple_gate) {
            assert(!enable);
            xspice = find_xspice_for_delay(itype);
        } else if (tristate_gate) {
            assert(enable);
            xspice = find_xspice_for_delay(itype);
            if (strcmp(itype, "buf3") != 0) {
                add_tristate = TRUE;
            }
        }
        assert(xspice);
        xxp = create_xlator();
        work = TMALLOC(char, BUFSIZE);
        work[0] = '\0';
        /* Now build the instance name and inputs section */
        /* instance name */
        sprintf(work, "a%s ", iname);
        if (vector) { strcat(work, "["); }
        /* inputs */
        for (i = 0; i < width; i++) {
            sprintf(work + strlen(work), " %s", inarr[i]);
        }
        if (vector) { strcat(work, " ]"); }
        /* add the tristate enable if required on original */
        if (enable) {
            assert(tristate_gate);
            if (!add_tristate) {
                sprintf(work + strlen(work),"  %s", enable);
            }
        }
        /* connector if required for tristate */
        sprintf(smallbuf, "a%s_%s", iname, outarr[0]);
        connector = TMALLOC(char, strlen(smallbuf) + 1);
        connector[0] = '\0';
        (void) memcpy(connector, smallbuf, strlen(smallbuf) + 1);

        /* keep a copy of the model name of original gate */
        modelbuf[0] = '\0';
        sprintf(modelbuf, "d_a%s%s", iname, itype);

        if (!add_tristate) {
            /* add output + model name  => translated instance */
            sprintf(work + strlen(work), " %s %s", outarr[0], modelbuf);
            xdata = create_xlate_instance(work, xspice, tmodel, modelbuf);
            xxp = add_xlator(xxp, xdata);
            if (simple_gate) {
                if (!gen_timing_model(tmodel, "ugate", xspice,
                    modelbuf, xxp)) {
                    printf("WARNING unable to find tmodel %s for %s %s\n",
                        tmodel, modelbuf, xspice);
                }
            } else { /* must be trstate gate buf3 */
                if (!gen_timing_model(tmodel, "utgate", xspice,
                    modelbuf, xxp)) {
                    printf("WARNING unable to find tmodel %s for %s %s\n",
                        tmodel, modelbuf, xspice);
                }
            }
        } else {
            /*
             Use connector as original gate output and tristate input;
             tristate has original gate output and utgate delay;
             original gate has zero delay timing model.
             Complete the translation of the original gate adding
             the connector as output + model name.
            */
            sprintf(work + strlen(work), " %s %s", connector, modelbuf);
            xdata = create_xlate_instance(work, xspice, "", modelbuf);
            xxp = add_xlator(xxp, xdata);
            /* new model statement e.g.   .model d_au2nand3 d_nand */
            work[0] = '\0';
            sprintf(work, ".model %s %s", modelbuf, xspice);
            xdata = create_xlate_translated(work);
            xxp = add_xlator(xxp, xdata);
            /* now the added tristate */
            work[0] = '\0';
            modelbuf[0] = '\0';
            /* model name of added tristate */
            sprintf(modelbuf, "d_a%stribuf", iname);
            sprintf(work, "a%s_tri %s %s %s %s", iname, connector,
                enable, outarr[0], modelbuf);
            xdata = create_xlate_instance(work, "d_tristate", tmodel, modelbuf);
            xxp = add_xlator(xxp, xdata);
            if (!gen_timing_model(tmodel, "utgate", "d_tristate",
                modelbuf, xxp)) {
                printf("WARNING unable to find tmodel %s for %s %s\n",
                    tmodel, modelbuf, xspice);
            }
        }
        tfree(work);
        tfree(connector);
        return xxp;

    } else {
        /* arrays of gates */
        /* NOTE (n)and3a, (n)or3a, (n)xora types are not supported */
        assert(num_outs == num_gates);
        assert(num_ins == num_gates * width);
        simple_array = is_gate_array(itype);
        tristate_array = is_tristate_array(itype);
        add_tristate = FALSE;
        if (simple_array) {
            assert(!tristate_array);
            assert(!enable);
            xspice = find_xspice_for_delay(itype);
        } else if (tristate_array) {
            xspice = find_xspice_for_delay(itype);
            if (strcmp("inv3a", itype) == 0) {
                add_tristate = TRUE;
            } else if (strcmp(itype, "buf3a") != 0) {
                return NULL;
            }
            assert(enable);
            assert(!vector);
        } else {
            assert(FALSE);
        }
        assert(xspice);
        xxp = create_xlator();
        sz = BUFSIZE;
        work = TMALLOC(char, sz);
        k = 0;
        connector = NULL;
        /* model name, same for all primary gates */
        modelbuf[0] = '\0';
        sprintf(modelbuf, "d_a%s%s", iname, itype); 
        for (i = 0; i < num_gates; i++) {
            work[0] = '\0';
            /* create new instance name for primary gate */
            sprintf(work, "a%s_%d ", iname, i);
            if (vector) { strcat(work, "["); }
            for (j = 0; j < width; j++) {
                /* inputs for primary gate */
                sprintf(work + strlen(work), " %s", inarr[k]);
                k++;
            }
            if (vector) { strcat(work, " ]"); }
            if (enable) {
                if (!add_tristate) {
                    sprintf(work + strlen(work),"  %s", enable);
                } else {
                    /* connector if required for tristate */
                    sprintf(smallbuf, "a%s_%d_%s", iname, i, outarr[i]);
                    connector = TMALLOC(char, strlen(smallbuf) + 1);
                    connector[0] = '\0';
                    (void) memcpy(connector, smallbuf, strlen(smallbuf) + 1);
                }
            }
            /* output of primary gate */
            if (add_tristate) {
                sprintf(work + strlen(work), " %s", connector);
            } else {
                sprintf(work + strlen(work), " %s", outarr[i]);
            }
            /* translated instance */
            sprintf(work + strlen(work), " %s", modelbuf);
            if (add_tristate) {
                xdata = create_xlate_instance(work, xspice, "", modelbuf);
                xxp = add_xlator(xxp, xdata);
            } else {
                xdata = create_xlate_instance(work, xspice, tmodel, modelbuf);
                xxp = add_xlator(xxp, xdata);
                if (tristate_array) {
                    assert(strcmp(xspice, "d_tristate") == 0);
                    assert(strcmp(itype, "buf3a") == 0);
                    if (i == 0 && !gen_timing_model(tmodel, "utgate",
                                            xspice, modelbuf, xxp)) {
                        printf("WARNING unable to find tmodel %s for %s %s\n",
                            tmodel, modelbuf, xspice);
                    }
                } else {
                    if (i == 0 && !gen_timing_model(tmodel, "ugate",
                                            xspice, modelbuf, xxp)) {
                        printf("WARNING unable to find tmodel %s for %s %s\n",
                            tmodel, modelbuf, xspice);
                    }
                }
            }

            if (add_tristate) {
                if (i == 0) {
                    /* Zero delay model for all original array instances */
                    work[0] = '\0';
                    sprintf(work, ".model %s %s", modelbuf, xspice); 
                    xdata = create_xlate_translated(work);
                    xxp = add_xlator(xxp, xdata);
                }

                work[0] = '\0';
                smallbuf[0] = '\0';
                /* model name of added tristate */
                sprintf(smallbuf, "d_a%stribuf", iname);
                /*
                 instance name of added tristate, connector,
                 enable, original primary gate output, timing model.
                */
                sprintf(work, "a%s_%d_tri %s %s %s %s", iname, i, connector,
                    enable, outarr[i], smallbuf);
                xdata = create_xlate_instance(work, "d_tristate",
                    tmodel, smallbuf);
                xxp = add_xlator(xxp, xdata);
                if (i == 0 && !gen_timing_model(tmodel, "utgate",
                                        "d_tristate", smallbuf, xxp)) {
                    printf("WARNING unable to find tmodel %s for %s %s\n",
                        tmodel, smallbuf, "d_tristate");
                }
                tfree(connector);
            }

            if (strlen(work) > BUFSIZE/2) {
                tfree(work);
                sz += BUFSIZE;
                printf("TMALLOC work size %lu\n", sz);
                work = TMALLOC(char, sz);
            }
        }
        tfree(work);
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
    static char avebuf[SMBUFSIZE];

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
            sprintf(avebuf, "%.2f%s", average, units2);
            if (strcmp(units1, units2) != 0) {
                printf("WARNING units do not match\n");
            }
            tdp->ave = TMALLOC(char, strlen(avebuf) + 1);
            (void) memcpy(tdp->ave, avebuf, strlen(avebuf) + 1);
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
    if (tdp->estimate == EST_MIN) { return tdp->min; }
    if (tdp->estimate == EST_TYP) { return tdp->typ; }
    if (tdp->estimate == EST_MAX) { return tdp->max; }
    if (tdp->estimate == EST_AVE) { return tdp->ave; }
    return NULL;
}

static char *get_delays_ugate(char *rem, char *d_name)
{
    char *rising, *falling, *delays = NULL;
    int count;
    char work_buf[BUFSIZE];
    struct timing_data *tdp1, *tdp2;

    tdp1 = create_min_typ_max("tplh", rem);
    estimate_typ(tdp1);
    rising = get_estimate(tdp1);
    tdp2 = create_min_typ_max("tphl", rem);
    estimate_typ(tdp2);
    falling = get_estimate(tdp2);
    if (rising && falling) {
        if (strlen(rising) > 0 && strlen(falling) > 0) {
            count = sprintf(work_buf, "(rise_delay = %s fall_delay = %s)",
                            rising, falling);
            count++;
            delays = TMALLOC(char, count);
            (void) memcpy(delays, work_buf, count);
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
    int count;
    char work_buf[BUFSIZE];
    struct timing_data *tdp1, *tdp2;

    tdp1 = create_min_typ_max("tplh", rem);
    estimate_typ(tdp1);
    rising = get_estimate(tdp1);
    tdp2 = create_min_typ_max("tphl", rem);
    estimate_typ(tdp2);
    falling = get_estimate(tdp2);
    if (rising && falling) {
        if (strlen(rising) > 0 && strlen(falling) > 0) {
            count = sprintf(work_buf, "(delay = %s)", rising);
            count++;
            delays = TMALLOC(char, count);
            (void) memcpy(delays, work_buf, count);
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
    int count;
    char work_buf[BUFSIZE];
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
        count = sprintf(work_buf, "(clk_delay = %s "
            "set_delay = %s reset_delay = %s "
            "rise_delay = 1.0ns fall_delay = 2.0ns)",
            clkd, setd, resetd);
    } else if (clkd) {
        count = sprintf(work_buf, "(clk_delay = %s "
            "rise_delay = 1.0ns fall_delay = 2.0ns)",
            clkd);
    } else if (setd) {
        count = sprintf(work_buf, "(set_delay = %s reset_delay = %s "
            "rise_delay = 1.0ns fall_delay = 2.0ns)",
            setd, resetd);
    } else {
        count = sprintf(work_buf, "(rise_delay = 1.0ns fall_delay = 2.0ns)");
    }
    count++;
    delays = TMALLOC(char, count);
    (void) memcpy(delays, work_buf, count);
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
    char work_buf[BUFSIZE];
    struct timing_data *tdp1, *tdp2, *tdp3, *tdp4, *tdp5, *tdp6;

    if (strcmp(d_name, "d_dlatch") == 0) {
        dname = "data_delay";
    } else if (strcmp(d_name, "d_srlatch") == 0) {
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
    work_buf[0] = '\0';
    strcpy(work_buf, "(");
    d_delay = NULL;
    if (tpdqlh && strlen(tpdqlh) > 0) {
        d_delay = tpdqlh;
    } else if (tpdqhl && strlen(tpdqhl) > 0) {
        d_delay = tpdqhl;
    }
    if (d_delay) {
        sprintf(&work_buf[strlen(work_buf)], "%s = %s ", dname, d_delay);
    }
    enab = NULL;
    if (tpgqlh && strlen(tpgqlh) > 0) {
        enab = tpgqlh;
    } else if (tpgqhl && strlen(tpgqhl) > 0) {
        enab = tpgqhl;
    }
    if (enab) {
        sprintf(&work_buf[strlen(work_buf)], "enable_delay = %s ", enab);
    }
    setd = NULL;
    resetd = NULL;
    if (tppcqlh && strlen(tppcqlh) > 0) {
        setd = resetd = tppcqlh;
    } else if (tppcqhl && strlen(tppcqhl) > 0) {
        setd = resetd = tppcqhl;
    }
    if (setd) {
        sprintf(&work_buf[strlen(work_buf)],
            "set_delay = %s reset_delay = %s ", setd, resetd);
    }
    sprintf(&work_buf[strlen(work_buf)],
        "rise_delay = 1.0ns fall_delay = 2.0ns)");
    delays = TMALLOC(char, strlen(work_buf) + 1);
    (void) memcpy(delays, work_buf, strlen(work_buf) + 1);
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
    printf("\nmodel_name -> %s\n", tok);
    tmodel = TMALLOC(char, strlen(tok) + 1);
    memcpy(tmodel, tok, strlen(tok) + 1);
    /* model utype */
    tok = strtok(NULL, " \t(");
    printf("model_utype -> %s\n", tok);
    utype = TMALLOC(char, strlen(tok) + 1);
    memcpy(utype, tok, strlen(tok) + 1);

    /* delay info */
    remainder = strchr(original, '(');
    if (remainder) {
        if (strcmp(utype, "ugate") == 0) {
            delays = get_delays_ugate(remainder, xspice);
            printf("<%s>\n", delays);
            if (delays) {
                add_delays_to_model_xlator(delays, utype, "", tmodel);
            } else {
                add_delays_to_model_xlator("", utype, "", tmodel);
            }
            if (delays) { tfree(delays); }
        } else if (strcmp(utype, "utgate") == 0) {
            delays = get_delays_utgate(remainder, xspice);
            printf("<%s>\n", delays);
            if (delays) {
                add_delays_to_model_xlator(delays, utype, "", tmodel);
            } else {
                add_delays_to_model_xlator("", utype, "", tmodel);
            }
            if (delays) { tfree(delays); }
        } else if (strcmp(utype, "ueff") == 0) {
            delays = get_delays_ueff(remainder, xspice);
            printf("<%s>\n", delays);
            if (delays) {
                add_delays_to_model_xlator(delays, utype, "", tmodel);
            } else {
                add_delays_to_model_xlator("", utype, "", tmodel);
            }
            if (delays) { tfree(delays); }
        } else if (strcmp(utype, "ugff") == 0) {
            delays = get_delays_ugff(remainder, "d_dlatch");
            printf("<%s>\n", delays);
            if (delays) {
                add_delays_to_model_xlator(delays, utype, "d_dlatch", tmodel);
            } else {
                add_delays_to_model_xlator("", utype, "d_dlatch", tmodel);
            }
            if (delays) { tfree(delays); }
            delays = get_delays_ugff(remainder, "d_srlatch");
            printf("<%s>\n", delays);
            if (delays) {
                add_delays_to_model_xlator(delays, utype, "d_srlatch", tmodel);
            } else {
                add_delays_to_model_xlator("", utype, "d_srlatch", tmodel);
            }
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
    return dlp;
}

static struct jkff_instance *add_jkff_inout_timing_model(
    struct instance_hdr *hdr, char *start)
{
    char *tok, *copyline;
    char *name, **arrp;
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
    return jkffip;
}

static struct gate_instance *add_array_inout_timing_model(
    struct instance_hdr *hdr, char *start)
{
    char *tok, *copyline, *itype  = hdr->instance_type;
    BOOL first = TRUE, tristate = FALSE, verbose = FALSE;
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
    if (verbose) {
        printf("instance: %s itype: %s\n",
               hdr->instance_name, hdr->instance_type);
    }
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
            if (verbose) { printf(" gate %d input(%d): %s\n", i, j, tok); }
            k++;
        }
    }
    /* enable for tristate */
    if (tristate) {
        tok = strtok(NULL, " \t");
        name = TMALLOC(char, strlen(tok) + 1);
        (void) memcpy(name, tok, strlen(tok) + 1);
        gip->enable = name;
        if (verbose) { printf(" enable: %s\n", tok); }
    }
    /* outputs next */
    outarr = TMALLOC(char *, numgates);
    gip->outputs = outarr;
    for (i = 0; i < numgates; i++) {
        tok = strtok(NULL, " \t");
        name = TMALLOC(char, strlen(tok) + 1);
        (void) memcpy(name, tok, strlen(tok) + 1);
        outarr[i] = name;
        if (verbose) { printf(" gate %d output: %s\n", i, tok); }
    }
    /* timing model last */
    tok = strtok(NULL, " \t");
    name = TMALLOC(char, strlen(tok) + 1);
    (void) memcpy(name, tok, strlen(tok) + 1);
    gip->tmodel = name;
    if (verbose) { printf(" tmodel: %s\n", tok); }
    tfree(copyline);
    return gip;
}

static struct gate_instance *add_gate_inout_timing_model(
    struct instance_hdr *hdr, char *start)
{
    char *tok, *copyline, *itype  = hdr->instance_type;
    int i, n1 = hdr->num1, n2 = hdr->num2, inwidth;
    BOOL first = TRUE, tristate = FALSE, verbose = FALSE;
    struct gate_instance *gip = NULL;
    char **inarr = NULL, **outarr = NULL, *name;

    assert(n2 == -1);
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
    if (verbose) {
        printf("instance: %s itype: %s\n",
               hdr->instance_name, hdr->instance_type);
    }
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
        if (verbose) { printf(" input(%d): %s\n", i, tok); }
    }
    /* enable for tristate */
    if (tristate) {
        tok = strtok(NULL, " \t");
        name = TMALLOC(char, strlen(tok) + 1);
        (void) memcpy(name, tok, strlen(tok) + 1);
        gip->enable = name;
        if (verbose) { printf(" enable: %s\n", tok); }
    }
    /* output */
    assert(gip->num_outs == 1);
    outarr = TMALLOC(char *, gip->num_outs);
    gip->outputs = outarr;
    tok = strtok(NULL, " \t");
    name = TMALLOC(char, strlen(tok) + 1);
    (void) memcpy(name, tok, strlen(tok) + 1);
    outarr[0] = name;
    if (verbose) { printf(" output: %s\n", tok); }
    /* timing model last */
    tok = strtok(NULL, " \t");
    name = TMALLOC(char, strlen(tok) + 1);
    (void) memcpy(name, tok, strlen(tok) + 1);
    gip->tmodel = name;
    if (verbose) { printf(" tmodel: %s\n", tok); }
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

static Xlatorp translate_ff_latch(struct instance_hdr *hdr, char *start)
{
    char *itype;
    struct dff_instance *dffp = NULL;
    struct jkff_instance *jkffp = NULL;
    struct dltch_instance *dltchp = NULL;
    Xlatorp xp;

    itype = hdr->instance_type;
    if (strcmp(itype, "dff") == 0) {
        dffp = add_dff_inout_timing_model(hdr, start);
        if (dffp) {
            xp = gen_dff_instance(dffp);
            delete_dff_instance(dffp);
            return xp;
        }
    } else if (strcmp(itype, "jkff") == 0) {
        jkffp = add_jkff_inout_timing_model(hdr, start);
        if (jkffp) {
            xp = gen_jkff_instance(jkffp);
            delete_jkff_instance(jkffp);
            return xp;
        }
    } else if (strcmp(itype, "dltch") == 0) {
        dltchp = add_dltch_inout_timing_model(hdr, start);
        if (dltchp) {
            xp = gen_dltch_instance(dltchp);
            delete_dltch_instance(dltchp);
            return xp;
        }
    } else {
        assert(FALSE);
        return NULL;
    }
    return NULL;
}

static Xlatorp translate_gate(struct instance_hdr *hdr, char *start)
{
    /* if unable to translate return 0, else return 1 */
    char *itype;
    struct gate_instance *igatep;
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
    } else {
        assert(FALSE);
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

BOOL u_process_instance(char *nline)
{
    /* Return TRUE if ok */
    char *p1, *itype, *xspice;
    struct instance_hdr *hdr = create_instance_header(nline);
    Xlatorp xp = NULL;
    BOOL retval = TRUE;

    itype = hdr->instance_type;
    xspice = find_xspice_for_delay(itype);
    if (!xspice) {
        delete_instance_hdr(hdr);
        return FALSE;
    }
    /* Skip past instance name, type, pwr, gnd */
    p1 = skip_past_words(nline, 4);
    if (is_gate(itype) || is_gate_array(itype)) {
        xp = translate_gate(hdr, p1);
    } else if (is_tristate(itype) || is_tristate_array(itype)) {
        xp = translate_gate(hdr, p1);
    } else if (strcmp(itype, "dff") == 0 || strcmp(itype, "jkff") == 0 ||
        strcmp(itype, "dltch") == 0) {
        xp = translate_ff_latch(hdr, p1);
    } else {
        delete_instance_hdr(hdr);
        retval = FALSE;
    }
    if (xp) {
        //interpret_xlator(xp, TRUE);
        delete_xlator(xp);
    }
    return retval;
}

BOOL u_process_model_line(char *line)
{
    /* Translate a .model line to find the delays */
    /* Return TRUE if ok */
    char *newline;
    size_t n = strlen(line) - 1;

    if (n > 0 && line[n] == '\n') line[n] = '\0';
    if (strncmp(line, ".model ", strlen(".model ")) == 0) {
        newline = TMALLOC(char, strlen(line) + 1);
        (void) memcpy(newline, line, strlen(line) + 1);
        u_process_model(newline, line, "model_new_name", "d_xspice");
        tfree(newline);
        return TRUE;
    } else {
        return FALSE;
    }
}

