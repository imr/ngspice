/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Routines for dealing with the vector database.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/sim.h"
#include "ngspice/stringskip.h"

#include "circuits.h"
#include "completion.h"
#include "variable.h"
#include "dimens.h"
#include "../misc/misc_time.h"
#include "vectors.h"
#include "ngspice/dstring.h"
#include "plotting/plotting.h"
#ifdef XSPICE
#include "ngspice/evt.h"
#include "ngspice/mif.h"
#endif

static struct dvec *findvec_all(struct plot *pl);
static struct dvec *findvec_allv(struct plot *pl);
static struct dvec *findvec_alli(struct plot *pl);
static struct dvec *findvec_ally(struct plot *pl);
#ifdef XSPICE
static struct dvec *findvec_alle(void);
#endif
static struct dvec *find_permanent_vector_by_name(
        NGHASHPTR pl_lookup_table, char *name);
static enum ALL_TYPE_ENUM get_all_type(const char *word);
static bool plot_prefix(const char *pre, const char *str);

#ifdef XSPICE
extern int EVTswitch_plot(CKTcircuit* ckt, const char* plottypename);
struct dvec *EVTfindvec(char *node);
#endif


static void vec_rebuild_lookup_table(struct plot *pl)
{
    if (pl->pl_lookup_table) { /* existing table */
        nghash_empty(pl->pl_lookup_table, NULL, NULL);
    }
    else { /* new table */
        int cnt = 0; /* count entries */
        struct dvec *d; /* dynamic vector */
        for (d = pl->pl_dvecs; d; d = d->v_next) { /* get # vec */
            cnt++;
        }
        pl->pl_lookup_table = nghash_init(cnt);
        /* allow multiple entries */
        nghash_unique(pl->pl_lookup_table, FALSE);
    }

    {
        /* Access lookup table directly for speed */
        NGHASHPTR lookup_p = pl->pl_lookup_table;
        DS_CREATE(dbuf, 200); /* make dynamic buffer */
        struct dvec *d; /* dynamic vector */
        for (d = pl->pl_dvecs; d; d = d->v_next) {
            ds_clear(&dbuf);
            if (ds_cat_str_case(&dbuf, d->v_name, ds_case_lower) != DS_E_OK) {
                fprintf(stderr, "Error: DS could not add string %s\n", d->v_name);
                controlled_exit(-1);
            }
            char * const lower_name = ds_get_buf(&dbuf);
            nghash_insert(lookup_p, lower_name, d); /* add lower-cased name */
        } /* end of loop over vectors */
        ds_free(&dbuf);
    }

    pl->pl_lookup_valid = TRUE; /* now lookup table valid */
} /* end of function vec_rebuild_lookup_table */



enum ALL_TYPE_ENUM {
    ALL_TYPE_NONE,
    ALL_TYPE_ALL,
    ALL_TYPE_ALLV,
    ALL_TYPE_ALLI,
    ALL_TYPE_ALLY,
    ALL_TYPE_ALLE
};


/* Efficient identification of "all", "allv", "alli", "ally", "alle", and anything
 * else */
static enum ALL_TYPE_ENUM get_all_type(const char *word)
{
    /* Check for start of "all" */
    if (tolower(word[0] != 'a')) {
        return ALL_TYPE_NONE;
    }
    if (tolower(word[1] != 'l')) {
        return ALL_TYPE_NONE;
    }
    if (tolower(word[2] != 'l')) {
        return ALL_TYPE_NONE;
    }

    /* It may be some type of all */
    switch (tolower(word[3])) {
    case '\0':
        return ALL_TYPE_ALL;
    case 'v':
        if (word[4] == '\0') {
            return ALL_TYPE_ALLV;
        }
        else {
            return ALL_TYPE_NONE;
        }
    case 'i':
        if (word[4] == '\0') {
            return ALL_TYPE_ALLI;
        }
        else {
            return ALL_TYPE_NONE;
        }
    case 'y':
        if (word[4] == '\0') {
            return ALL_TYPE_ALLY;
        }
        else {
            return ALL_TYPE_NONE;
        }
    case 'e':
        if (word[4] == '\0') {
            return ALL_TYPE_ALLE;
        }
        else {
            return ALL_TYPE_NONE;
        }
    default:
        return ALL_TYPE_NONE;
    } /* end of swith over char after "all" */
} /* end of function get_all_type */



/* Find a named vector in a plot. We are careful to copy the vector if
 * v_link2 is set, because otherwise we will get screwed up.  */
static struct dvec *findvec(char *word, struct plot *pl)
{
    /* If no plot, cannot find */
    if (pl == NULL) {
        return NULL;
    }

    /* Identify and handle special cases all, allv, alli, ally, alle */
    switch (get_all_type(word)) {
    case ALL_TYPE_ALL:
        return findvec_all(pl);
    case ALL_TYPE_ALLV:
        return findvec_allv(pl);
    case ALL_TYPE_ALLI:
        return findvec_alli(pl);
    case ALL_TYPE_ALLY:
        return findvec_ally(pl);
#ifdef XSPICE
    case ALL_TYPE_ALLE:
        return findvec_alle();
#endif
    default: /* case ALL_TYPE_NOT_ALL -- not some type of ALL */
        break;
    }

    /* The find is not for one of the "all" cases */
    if (!pl->pl_lookup_valid) {
        /* Table lookup not valid, so rebuild to make valid */
        vec_rebuild_lookup_table(pl);
    }

    DS_CREATE(dbuf, 200); /* make dynamic buffer */
    if (ds_cat_str_case(&dbuf, word, ds_case_lower) != DS_E_OK) {
        fprintf(stderr, "Error: DS could not add string %s\n", word);
        controlled_exit(-1);
    }
    char * const lower_name = ds_get_buf(&dbuf);
    NGHASHPTR pl_lookup_table = pl->pl_lookup_table;
    struct dvec *d = find_permanent_vector_by_name(pl_lookup_table,
            lower_name);

    /* If the vector was not using the lowercased name, try finding it as
     * v(lowercased name) */
    if (!d) {
        ds_clear(&dbuf);
        bool f_ok = ds_cat_str(&dbuf, "v(") == DS_E_OK;
        f_ok &= ds_cat_str_case(&dbuf, word,
                ds_case_lower) == DS_E_OK;
        f_ok &= ds_cat_char(&dbuf, ')') == DS_E_OK;
        if (!f_ok) {
            fprintf(stderr, "Error: DS could not add string V() around %s\n", word);
            controlled_exit(-1);
        }
        char * const node_name = ds_get_buf(&dbuf);
        d = find_permanent_vector_by_name(pl_lookup_table, node_name);
    }

    ds_free(&dbuf);

#ifdef XSPICE
    /* gtri - begin - Add processing for getting event-driven vector */

    if (!d) {
        d = EVTfindvec(word);
    }

    /* gtri - end   - Add processing for getting event-driven vector */
#endif
    if (d && d->v_link2) {
        d = vec_copy(d);
        vec_new(d);
    }

    return d;
} /* end of function findvec */



/* Macro taking a function name and vector filter as arguments that
 * generates the function that applies the filter */
#define FINDVEC_ALL_GEN(fun_name, filter)\
static struct dvec *fun_name(struct plot *pl)\
{\
    struct dvec *d, *newv = NULL, *end = NULL, *v;\
    for (d = pl->pl_dvecs; d; d = d->v_next) {\
        if (filter) {\
            if (d->v_link2) {\
                v = vec_copy(d);\
                vec_new(v);\
            }\
            else {\
                v = d;\
            }\
            if (end) {\
                end->v_link2 = v;\
            }\
            else {\
                newv = v;\
            }\
            end = v;\
        }\
    } /* end of loop over vectors in plot */\
\
    return newv;\
} /* end of function */

/* Generate the functions for each filter */
FINDVEC_ALL_GEN(findvec_all, d->v_flags & VF_PERMANENT)
FINDVEC_ALL_GEN(findvec_allv,
        (d->v_flags & VF_PERMANENT) && (d->v_type == SV_VOLTAGE))
FINDVEC_ALL_GEN(findvec_alli,
        (d->v_flags & VF_PERMANENT) && (d->v_type == SV_CURRENT))
FINDVEC_ALL_GEN(findvec_ally,
        (d->v_flags & VF_PERMANENT) &&
                (!cieq(d->v_name, pl->pl_scale->v_name)))

#if defined (XSPICE) && defined (SIMULATOR) /* SIMULATOR: disable old app nutmeg */
/* special case for finding all event nodes and return them as linked vectors */
static struct dvec* findvec_alle(void) {

    struct dvec* d, * newv = NULL, * end = NULL, * v;
    int i, num_nodes;
    Evt_Node_Info_t** node_table;

    /* Look for node name in the event-driven node list */
    num_nodes = g_mif_info.ckt->evt->counts.num_nodes;
    node_table = g_mif_info.ckt->evt->info.node_table;
    if (num_nodes == 0 || !node_table)
        return NULL;

    /* We need to create a new plot because of veccmp() is used */
    struct plot* pl = plot_alloc("digi");
    pl->pl_title = copy("DigitalData");
    pl->pl_name = copy("digital");
    pl->pl_date = copy(datestring());
    plot_new(pl);

    /* find all event data, create vectors, link them to v_link2 */
    for (i = 0; i < num_nodes; i++) {
        char* name = node_table[i]->name;
        d = EVTfindvec(name);
        if (!d)
            continue;
        /* nothing to plot */
        if (d->v_length == 1)
            continue;
        d->v_plot = pl;
        d->v_plot->pl_typename = copy("dig1");
        if (d->v_link2) {
            v = vec_copy(d);
            vec_new(v);
        }
        else {
            v = d;
        }
        if (end) {
            end->v_link2 = v;
         }
         else {
            newv = v;
         }
         end = v;
    }
    return newv;
}
#endif

#ifndef SIMULATOR
static struct dvec* findvec_alle(void) {
    return NULL;
}
#endif

/* Find a permanent vector with the given name */
static struct dvec *find_permanent_vector_by_name(
        NGHASHPTR pl_lookup_table, char *name)
{
    struct dvec *d;
    /* Find the first vector with the given name and then find others
     * until one having the VF_PERMANENT flag set is found. */
    for (d = nghash_find(pl_lookup_table, name);
            d;
            d = nghash_find_again(pl_lookup_table, name)) {
        if (d->v_flags & VF_PERMANENT) {
            /* A "permanent" vector was found with the name, so done */
            return d;
        }
    } /* end of loop over vectors in the plot having this name */
    /* try again, this time without quotes around the name */
    char *nname = cp_unquote(name);
    for (d = nghash_find(pl_lookup_table, nname);
            d;
            d = nghash_find_again(pl_lookup_table, nname)) {
        if (d->v_flags & VF_PERMANENT) {
            /* A "permanent" vector was found with the name, so done */
            tfree(nname);
            return d;
        }
    } /* end of loop over vectors in the plot having this name */
    tfree(nname);
    return (struct dvec *) NULL; /* not found */
} /* end of function find_permanent_vector_by_name */



/* If there are imbedded numeric strings, compare them numerically, not
 * alphabetically.
 */
static int namecmp(const void *a, const void *b)
{
    int i, j;

    const char *s = (const char *) a;
    const char *t = (const char *) b;

    for (;;) {
        while ((*s == *t) && !isdigit_c(*s) && *s)
            s++, t++;
        if (!*s)
            return (0);
        if ((*s != *t) && (!isdigit_c(*s) || !isdigit_c(*t)))
            return (*s - *t);

        /* The beginning of a number... Grab the two numbers and then
         * compare them...  */
        for (i = 0; isdigit_c(*s); s++)
            i = i * 10 + *s - '0';
        for (j = 0; isdigit_c(*t); t++)
            j = j * 10 + *t - '0';

        if (i != j)
            return (i - j);
    }
}


static int
veccmp(const void *a, const void *b)
{
    int i;
    struct dvec **d1 = (struct dvec **) a;
    struct dvec **d2 = (struct dvec **) b;

    if ((i = namecmp((*d1)->v_plot->pl_typename,
                     (*d2)->v_plot->pl_typename)) != 0)
        return (i);
    return (namecmp((*d1)->v_name, (*d2)->v_name));
}


/* Sort all the vectors in d, first by plot name and then by vector
 * name.  Do the right thing with numbers.  */
static struct dvec *
sortvecs(struct dvec *d) {
    struct dvec **array, *t;
    int i, j;

    for (t = d, i = 0; t; t = t->v_link2)
        i++;
    if (i < 2)
        return (d);
    array = TMALLOC(struct dvec *, i);
    for (t = d, i = 0; t; t = t->v_link2)
        array[i++] = t;

    qsort(array, (size_t) i, sizeof(struct dvec *), veccmp);

    /* Now string everything back together... */
    for (j = 0; j < i - 1; j++)
        array[j]->v_link2 = array[j + 1];
    array[j]->v_link2 = NULL;
    d = array[0];
    tfree(array);
    return (d);
}


/* Load in a rawfile. */
void
ft_loadfile(char *file)
{
    struct plot *pl, *np, *pp;

    fprintf(cp_out, "Loading raw data file (\"%s\") ...\n", file);
    pl = raw_read(file);
    if (pl)
        fprintf(cp_out, "done.\n");
    else
        fprintf(cp_out, "no data read.\n");

    /* This is a minor annoyance -- we should reverse the plot list so
     * they get numbered in the correct order.
     */
    for (pp = pl, pl = NULL; pp; pp = np) {
        np = pp->pl_next;
        pp->pl_next = pl;
        pl = pp;
    }
    for (; pl; pl = np) {
        np = pl->pl_next;
        plot_add(pl);
        /* Don't want to get too many "plot not written" messages. */
        pl->pl_written = TRUE;
    }
    plot_num++;
    plotl_changed = TRUE;
}


void
plot_add(struct plot *pl)
{
    struct dvec *v;
    struct plot *tp;
    char *s, buf[BSIZE_SP];

    fprintf(cp_out, "Title:  %s\nName: %s\nDate: %s\n\n", pl->pl_title,
            pl->pl_name, pl->pl_date);

    if (plot_cur)
        plot_cur->pl_ccom = cp_kwswitch(CT_VECTOR, pl->pl_ccom);

    for (v = pl->pl_dvecs; v; v = v->v_next)
        cp_addkword(CT_VECTOR, v->v_name);
    cp_addkword(CT_VECTOR, "all");

    if ((s = ft_plotabbrev(pl->pl_name)) == NULL)
        s = "unknown";
    do {
        (void) sprintf(buf, "%s%d", s, plot_num);
        for (tp = plot_list; tp; tp = tp->pl_next)
            if (cieq(tp->pl_typename, buf)) {
                plot_num++;
                break;
            }
    } while (tp);

    pl->pl_typename = copy(buf);
    plot_new(pl);
    cp_addkword(CT_PLOT, buf);
    pl->pl_ccom = cp_kwswitch(CT_VECTOR, NULL);
    plot_setcur(pl->pl_typename);
}


/* Remove a vector from the database, if it is there. */

void
vec_remove(const char *name)
{
    struct dvec *ov;

    for (ov = plot_cur->pl_dvecs; ov; ov = ov->v_next)
        if (cieq(name, ov->v_name) && (ov->v_flags & VF_PERMANENT))
            break;

    if (!ov)
        return;

    ov->v_flags &= ~VF_PERMANENT;

    /* Remove from the keyword list. */
    cp_remkword(CT_VECTOR, name);
}


/* Get a vector by name. This deals with v(1), etc. almost properly. Also,
 * it checks for pre-defined vectors.
 */

struct dvec *vec_fromplot(char *word, struct plot *plot) {
    struct dvec *d = findvec(word, plot);
    if (d != (struct dvec *) NULL) {
        return d;
    }

    /* Forms I(node) and i(node) are converted to node#branch;
     * forms x(node), x != i, x != I, and x != '(' are converted to node */
    if (word[0] != '\0' && word[0] != '(') { /* 1 or more char, not '(' */
        if (word[1] == '(') { /* x(, x != '(' */
            const char * const p_last_close_paren = strrchr(word + 2, ')');
            if (p_last_close_paren != (char *) NULL &&
                    p_last_close_paren - word > (ptrdiff_t) 2 &&
                    p_last_close_paren[1] == '\0') {
                /* Of form x(node). Create node string. */
                DS_CREATE(ds, 100);
                const char * const node_start = word + 2;
                bool ds_ok = ds_cat_mem(&ds, node_start,
                        (size_t) (p_last_close_paren - node_start)) ==
                        DS_E_OK;
                /* If i(node) or I(node), append #branch */
                if (tolower(word[0]) == (int) 'i') {
                    /* i(node) or I(node) */
                    ds_ok &= ds_cat_mem(&ds, "#branch", 7) == DS_E_OK;
                }
                if (!ds_ok) { /* Dstring error (allocation failure) */
                    (void) fprintf(cp_err, "Unable to build vector name.\n");
                }
                else { /* name built OK */
                    d = findvec(ds_get_buf(&ds), plot);
                } /* end of case of vector name built OK */
                ds_free(&ds);
            } /* end of case of x(node) */
        } /* end of case of x( */
    } /* end of case of non-empty string and not leading '(' */

    return d;
} /* end of function vec_fromplot */



/* This is the main lookup routine for names. The possible types of names are:
 *  name        An ordinary vector.
 *  plot.name   A vector from a particular plot.
 *  @device[parm]   A device parameter.
 *  @model[parm]    A model parameter.
 *  @param      A circuit parameter.
 * For the @ cases, we construct a dvec with length 1 to hold the value.
 * In the other two cases, either the plot or the name can be "all", a
 * wildcard.
 * The vector name may have imbedded dots -- if the first component is a plot
 * name, it is considered the plot, otherwise the current plot is used.
 */

#define SPECCHAR '@'

struct dvec *
vec_get(const char *vec_name) {
    struct dvec *d, *end = NULL, *newv = NULL;
    struct plot *pl;
    char buf[BSIZE_SP], *s, *wd, *word, *whole, *name = NULL, *param;
    int i = 0;
    struct variable *vv;

    wd = word = copy(vec_name);   /* Gets mangled below... */

    if (strchr(word, '.')) {
        /* Snag the plot... */
        for (i = 0, s = word; *s != '.'; i++, s++)
            buf[i] = *s;
        buf[i] = '\0';
        if (cieq(buf, "all")) {
            word = ++s;
            pl = NULL;  /* NULL pl signifies a wildcard. */
        } else {
            for (pl = plot_list;
                    pl && !plot_prefix(buf, pl->pl_typename);
                    pl = pl->pl_next)
                ;
            if (pl) {
                word = ++s;
            } else {
                /* This used to be an error... */
                pl = plot_cur;
            }
        }
    } else {
        pl = plot_cur;
    }

    if (pl) {
        d = vec_fromplot(word, pl);
        if (!d)
            d = vec_fromplot(word, &constantplot);
    } else {
        for (pl = plot_list; pl; pl = pl->pl_next) {
            if (cieq(pl->pl_typename, "const"))
                continue;
            d = vec_fromplot(word, pl);
            if (d) {
                if (end)
                    end->v_link2 = d;
                else
                    newv = d;
                for (end = d; end->v_link2; end = end->v_link2)
                    ;
            }
        }
        d = newv;
        if (!d) {
            fprintf(cp_err,
                    "Error: plot wildcard (name %s) matches nothing\n",
                    word);
            tfree(wd); /* MW. I don't want core leaks here */
            return (NULL);
        }
    }

    if (!d && (*word == SPECCHAR)) { /* "@" */
        /* This is a special quantity... */
        if (ft_nutmeg) {
            fprintf(cp_err,
                    "Error: circuit parameters only available with spice\n");
            tfree(wd);  /* MW. Memory leak fixed again */
            return (NULL); /* va: use NULL */
        }

        whole = copy(word);
        name = ++word;
        for (param = name; *param && (*param != '['); param++)
            ;

        if (*param) {
            *param++ = '\0';
            for (s = param; *s && *s != ']'; s++)
                ;
            *s = '\0';
        } else {
            param = NULL;
        }


        if (ft_curckt) {
            /*
             *  This is what is done in case of "alter r1 resistance = 1234"
             *                                r1    resistance, 0
             * if_setparam(ft_curckt->ci_ckt, &dev, param, dv, do_model);
             */

            /* vv = if_getparam (ft_curckt->ci_ckt, &name, param, 0, 0); */
            vv = if_getparam(ft_curckt->ci_ckt, &name, param, 0, 0);
            if (!vv) {
                tfree(whole);
                tfree(wd);
                return (NULL);
            }
        } else {
            fprintf(cp_err, "Error: No circuit loaded.\n");
            tfree(whole);
            tfree(wd);
            return (NULL);
        }

        d = dvec_alloc(copy(whole), /* MW. The same as word before */
                       SV_NOTYPE,
                       VF_REAL,  /* No complex values yet... */
                       1, NULL);

        /* In case the represented variable is a REAL vector this takes
         * the actual value of the first element of the linked list which
         * does not make sense.
         * This is an error.
         */

        /* This will copy the contents of the structure vv in another structure
         * dvec (FTEDATA.H) that do not have INTEGER so that those parameters
         * defined as IF_INTEGER are not given their value when using
         * print @pot[pos_node]
         * To fix this, it is necessary to define:
         * OPU( "pos_node",    POT_QUEST_POS_NODE, IF_REAL,"Positive node of potenciometer"),
         * int POTnegNode;     // number of negative node of potenciometer (Nodo_3)
         *  case POT_QUEST_POS_NODE:
         *  value->rValue = (double)fast->POTposNode;
         *  return (OK);
         *  Works but with the format 1.00000E0
         */

        /* We must make a change in format between the data that carries a variable to
         * put in a dvec.
         */

        /*
         * #define va_bool    va_V.vV_bool
         * #define va_num     va_V.vV_num
         * #define va_real    va_V.vV_real
         * #define va_string  va_V.vV_string
         * #define va_vlist   va_V.vV_list
         * enum cp_types {
         *   CP_BOOL,
         *   CP_NUM,
         *   CP_REAL,
         *   CP_STRING,
         *   CP_LIST
         ° };
        */

        /* The variable is a vector */
        if (vv->va_type == CP_LIST) {
            /* Compute the length of the vector,
             * used with the parameters of isrc and vsrc
             */
            struct variable *nv;

            /* Count the number of nodes in the list */
            i = 0;
            for (nv = vv->va_vlist; nv; nv = nv->va_next) {
                i++;
            }

            dvec_realloc(d, i, NULL); /* Resize to # nodes */

            /* Step through the list again, setting values this time */
            i = 0;
            for (nv = vv->va_vlist; nv; nv = nv->va_next) {
                d->v_realdata[i++] = nv->va_real;
            }

            /* To be able to identify the vector to represent
             * belongs to a special "conunto" and should be printed in a
             * special way.
             */
            d->v_dims[1] = 1;
        }
        else if (vv->va_type == CP_NUM) { /* Variable is an integer */
            *d->v_realdata = (double) vv->va_num;
        }
        else if (vv->va_type == CP_REAL) { /* Variable is a real */
            if (!(vv->va_next)) {
                /* Only a real data
                 * usually normal
                 */
                *d->v_realdata = vv->va_real;
            }
            else {
                /* Real data set
                 * When you print a model @ [all]
                 * Just print numerical values, not the string
                 */
                struct variable *nv;
                /* We go to print the list of values
                 * nv->va_name = Parameter description
                 * nv->va_string = Parameter
                 * nv->va_real= Value
                 */
                nv = vv;
                for (i = 1; ; i++) {
                    switch (nv->va_type) {
                    case  CP_REAL:
                        fprintf(stdout, "%s=%g\n", nv->va_name, nv->va_real);
                        break;
                    case  CP_STRING:
                        fprintf(stdout, "%s=%s\n", nv->va_name, nv->va_string);
                        break;
                    case  CP_NUM:
                        fprintf(stdout, "%s=%d\n", nv->va_name, nv->va_num);
                        break;
                    case  CP_BOOL:
                        fprintf(stdout, "%s=%d\n", nv->va_name, nv->va_bool);
                        break;
                    default: {
                        fprintf(stderr, "ERROR: enumeration value `CP_LIST' not handled in vec_get\nIgnoring...\n");
                        break;
                    }
                    }
                    nv = nv->va_next;

                    if (!nv) {
                        break;
                    }
                }

                /* To distinguish those does not take anything for print screen to
                 * make a print or M1 @ @ M1 [all] leaving only the correct data
                 * and not the last
                 */
                d->v_rlength = 1;
            }
        }

        free_struct_variable(vv);
        tfree(wd);
        vec_new(d);
        tfree(whole);
        return d;
    }

    tfree(wd);
    return (sortvecs(d));
}


/* Execute the commands for a plot. This is done whenever a plot becomes
 * the current plot.
 */

void
plot_docoms(wordlist *wl)
{
    bool inter;

    inter = cp_interactive;
    cp_interactive = FALSE;
    while (wl) {
        (void) cp_evloop(wl->wl_word);
        wl = wl->wl_next;
    }
    cp_resetcontrol(TRUE);
    cp_interactive = inter;
}


/* Create a copy of a vector. The vector is not "permananent" */
struct dvec *vec_copy(struct dvec *v) {
    struct dvec *nv;

    if (!v) {
        return (struct dvec *) NULL;
    }

    /* Make a copy with the VF_PERMANENT bit cleared in v_flags */
    nv = dvec_alloc(copy(v->v_name),
                    v->v_type,
                    v->v_flags & ~VF_PERMANENT,
                    v->v_length, NULL);

    /* Copy the data to the new vector */
    if (isreal(v)) {
        (void) memcpy(nv->v_realdata, v->v_realdata,
               sizeof(double) * (size_t) v->v_length);
    }
    else {
        (void) memcpy(nv->v_compdata, v->v_compdata,
               sizeof(ngcomplex_t) * (size_t) v->v_length);
    }

    nv->v_minsignal = v->v_minsignal;
    nv->v_maxsignal = v->v_maxsignal;
    nv->v_gridtype = v->v_gridtype;
    nv->v_plottype = v->v_plottype;

    /* Modified to copy the rlength of origin to destination vecor
     * instead of always putting it to 0.
     * As when it comes to make a print does not leave M1 @ @ M1 = 0.0,
     * to do so in the event that rlength = 0 not print anything on screen
     * nv-> v_rlength = 0;
     * Default -> v_rlength = 0 and only if you come from a print or M1 @
     * @ M1 [all] rlength = 1, after control is one of
     * if (v-> v_rlength == 0) com_print (wordlist * wl)
     */
    nv->v_rlength = v->v_rlength;

    nv->v_outindex = 0; /*XXX???*/
    nv->v_linestyle = 0; /*XXX???*/
    nv->v_color = 0; /*XXX???*/
    nv->v_defcolor = v->v_defcolor;
    nv->v_numdims = v->v_numdims;

    /* Copy defined dimensions */
    (void) memcpy(nv->v_dims, v->v_dims,
            (size_t) v->v_numdims * sizeof *v->v_dims);

    nv->v_plot = v->v_plot;
    nv->v_next = NULL;
    nv->v_link2 = NULL;
    nv->v_scale = v->v_scale;

    return nv;
} /* end of function vec_copy */



/* Create a new plot structure. This just fills in the typename and sets up
 * the ccom struct.
 */

struct plot * plot_alloc(char *name)
{
    struct plot *pl = TMALLOC(struct plot, 1), *tp;
    char *s;
    struct ccom *ccom;
    char buf[BSIZE_SP];

    ZERO(pl, struct plot);
    if ((s = ft_plotabbrev(name)) == NULL)
        s = "unknown";
    do {
        (void) sprintf(buf, "%s%d", s, plot_num);
        for (tp = plot_list; tp; tp = tp->pl_next)
            if (cieq(tp->pl_typename, buf)) {
                plot_num++;
                break;
            }
    } while (tp);
    pl->pl_typename = copy(buf);
    cp_addkword(CT_PLOT, buf);
    /* va: create a new, empty keyword tree for class CT_VECTOR, s=old tree */
    ccom = cp_kwswitch(CT_VECTOR, NULL);
    cp_addkword(CT_VECTOR, "all");
    pl->pl_ccom = cp_kwswitch(CT_VECTOR, ccom);
    /* va: keyword tree is old tree again, new tree is linked to pl->pl_ccom */
    return (pl);
}


/* Stick a new vector in the proper place in the plot list. */

void
vec_new(struct dvec *d)
{
#ifdef FTEDEBUG
    if (ft_vecdb)
        fprintf(cp_err, "new vector %s\n", d->v_name);
#endif
    /* Note that this can't happen. */
    if (plot_cur == NULL) {
        fprintf(cp_err, "vec_new: Internal Error: no cur plot\n");
    }
    else {
        plot_cur->pl_lookup_valid = FALSE;
        if ((d->v_flags & VF_PERMANENT) && (plot_cur->pl_scale == NULL)) {
            plot_cur->pl_scale = d;
        }
        if (!d->v_plot) {
            d->v_plot = plot_cur;
        }
    }

    /* This code appears to be a patch for incorrectly specified vectors */
    if (d->v_numdims < 1) {
        d->v_numdims = 1;
        d->v_dims[0] = d->v_length;
    }

    {
        /* Make this vector the first plot vector and link the old first plot
         * vector via its next pointer */
        struct plot *v_plot = d->v_plot;
        d->v_next = v_plot->pl_dvecs;
        v_plot->pl_dvecs = d;
    }
}


/* Because of the way that all vectors, including temporary vectors,
 * are linked together under the current plot, they can often be
 * left lying around. This gets rid of all vectors that don't have
 * the permanent flag set. Also, for the remaining vectors, it
 * clears the v_link2 pointer.
 */

void
vec_gc(void)
{
    struct dvec *d, *nd;
    struct plot *pl;

    for (pl = plot_list; pl; pl = pl->pl_next)
        for (d = pl->pl_dvecs; d; d = nd) {
            nd = d->v_next;
            if (!(d->v_flags & VF_PERMANENT)) {
                if (ft_vecdb)
                    fprintf(cp_err,
                            "vec_gc: throwing away %s.%s\n",
                            pl->pl_typename, d->v_name);
                vec_free(d);
            }
        }

    for (pl = plot_list; pl; pl = pl->pl_next)
        for (d = pl->pl_dvecs; d; d = d->v_next)
            d->v_link2 = NULL;
}


/* Free a dvector. This is sort of a pain because we also have to make sure
 * that it has been unlinked from its plot structure. If the name of the
 * vector is NULL, then we have already freed it so don't try again. (This
 * situation can happen with user-defined functions.) Note that this depends
 * on our having tfree set its argument to NULL. Note that if all the vectors
 * in a plot are gone it stays around...
 */

void vec_free_x(struct dvec *v)
{
    /* Do not free if NULL or name is NULL. The second possibility is a
     * special case */
    if ((v == NULL) || (v->v_name == NULL)) {
        return;
    }
    struct plot * const pl = v->v_plot;

    /* Now we have to take this dvec out of the plot list. */
    if (pl != NULL) {
        pl->pl_lookup_valid = FALSE;

        /* If at head of list of vectors in the plot, make the next one
         * the new head of the list */
        if (pl->pl_dvecs == v) {
            pl->pl_dvecs = v->v_next;
        }
        else {
            /* Not at head of list so must locate and fix links */
            struct dvec *lv = pl->pl_dvecs;
            if (lv) { /* the plot has at least one vector */
                for ( ; lv->v_next; lv = lv->v_next) {
                    if (lv->v_next == v) { /* found prev vector */
                        break;
                    }
                }
            }

            /* If found in the list, link prev vector to next one */
            if (lv && lv->v_next) {
                lv->v_next = v->v_next;
            }
            else {
                (void) fprintf(cp_err,
                        "vec_free: Internal Error: %s not in plot\n",
                        v->v_name);
            }
        } /* end of case that vector being freed is not at head of list */

        if (pl->pl_scale == v) {
            if (pl->pl_dvecs) {
                pl->pl_scale = pl->pl_dvecs;    /* Random one... */
            }
            else {
                pl->pl_scale = NULL;
            }
        }
    } /* end of case that have a plot */

    dvec_free(v);
} /* end of function vec_free_x */



/* This function returns TRUE if every element of v and every element of
 * every vector linked to v through v_link2 is zero and FALSE otherwise. */
bool vec_iszero(const struct dvec *v)
{
    for (; v; v = v->v_link2) { /* step through linked vectors */
        if (isreal(v)) { /* current vector is real */
            const int n = v->v_length;
            int i;
            for (i = 0; i < n; i++) {
                if (v->v_realdata[i] != 0.0) {
                    return FALSE;
                }
            }
        }
        else { /* current vector is complex */
            const int n = v->v_length;
            int i;
            for (i = 0; i < n; i++) {
                if (realpart(v->v_compdata[i]) != 0.0) {
                    return FALSE;
                }
                if (imagpart(v->v_compdata[i]) != 0.0) {
                    return FALSE;
                }
            }
        }
    }

    return TRUE; /* every value tested was 0.0 */
} /* end of function vec_iszero */



/* This is something we do in a few places...  Since vectors get copied a lot,
 * we can't just compare pointers to tell if two vectors are 'really' the same.
 */

bool
vec_eq(struct dvec *v1, struct dvec *v2)
{
    char *s1, *s2;
    bool rtn;

    if (v1->v_plot != v2->v_plot)
        return (FALSE);

    s1 = vec_basename(v1);
    s2 = vec_basename(v2);

    if (cieq(s1, s2))
        rtn = TRUE;
    else
        rtn = FALSE;

    tfree(s1);
    tfree(s2);
    return rtn;
}


/* Return the name of the vector with the plot prefix stripped off.  This
 * is no longer trivial since '.' doesn't always mean 'plot prefix'.
 */

char *
vec_basename(struct dvec *v)
{
    char buf[BSIZE_SP], *t, *s;

    if (strchr(v->v_name, '.')) {
        if (cieq(v->v_plot->pl_typename, v->v_name))
            (void) strcpy(buf, v->v_name + strlen(v->v_name) + 1);
        else
            (void) strcpy(buf, v->v_name);
    } else {
        (void) strcpy(buf, v->v_name);
    }

    strtolower(buf);
    s = skip_ws(buf);
    for (t = s; *t; t++)
        ;
    while ((t > s) && isspace_c(t[-1]))
        *--t = '\0';
    return (copy(s));
}

/* get address of plot named 'name' */
struct plot *get_plot(const char *name)


{
    struct plot *pl;
    for (pl = plot_list; pl; pl = pl->pl_next) {
        if (plot_prefix(name, pl->pl_typename)) {
            return pl;
        }
    }
    fprintf(cp_err, "Error: no such plot named %s\n", name);
    return (struct plot *) NULL;
} /* end of function get_plot */



/* Make a plot the current one.  This gets called by cp_usrset() when one
 * does a 'set curplot = name'.
 * va: ATTENTION: has unlinked old keyword-class-tree from keywords[CT_VECTOR]
 *                (potentially memory leak)
 */
void plot_setcur(const char *name)
{
    struct plot *pl;

    if (cieq(name, "new")) {
        pl = plot_alloc("unknown");
        pl->pl_title = copy("Anonymous");
        pl->pl_name = copy("unknown");
        pl->pl_date = copy(datestring());
        plot_new(pl);
        plot_cur = pl;
        return;
    }
    /* plots are listed in pl in reverse order */
    else if (cieq(name, "previous")) {
        if (plot_cur->pl_next) {
            plot_cur = plot_cur->pl_next;
#ifdef XSPICE
            if (ft_curckt) {
                EVTswitch_plot(ft_curckt->ci_ckt, plot_cur->pl_typename);
            }
#endif
        }
        else {
            fprintf(cp_err,
                    "Warning: No previous plot is available. "
                    "Plot remains unchanged (%s).\n",
                    plot_cur->pl_typename);
        }
        return;
    }
    else if (cieq(name, "next")) {
        /* Step through the list, which has plots in reverse order */
        struct plot *prev_pl = NULL;
        for (pl = plot_list; pl; pl = pl->pl_next) {
            if (pl == plot_cur) {
                break;
            }
            prev_pl = pl;
        }
        if (prev_pl) { /* found */
            plot_cur = prev_pl;
#ifdef XSPICE
            if (ft_curckt) {
                EVTswitch_plot(ft_curckt->ci_ckt, plot_cur->pl_typename);
            }
#endif
        }
        else { /* no next plot */
            fprintf(cp_err,
                    "Warning: No next plot is available. "
                    "Plot remains unchanged (%s).\n",
                    plot_cur->pl_typename);
        }
        return;
    }

    pl = get_plot(name);
    if (!pl) {
        return;
    }

    /* va: we skip cp_kwswitch, because it confuses the keyword-tree management for
     *     repeated op-commands. When however cp_kwswitch is necessary for other
     *     reasons, we should hold the original keyword table pointer in an
     *     permanent variable, since it will lost here, and can never tfree'd.
     if (plot_cur)
     {
     plot_cur->pl_ccom = cp_kwswitch(CT_VECTOR, pl->pl_ccom);
     }
    */
#ifdef XSPICE
    /* XSPICE event data are linked to the current circuit. It must not be removed
       when manipulating the data by any command.
    */
    if (ft_curckt) {
        EVTswitch_plot(ft_curckt->ci_ckt, name);
    }
#endif
    plot_cur = pl;
} /* end of function plot_setcur */



/* Add a plot to the plot list. This is different from plot_add() in that
 * all this does is update the list and the variable $plots.
 */
void plot_new(struct plot *pl)
{
    pl->pl_next = plot_list;
    plot_list = pl;
}


/* This routine takes a multi-dimensional vector, treats it as a
 * group of 2-dimensional matrices and transposes each matrix.
 * The data array is replaced with a new one that has the elements
 * in the proper order.  Otherwise the transposition is done in place.
 */

void
vec_transpose(struct dvec *v)
{
    int dim0, dim1, nummatrices;
    int i, j, k, joffset, koffset, blocksize;
    double *newreal, *oldreal;
    ngcomplex_t *newcomp, *oldcomp;

    if (v->v_numdims < 2 || v->v_length <= 1)
        return;

    dim0 = v->v_dims[v->v_numdims-1];
    dim1 = v->v_dims[v->v_numdims-2];
    v->v_dims[v->v_numdims-1] = dim1;
    v->v_dims[v->v_numdims-2] = dim0;
    /* Assume length is a multiple of each dimension size.
     * This may not be safe, in which case a test should be
     * made that the length is the product of all the dimensions.
     */
    blocksize = dim0*dim1;
    nummatrices = v->v_length / blocksize;

    /* Note:
     *   olda[i,j] is at data[i*dim0+j]
     *   newa[j,i] is at data[j*dim1+i]
     *   where j is in [0, dim0-1]  and  i is in [0, dim1-1]
     * Since contiguous data in the old array is scattered in the new array
     * we can't use memcpy :(.  There is probably a BLAS2 function for this
     * though.  The formulation below gathers scattered old data into
     * consecutive new data.
     */

    if (isreal(v)) {
        newreal = TMALLOC(double, v->v_length);
        oldreal = v->v_realdata;
        koffset = 0;
        for (k = 0; k < nummatrices; k++) {
            joffset = 0;
            for (j = 0; j < dim0; j++) {
                for (i = 0; i < dim1; i++) {
                    newreal[ koffset + joffset + i ] =
                        oldreal[ koffset + i*dim0 + j ];
                }
                joffset += dim1;  /* joffset = j*dim0 */
            }
            koffset += blocksize; /* koffset = k*blocksize = k*dim0*dim1 */
        }
        dvec_realloc(v, v->v_length, newreal);
    } else {
        newcomp = TMALLOC(ngcomplex_t, v->v_length);
        oldcomp = v->v_compdata;
        koffset = 0;
        for (k = 0; k < nummatrices; k++) {
            joffset = 0;
            for (j = 0; j < dim0; j++) {
                for (i = 0; i < dim1; i++) {
                    newcomp[ koffset + joffset + i ] =
                        oldcomp[ koffset + i*dim0 + j ];
                }
                joffset += dim1;  /* joffset = j*dim0 */
            }
            koffset += blocksize; /* koffset = k*blocksize = k*dim0*dim1 */
        }
        dvec_realloc(v, v->v_length, newcomp);
    }
}


/* This routine takes a multi-dimensional vector and turns it into a family
 * of 1-d vectors, linked together with v_link2.  It is here so that plot
 * can do intelligent things.
 */

struct dvec *
vec_mkfamily(struct dvec *v) {
    int size, numvecs, i, count[MAXDIMS];
    struct dvec *vecs, *d, **t;
    char buf2[BSIZE_SP];

    if (v->v_numdims < 2)
        return (v);

    size = v->v_dims[v->v_numdims - 1];
    for (i = 0, numvecs = 1; i < v->v_numdims - 1; i++)
        numvecs *= v->v_dims[i];
    for (i = 0; i < MAXDIMS; i++)
        count[i] = 0;
    for (t = &vecs, i = 0; i < numvecs; i++) {

        indexstring(count, v->v_numdims - 1, buf2);

        d = dvec_alloc(tprintf("%s%s", v->v_name, buf2),
                       v->v_type,
                       v->v_flags,
                       size, NULL);

        d->v_minsignal = v->v_minsignal;
        d->v_maxsignal = v->v_maxsignal;
        d->v_gridtype = v->v_gridtype;
        d->v_plottype = v->v_plottype;
        d->v_scale = v->v_scale;
        /* Don't copy the default color, since there will be many
         * of these things...
         */
        d->v_numdims = 1;
        d->v_dims[0] = size;

        if (isreal(v)) {
            memcpy(d->v_realdata, v->v_realdata + (size_t) (size * i),
                    (size_t) size * sizeof(double));
        } else {
            memcpy(d->v_compdata, v->v_compdata + (size_t) (size * i),
                    (size_t) size * sizeof(ngcomplex_t));
        }
        /* Add one to the counter. */
        (void) incindex(count, v->v_numdims - 1, v->v_dims, v->v_numdims);

        *t = d;
        t = &(d->v_link2);
    }

    for (d = vecs; d; d = d->v_link2)
        vec_new(d);

    return (vecs);
}


/* This function will match "op" with "op1", but not "op1" with "op12". */
static bool plot_prefix(const char *pre, const char *str)
{
    if (!*pre) { /* prefix is empty string */
        return TRUE; /* Define "" to be prefix */
    }

    while (*pre && *str) {
        if (*pre != *str) { /* stop at first mismatch */
            break;
        }
        pre++;
        str++;
    }

    if (*pre || (*str && isdigit_c(pre[-1])))
        return (FALSE);
    else
        return (TRUE);
}

struct dvec*
copycut(struct dvec* v, struct dvec* newscalevec, int istart, int istop)
{
    struct dvec* nv;
    int i;
    int len = istop - istart;

    if (!v) {
        return (struct dvec*)NULL;
    }

    /* Make a copy with the VF_PERMANENT bit cleared in v_flags */
    nv = dvec_alloc(copy(v->v_name),
        v->v_type,
        v->v_flags,// & ~VF_PERMANENT,
        len, NULL);

    /* Copy the data to the new vector */
    if (isreal(v)) {
        for (i = 0; i < len; i++) {
            nv->v_realdata[i] = v->v_realdata[istart + i];
        }
    }
    else {
        for (i = 0; i < len; i++) {
            nv->v_compdata[i] = v->v_compdata[istart + i];
        }
    }

    nv->v_minsignal = v->v_minsignal;
    nv->v_maxsignal = v->v_maxsignal;
    nv->v_gridtype = v->v_gridtype;
    nv->v_plottype = v->v_plottype;

    /* Modified to copy the rlength of origin to destination vecor
        * instead of always putting it to 0.
        * As when it comes to make a print does not leave M1 @ @ M1 = 0.0,
        * to do so in the event that rlength = 0 not print anything on screen
        * nv-> v_rlength = 0;
        * Default -> v_rlength = 0 and only if you come from a print or M1 @
        * @ M1 [all] rlength = 1, after control is one of
        * if (v-> v_rlength == 0) com_print (wordlist * wl)
        */
    nv->v_rlength = v->v_rlength;

    nv->v_outindex = 0; /*XXX???*/
    nv->v_linestyle = 0; /*XXX???*/
    nv->v_color = 0; /*XXX???*/
    nv->v_defcolor = v->v_defcolor;
    nv->v_numdims = v->v_numdims;

    /* Copy defined dimensions */
    (void)memcpy(nv->v_dims, v->v_dims,
        (size_t)v->v_numdims * sizeof * v->v_dims);

    nv->v_plot = newscalevec->v_plot;
    nv->v_next = NULL;
    nv->v_link2 = NULL;

    return nv;
} /* end of function copycut */
