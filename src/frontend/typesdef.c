/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1986 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Stuff to deal with nutmeg "types" for vectors and plots.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/sim.h"
#include "typesdef.h"


#define NUMTYPES (128 + 4)  /* If this is too little we can use a list. */
#define NUMPLOTTYPES 512    /* Since there may be more than 1 pat/type. */

struct type {
    char *t_name;
    char *t_abbrev;
    bool f_name_is_alloc; /* Flag that name was allocated */
    bool f_abbrev_is_alloc; /* Flag that abbrev was allocated */
};

/* The stuff for plot names. */

struct plotab {
    char *p_name;
    char *p_pattern;
    bool f_name_is_alloc; /* Flag that name was allocated */
    bool f_pattern_is_alloc; /* Flag that pattern was allocated */
};

/* note:  This should correspond to SV_ defined in sim.h */
static struct type types[NUMTYPES] = {
    { "notype", NULL, FALSE, FALSE },
    { "time", "s", FALSE, FALSE },
    { "frequency", "Hz", FALSE, FALSE },
    { "voltage", "V", FALSE, FALSE },
    { "current", "A", FALSE, FALSE },
    { "voltage-density", "V/sqrt(Hz)", FALSE, FALSE },
    { "current-density", "A/sqrt(Hz)", FALSE, FALSE },
    { "voltage^2-density", "(V^2)/Hz", FALSE, FALSE },
    { "current^2-density", "(A^2)/Hz", FALSE, FALSE },
    { "voltage^2", "(V^2)", FALSE, FALSE },
    { "current^2", "(A^2)", FALSE, FALSE },
    { "pole", NULL, FALSE, FALSE },
    { "zero", NULL, FALSE, FALSE },
    { "s-param", NULL, FALSE, FALSE },
    { "temp-sweep", "Celsius", FALSE, FALSE },
    { "res-sweep", "Ohms", FALSE, FALSE },
    { "impedance", "Ohms", FALSE, FALSE },
    { "admittance", "Mhos", FALSE, FALSE },
    { "power", "W", FALSE, FALSE },
    { "phase", "rad", FALSE, FALSE },
    { "decibel", "dB", FALSE, FALSE },
    { "capacitance", "F", FALSE, FALSE },
    { "charge", "C", FALSE, FALSE },
    { "temperature", "Celsius", FALSE, FALSE }
};

/* The stuff for plot names. */

static struct plotab plotabs[NUMPLOTTYPES] = {
    { "tran", "transient", FALSE, FALSE },
    { "op", "op", FALSE, FALSE },
    { "tf", "function", FALSE, FALSE },
    { "dc", "d.c.", FALSE, FALSE },
    { "dc", "dc", FALSE, FALSE },
    { "dc", "transfer", FALSE, FALSE },
    { "ac", "a.c.", FALSE, FALSE },
    { "ac", "ac", FALSE, FALSE },
    { "pz", "pz", FALSE, FALSE },
    { "pz", "p.z.", FALSE, FALSE },
    { "pz", "pole-zero", FALSE, FALSE},
    { "disto", "disto", FALSE, FALSE },
    { "dist", "dist", FALSE, FALSE },
    { "noise", "noise", FALSE, FALSE },
    { "sens", "sens", FALSE, FALSE },
    { "sens", "sensitivity", FALSE, FALSE },
    { "sens2", "sens2", FALSE, FALSE },
    { "sp", "s.p.", FALSE, FALSE },
    { "sp", "sp", FALSE, FALSE },
    { "harm", "harm", FALSE, FALSE },
    { "spect", "spect", FALSE, FALSE },
    { "pss", "periodic", FALSE, FALSE }
};


/* A command to define types for vectors and plots.  This will generally
 * be used in the Command: field of the rawfile.
 * The syntax is "deftype v typename abbrev", where abbrev will be used to
 * parse things like abbrev(name) and to label axes with M<abbrev>, instead
 * of numbers. It may be ommitted.
 * Also, the command "deftype p plottype pattern ..." will assign plottype as
 * the name to any plot with one of the patterns as its p_name field.
 *
 * Parameter
 * wl: A linked list of strings of words providing arguments to the
 *      deftype command.
 *
 * Remarks
 * The caller must ensure that there are 3 words in the linked
 * list with the v subcommand or the function will cause access violations
 * as it tries to access the required arguments. With the p subcommand,
 * only 2 arguments are strictly required, but the function will not
 * do anything useful unless at least 3 are provided.
 */
void
com_dftype(wordlist *wl)
{
    /* Identify the subcommand and partially validate it */
    const char * const subcmd_word = wl->wl_word;
    const char subcmd_char = *subcmd_word;
    if (subcmd_char == '\0' || subcmd_word[1] != '\0') {
        (void) fprintf(cp_err, "Error: invalid subcommand \"%s\".\n",
                subcmd_word);
        return;
    }

    switch (subcmd_char) {
    case 'v':
    case 'V': {
        wl = wl->wl_next;
        const char * const name = wl->wl_word; /* type name */
        wl = wl->wl_next;
        const char * const abb = wl->wl_word; /* abbreviation */

        /* Test for invalid arguments */
        if ((wl = wl->wl_next) != (wordlist *) NULL) {
            (void) fprintf(cp_err, "Error: extraneous argument%s supplied "
                    "with the v subcommand: \"%s\"",
                    wl->wl_next == (wordlist *) NULL ?  "" : "s",
                    wl->wl_word);
            wl = wl->wl_next;
            for ( ; wl != (wordlist *) NULL; wl = wl->wl_next) {
                (void) fprintf(cp_err, ", \"%s\"", wl->wl_word);
            }
            (void) fprintf(cp_err, "\n");
            return;
        }

        int i;

        /* Sequentially scan :( the defined types until the desired type
         * name is found or all names have been checked. */
        for (i = 0; i < NUMTYPES && types[i].t_name; i++)
            if (cieq(types[i].t_name, name)) 
                break; /* match found at index i */
        if (i == NUMTYPES) { /* array is full */
            fprintf(cp_err, "Error: too many types (%d) defined\n",
                    NUMTYPES);
            return;
        }

        {
            struct type * const type_cur = types + i;
            /* If reached the end of the list of defined types,
             * define a new type */
            if (type_cur->t_name == (char *) NULL) {
                type_cur->t_name = copy(name);
                type_cur->f_name_is_alloc = TRUE;
            }
            else { /* already exists, so may be abbrev to free */
                /* If the abbreviation has already been defined via an
                 * allocated buffer, free the allocation */
                if (type_cur->t_abbrev != (char *) NULL &&
                        type_cur->f_abbrev_is_alloc) {
                    txfree((void *) type_cur->t_abbrev);
                }
            }

            /* Set the new abbreviation */
            type_cur->t_abbrev = copy(abb);
            type_cur->f_abbrev_is_alloc = TRUE;
        }
        break;
    }
    case 'p':
    case 'P': {
        wl = wl->wl_next;
        char * const name = copy(wl->wl_word); /* plot type name */
        bool f_name_used = FALSE; /* flag that name copy alloc is used */

        /* For each pattern supplied in the command, locate it in the
         * list of known patterns or add it if it does not exist and
         * the list is not full. */
        for (wl = wl->wl_next; wl; wl = wl->wl_next) {
            char *pattern = wl->wl_word;
            int i;
            for (i = 0; i < NUMPLOTTYPES && plotabs[i].p_pattern; i++)
                if (cieq(plotabs[i].p_pattern, pattern))
                    break; /* match found at index i */
            if (i == NUMPLOTTYPES) { /* array is full */
                if (!f_name_used) {
                    /* Free the name copy that was never used */
                    txfree((void *) name);
                }
                fprintf(cp_err, "Error: too many plot abs (%d) defined.\n",
                        NUMPLOTTYPES);
                return;
            }

            {
                struct plotab * const plotab_cur = plotabs + i;
                /* If reached the end of the list of defined patterns,
                 * define a new one */
                if (plotab_cur->p_pattern == (char *) NULL) {
                    plotab_cur->p_pattern = copy(pattern);
                    plotab_cur->f_pattern_is_alloc = TRUE;
                }

                /* Assign the name for the pattern. Freeing the old
                 * name, if present, is complicated by the fact that
                 * the same name allocation is used for all matching
                 * patterns. */
                else {
                    char * const p_name_old = plotab_cur->p_name;
                    if (p_name_old != (char *) NULL &&
                            plotab_cur->f_name_is_alloc) {
                        /* Alloc exists. Must free if this is the only use.
                         * Find usage count to make this decision. */
                        int j;
                        int n_use = 0;
                        for (j = 0; j < NUMPLOTTYPES; j++) {
                            const char * const p_name_cur = plotabs[j].p_name;

                            /* Test for end of list */
                            if (p_name_cur == (char *) NULL) { /* end */
                                break;
                            }

                            /* More entries, so check for the allocation of
                             * the old name */
                            if (p_name_cur == p_name_old) { /* match */
                                n_use++;
                            }
                        } /* end of loop over plot types */

                        /* Now can free if the usage count is exactly one,
                         * that use being the current one */
                        if (n_use == 1) {
                            txfree((void *) p_name_old);
                        }
                    } /* end of case that there was an existing name here */
                } /* end of case that there was an old pattern */

                /* Assign the (new) name */
                plotab_cur->p_name = name;
                plotab_cur->f_name_is_alloc = TRUE;
            } /* end of block for assigning pattern and abbrev */
            f_name_used = TRUE; /* flag that the allocated name was used */
        } /* end of loop over patterns */
        break;
    }
    default:
        fprintf(cp_err, "Error: invalid subcommand '%c'. "
                "Expecting 'p' or 'v'.\n", subcmd_char);
        break;
    } /* end of switch over subcommands */
} /* end of function com_dftype */



/* Return the abbreviation associated with a number. */

char *
ft_typabbrev(int typenum)
{
    if ((typenum < NUMTYPES) && (typenum >= 0)) {
        char* tp = types[typenum].t_abbrev;
        if (tp && cieq("rad", tp) && cx_degrees)
            return ("Degree");
        else
            return tp;
    }
    else
        return (NULL);
}


/* Return the typename associated with a number. */

char *
ft_typenames(int typenum)
{
    if ((typenum < NUMTYPES) && (typenum >= 0))
        return (types[typenum].t_name);
    else
        return (NULL);
}


static int
ft_typenum_x(char *type)
{
    int i;

    for (i = 0; i < NUMTYPES && types[i].t_name; i++)
        if (eq(type, types[i].t_name))
            return i;

    return -1;
}


/* Return the type number associated with the name. */

int
ft_typnum(char *name)
{
    int i;

    if (eq(name, "none"))
        name = "notype";

    for (i = 0; (i < NUMTYPES) && types[i].t_name; i++)
        if (cieq(name, types[i].t_name))
            return (i);

    return (SV_NOTYPE);
}


/* For plots... */

char *
ft_plotabbrev(char *string)
{
    char buf[128];
    int i;

    if (!string)
        return (NULL);

    strncpy(buf, string, sizeof(buf));
    buf[sizeof(buf) - 1] = '\0';
    strtolower(buf);

    for (i = 0; i < NUMPLOTTYPES && plotabs[i].p_name; i++)
        if (substring(plotabs[i].p_pattern, buf))
            return (plotabs[i].p_name);

    return (NULL);
}


/* Change the type of a vector. */

void
com_stype(wordlist *wl)
{
    char *type = wl->wl_word;
    int typenum = ft_typenum_x(type);

    if (typenum < 0) {
        fprintf(cp_err, "Error: no such vector type as '%s'\n", type);
            fprintf(cp_err, "    Command 'settype %s %s ...' is ignored\n\n", type, wl->wl_next->wl_word);
        return;
    }

    for (wl = wl->wl_next; wl; wl = wl->wl_next) {
        const char* vecname = wl->wl_word;
        if (*vecname == '@' && ft_curckt && !ft_curckt->ci_runonce) {
            fprintf(cp_err, "Warning: Vector %s is available only after the simulation has been run!\n", vecname);
            fprintf(cp_err, "    Command 'settype %s %s' is ignored\n\n", type, vecname);
            continue;
        }
        struct dvec *v = vec_get(vecname);
        if (!v) {
            fprintf(cp_err, "Warning: no such vector %s.\n", vecname);
            fprintf(cp_err, "    Command 'settype %s %s' is ignored\n\n", type, vecname);
        }
        else
            for (; v; v = v->v_link2)
                if (v->v_flags & VF_PERMANENT)
                    v->v_type = typenum;
    }
}
