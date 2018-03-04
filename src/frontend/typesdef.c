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


#define NUMTYPES 128+4      /* If this is too little we can use a list. */
#define NUMPLOTTYPES 512    /* Since there may be more than 1 pat/type. */

struct type {
    char *t_name;
    char *t_abbrev;
};

/* The stuff for plot names. */

struct plotab {
    char *p_name;
    char *p_pattern;
};

/* note:  This should correspond to SV_ defined in sim.h */
static struct type types[NUMTYPES] = {
    { "notype", NULL } ,
    { "time", "s" } ,
    { "frequency", "Hz" } ,
    { "voltage", "V" } ,
    { "current", "A" } ,
    { "voltage-density", "V/sqrt(Hz)" } ,
    { "current-density", "A/sqrt(Hz)" } ,
    { "voltage^2-density", "(V^2)/Hz" } ,
    { "current^2-density", "(A^2)/Hz" } ,
    { "voltage^2", "(V^2)" } ,
    { "current^2", "(A^2)" } ,
    { "pole", NULL } ,
    { "zero", NULL } ,
    { "s-param", NULL } ,
    { "temp-sweep", "Celsius" } , /* Added by HT */
    { "res-sweep", "Ohms" } ,     /* Added by HT */
    { "impedance", "Ohms" } ,     /* Added by A.Roldan */
    { "admittance", "Mhos" } ,    /* Added by A.Roldan */
    { "power", "W" } ,            /* Added by A.Roldan */
    { "phase", "Degree" } ,       /* Added by A.Roldan */
    { "decibel", "dB" } ,         /* Added by A.Roldan */
    { "capacitance", "F" } ,
    { "charge", "C" } ,
    { "temperature", "Celsius" } ,
};

/* The stuff for plot names. */

static struct plotab plotabs[NUMPLOTTYPES] = {
    { "tran", "transient" } ,
    { "op", "op" } ,
    { "tf", "function" },
    { "dc", "d.c." } ,
    { "dc", "dc" } ,
    { "dc", "transfer" } ,
    { "ac", "a.c." } ,
    { "ac", "ac" } ,
    { "pz", "pz" } ,
    { "pz", "p.z." } ,
    { "pz", "pole-zero"} ,
    { "disto", "disto" } ,
    { "dist", "dist" } ,
    { "noise", "noise" } ,
    { "sens", "sens" } ,
    { "sens", "sensitivity" } ,
    { "sens2", "sens2" } ,
    { "sp", "s.p." } ,
    { "sp", "sp" } ,
    { "harm", "harm" },
    { "spect", "spect" },
    { "pss", "periodic" },
};


/* A command to define types for vectors and plots.  This will generally
 * be used in the Command: field of the rawfile.
 * The syntax is "deftype v typename abbrev", where abbrev will be used to
 * parse things like abbrev(name) and to label axes with M<abbrev>, instead
 * of numbers. It may be ommitted.
 * Also, the command "deftype p plottype pattern ..." will assign plottype as
 * the name to any plot with one of the patterns in its Name: field.
 */

void
com_dftype(wordlist *wl)
{
    char *name, *abb;
    int i;

    switch (*wl->wl_word) {
    case 'v':
    case 'V':
        wl = wl->wl_next;
        name = wl->wl_word;
        wl = wl->wl_next;
        abb = wl->wl_word;
        for (i = 0; i < NUMTYPES && types[i].t_name; i++)
            if (cieq(types[i].t_name, name))
                break;
        if (i >= NUMTYPES) {
            fprintf(cp_err, "Error: too many types defined\n");
            return;
        }
        if (!types[i].t_name)
            types[i].t_name = copy(name);
        types[i].t_abbrev = copy(abb);
        break;

    case 'p':
    case 'P':
        wl = wl->wl_next;
        name = copy(wl->wl_word);
        wl = wl->wl_next;
        for (; wl; wl = wl->wl_next) {
            char *pattern = wl->wl_word;
            for (i = 0; i < NUMPLOTTYPES && plotabs[i].p_pattern; i++)
                if (cieq(plotabs[i].p_pattern, pattern))
                    break;
            if (i >= NUMPLOTTYPES) {
                fprintf(cp_err, "Error: too many plot abs\n");
                return;
            }
            if (!plotabs[i].p_pattern)
                plotabs[i].p_pattern = copy(pattern);
            plotabs[i].p_name = name;
        }
        break;

    default:
        fprintf(cp_err, "Error: missing 'p' or 'v' argument\n");
        break;
    }
}


/* Return the abbreviation associated with a number. */

char *
ft_typabbrev(int typenum)
{
    if ((typenum < NUMTYPES) && (typenum >= 0))
        return (types[typenum].t_abbrev);
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
        fprintf(cp_err, "Error: no such type as '%s'\n", type);
        return;
    }

    for (wl = wl->wl_next; wl; wl = wl->wl_next) {
        struct dvec *v = vec_get(wl->wl_word);
        if (!v)
            fprintf(cp_err, "Error: no such vector %s.\n", wl->wl_word);
        else
            for (; v; v = v->v_link2)
                if (v->v_flags & VF_PERMANENT)
                    v->v_type = typenum;
    }
}
