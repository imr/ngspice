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
struct type types[NUMTYPES] = {
    { "notype", NULL } ,
    { "time", "s" } ,
    { "frequency", "Hz" } ,
    { "voltage", "V" } ,
    { "current", "A" } ,
    { "onoise-spectrum", "(V or A)^2/Hz" } ,
    { "onoise-integrated", "V or A" } ,
    { "inoise-spectrum", "(V or A)^2/Hz" } ,
    { "inoise-integrated", "V or A" } ,
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

};

/* The stuff for plot names. */

struct plotab plotabs[NUMPLOTTYPES] = {
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
};

int notypes = 19;
int noplotabs = 21;


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
        name = copy(wl->wl_word);
        wl = wl->wl_next;
        abb = copy(wl->wl_word);
        for (i = 0; i < notypes; i++)
            if (cieq(types[i].t_name, name)) {
                types[i].t_abbrev = abb;
                return;
            }
        if (notypes >= NUMTYPES - 1) {
            fprintf(cp_err, "Error: too many types defined\n");
            return;
        }
        types[notypes].t_name = name;
        types[notypes].t_abbrev = abb;
        notypes++;
        break;

    case 'p':
    case 'P':
        wl = wl->wl_next;
        name = copy(wl->wl_word);
        wl = wl->wl_next;
        while (wl) {
            for (i = 0; i < noplotabs; i++)
                if (cieq(plotabs[i].p_pattern, wl->wl_word)) {
                    plotabs[i].p_name = name;
                    wl = wl->wl_next;
                    break;
                }
            if (i != noplotabs)
                continue;
            if (noplotabs >= NUMPLOTTYPES - 1) {
                fprintf(cp_err, "Error: too many plot abs\n");
                return;
            }
            plotabs[noplotabs].p_name = name;
            plotabs[noplotabs].p_pattern = copy(wl->wl_word);
            wl = wl->wl_next;
            noplotabs++;
        }
        break;

    default:
        fprintf(cp_err, "Error: missing 'p' or 'v' argument\n");
        break;
    }

    return;
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

    return (0);
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

    for (i = 0; plotabs[i].p_name; i++)
        if (substring(plotabs[i].p_pattern, buf))
            return (plotabs[i].p_name);

    return (NULL);
}


/* Change the type of a vector. */

void
com_stype(wordlist *wl)
{
    char *type = wl->wl_word;
    int tt;
    struct dvec *v, *vv;
    char *s;

    for (tt = 0; ; tt++)
        if ((s = ft_typenames(tt)) == NULL || eq(type, s))
            break;

    if (!s) {
        fprintf(cp_err, "Error: no such type as '%s'\n", type);
        return;
    }

    for (wl = wl->wl_next; wl; wl = wl->wl_next) {
        v = vec_get(wl->wl_word);
        if (!v)
            fprintf(cp_err, "Error: no such vector %s.\n", wl->wl_word);
        else
            for (vv = v; vv; vv = vv->v_link2)
                if (vv->v_flags & VF_PERMANENT)
                    vv->v_type = tt;
    }

    return;
}
