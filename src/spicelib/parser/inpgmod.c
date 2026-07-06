/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Copyright 2000 The ngspice team
3-Clause BSD license
(see COPYING or https://opensource.org/licenses/BSD-3-Clause)
Author: 1985 Thomas L. Quarles, 1991 David A. Gates
Modified: 2001 Paolo Nenzi (Cider Integration)
**********/

#include "ngspice/ngspice.h"
#include "ngspice/inpdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/cpstd.h"
#include "ngspice/fteext.h"
#include "ngspice/compatmode.h"
#include "ngspice/devdefs.h"
#include "ngspice/osdi_defer.h"
#include "inpxx.h"
#include <errno.h>
#include <stdio.h>
#include <string.h>

#ifdef CIDER

#include "ngspice/numcards.h"
#include "ngspice/carddefs.h"
#include "ngspice/numgen.h"
#include "ngspice/suffix.h"

#define E_MISSING    -1
#define E_AMBIGUOUS  -2

extern IFcardInfo *INPcardTab[];
extern int INPnumCards;

static int INPparseNumMod(CKTcircuit *ckt, INPmodel *model, INPtables *tab, char **errMessage);
static int INPfindCard(char *name, IFcardInfo *table[], int numCards);
static int INPfindParm(char *name, IFparm *table, int numParms);

#endif

extern INPmodel *modtab;
extern NGHASHPTR modtabhash;


static IFparm *
find_model_parameter(const char *name, IFdevice *device)
{
    IFparm *p = device->modelParms;
    IFparm *p_end = p + *(device->numModelParms);

    for (; p < p_end; p++)
        if (strcmp(name, p->keyword) == 0)
            return p;

    return NULL;
}


static IFparm *
find_instance_parameter(const char *name, IFdevice *device)
{
    IFparm *p = device->instanceParms;
    IFparm *p_end = p + *(device->numInstanceParms);

    for (; p < p_end; p++)
        if (strcmp(name, p->keyword) == 0)
            return p;

    return NULL;
}


/*
 * code moved from INPgetMod
 */
static int
create_model(CKTcircuit *ckt, INPmodel *modtmp, INPtables *tab)
{
    char    *err = NULL, *line, *parm, *endptr;
    int     error;

    /* not already defined, so create & give parameters */
    error = ft_sim->newModel(ckt, modtmp->INPmodType, &(modtmp->INPmodfast), modtmp->INPmodName);
    if (error)
        return error;

#ifdef CIDER
    /* Handle Numerical Models Differently */
    if (modtmp->INPmodType == INPtypelook("NUMD") ||
        modtmp->INPmodType == INPtypelook("NBJT") ||
        modtmp->INPmodType == INPtypelook("NUMD2") ||
        modtmp->INPmodType == INPtypelook("NBJT2") ||
        modtmp->INPmodType == INPtypelook("NUMOS"))
    {
        error = INPparseNumMod(ckt, modtmp, tab, &err);
        if (error)
            return error;
        modtmp->INPmodLine->error = err;
        return 0;
    }
#endif

    IFdevice *device = ft_sim->devices[modtmp->INPmodType];

    /* parameter isolation, identification, binding */

    line = modtmp->INPmodLine->line;

#ifdef TRACE
    printf("In INPgetMod, inserting new model into table.  line = %s ...\n", line);
#endif

    INPgetTok(&line, &parm, 1);        /* throw away '.model' */
    tfree(parm);
    INPgetNetTok(&line, &parm, 1);        /* throw away 'modname' */
    tfree(parm);

    bool is_osdi = false;
#ifdef OSDI
    /* osdi models don't accept their device type as an argument */
    if (device->registry_entry) {
        /* Skip the OSDI device type (the module name).  We must stop at '('
         * as well as whitespace: a no-space card like
         *   .model m slew_probe(max_pos=500, max_neg=-500)
         * is standard SPICE syntax, but INPgetNetTok does NOT treat '(' as a
         * token terminator, so it would grab "slew_probe(max_pos" as the
         * "type" and silently swallow the FIRST model parameter.  Extract just
         * the type name, stopping at the first '(', '=', ',' or whitespace, and
         * leave `line` pointing at the parameter list. */
        while (*line == ' ' || *line == '\t')
            line++;
        char *type_start = line;
        while (*line && *line != ' ' && *line != '\t' && *line != '(' &&
               *line != '=' && *line != ',')
            line++;
        parm = copy_substring(type_start, line);
        bool is_pmos = (strcasecmp(parm, "pmos") == 0 || strcasecmp(parm, "psoi") == 0);
        tfree(parm);
        is_osdi = true;
        /* Commercial simulators handle nmos/pmos polarity implicitly.  For OSDI
         * models with a 'type' parameter (e.g. BSIM-BULK TYPE=1/-1), inject -1
         * before card params are parsed so the PDK can still override explicitly. */
        if (is_pmos) {
            IFparm *type_parm = find_model_parameter("type", device);
            if (type_parm && (type_parm->dataType & IF_INTEGER)) {
                IFvalue type_val;
                type_val.iValue = -1;
                ft_sim->setModelParm(ckt, modtmp->INPmodfast, type_parm->id, &type_val, NULL);
            }
        }
    }
#endif

    while (*line) {
        INPgetTok(&line, &parm, 1);
        if (!*parm) {
            FREE(parm);
            continue;
        }

        IFparm *p = find_model_parameter(parm, device);

        if (p) {
#ifdef OSDI
            if (is_osdi && (p->dataType & IF_VECTOR)){
                // we need to get rid if the leading [ in order to make sure
                // that INPgetValue can parse the value properly
                // This is because, unlike other SPICEDev, OSDI models receive
                // array params in the syntax (param_name=[...])
                ++line;
            }
#endif
            IFvalue *val = INPgetValue(ckt, &line, p->dataType, tab);
            error = ft_sim->setModelParm(ckt, modtmp->INPmodfast, p->id, val, NULL);
            if (error)
                return error;
        } else if ((strcmp(parm, "level") == 0) || (strcmp(parm, "m") == 0)) {
            /* no instance parameter default for level and multiplier */
            /* just grab the number and throw away */
            /* since we already have that info from pass1 */
            INPgetValue(ckt, &line, IF_REAL, tab);
        
        } else {

            p = find_instance_parameter(parm, device);

            if (p) {
                char *value;

                INPgetTok(&line, &value, 1);
                if (p->dataType & IF_SET) {
                    modtmp->INPmodfast->defaults =
                        wl_cons(copy(parm),
                                wl_cons(value, modtmp->INPmodfast->defaults));
                } else {
                    fprintf(stderr,
                            "Ignoring attempt to set a default "
                            "for read-only instance parameter %s in:\n  %s\n",
                            p->keyword, modtmp->INPmodLine->line);
                }
            } else {

                double dval;

                /* want only the parameter names in output - not the values */
                errno = 0;    /* To distinguish success/failure after call */
                dval = strtod(parm, &endptr);
                /* Check for various possible errors */
                if ((errno == ERANGE && dval == HUGE_VAL) || errno != 0) {
                    perror("strtod");
                    controlled_exit(EXIT_FAILURE);
                }
                /* For OSDI models, PDK cards converted from HSPICE often contain
                 * simulator-specific parameters (tmemod, psatbmod, b0, etc.) that
                 * the public OSDI model binary does not implement.  Silently skip
                 * them rather than flooding the output with "Model issue" lines. */
                if (endptr == parm && !is_osdi)
                    err = INPerrCat(err,
                                    tprintf("unrecognized parameter (%s) - ignored",
                                            parm));
            }
        }
        FREE(parm);
    }

    modtmp->INPmodLine->error = err;
    return 0;
}


static bool
parse_line(char *line, char *tokens[], int num_tokens, double values[], bool found[])
{
    int get_index = -1;
    int i;

    for (i = 0; i < num_tokens; i++)
        found[i] = FALSE;

    while (*line) {

        if (get_index != -1) {
            int error;
            values[get_index] = INPevaluate(&line, &error, 1);
            found[get_index] = TRUE;
            get_index = -1;
            continue;
        }

        char *token = NULL;
        INPgetNetTok(&line, &token, 1);

        for (i = 0; i < num_tokens; i++)
            if (strcmp(tokens[i], token) == 0)
                get_index = i;

        txfree(token);
    }

    for (i = 0; i < num_tokens; i++)
        if (!found[i])
            return FALSE;

    return TRUE;
}


static bool
is_equal(double result, double expectedResult)
{
    return fabs(result - expectedResult) < 1e-9;
}


static bool
in_range(double value, double min, double max)
{
    /* the standard binning rule is: min <= value < max */
    return is_equal(value, min)  || is_equal(value, max) || (min < value && value < max);
}


char *
INPgetModBin(CKTcircuit *ckt, char *name, INPmodel **model, INPtables *tab, char *line)
{
    INPmodel    *modtmp;
    double       l, w, lmin, lmax, wmin, wmax;
    double       parse_values[4];
    bool         parse_found[4];
    static char *instance_tokens[] = { "l", "w", "nf", "wnflag" };
    static char *model_tokens[]    = { "lmin", "lmax", "wmin", "wmax" };
    double       scale;
    int          wnflag;

    if (!cp_getvar("scale", CP_REAL, &scale, 0))
        scale = 1;

    if (!cp_getvar("wnflag", CP_NUM, &wnflag, 0)) {
        if (newcompat.spe || newcompat.hs)
            wnflag = 1;
        else
            wnflag = 0;
    }

    *model = NULL;

    /* Read L (required) and W (optional).  FinFET PDKs (Samsung 14LPU
     * et al.) have no W on the instance line — only `l=` and
     * `nfin=` — and their `.model` cards bin on `lmin/lmax/nfinmin/
     * nfinmax`, not on W.  Previously this function bailed out as
     * soon as W was missing, which prevented bin selection for any
     * FinFET model.  Now: require only L, and if W is absent set it
     * to 0 (any model with a real `wmin/wmax` range still gets a
     * sensible has_w_range==true check and rejects this instance,
     * which matches HSPICE's behaviour of "instance W=0 ≠ any
     * wmin/wmax range starting above 0"). */
    if (!parse_line(line, instance_tokens, 1, parse_values, parse_found))
        return NULL;  /* No `l=` either — really not a MOSFET-shape line. */

    /* Now try to read W; if absent, w stays at parse_values[1]=0. */
    parse_values[1] = 0.;
    parse_line(line, instance_tokens, 2, parse_values, parse_found);

    /* This is for reading nf. If nf is not available, set to 1 if in HSPICE or Spectre compatibility mode */
    if (!parse_line(line, instance_tokens, 3, parse_values, parse_found)) {
        parse_values[2] = 1.; /* divisor */
    }
    /* This is for reading wnflag from instance. If it is not available, no change.
       If instance wnflag == 0, set divisor to 1, else use instance nf */
    else if (parse_line(line, instance_tokens, 4, parse_values, parse_found)) {
        /* wnflag from instance overrules: no use of nf */
        if (parse_values[3] == 0) {
            parse_values[2] = 1.; /* divisor */
        }
    }
    /* We do have nf, but no wnflag on the instance. Now it depends on the default
       wnflag or on the .options wnflag */
    else {
        if (wnflag == 0)
            parse_values[2] = 1.; /* divisor */
    }


    l = parse_values[0] * scale;
    w = parse_values[1] / parse_values[2] * scale;

    /* OSDI bin selection: compare per-finger W (w / nf, already applied
     * above) against the bin's wmin/wmax.  Do NOT divide by m — `m` is a
     * parallel-instance multiplier (the whole device is replicated m
     * times), so the per-finger geometry is unchanged by it.  TSMC and
     * GlobalFoundries BSIM-BULK decks author their wmin/wmax tables
     * against the HSPICE/Spectre convention of per-finger W only, so an
     * additional /m here would push the effective W below every bin's
     * lower bound for any moderately-large multi-instance device. */

    for (modtmp = modtab; modtmp; modtmp = modtmp->INPnextModel) {

        if (model_name_match(name, modtmp->INPmodName) < 2)
            continue;

        /* if illegal device type, skip this candidate rather than aborting */
        if (modtmp->INPmodType < 0)
            continue;

        /* skip if not binnable: accept OSDI models (detected via registry_entry)
         * as well as the classic BSIM/HiSIM families that have always been supported */
        IFdevice *dev = ft_sim->devices[modtmp->INPmodType];
        bool is_osdi = (dev->registry_entry != NULL);
        if (!is_osdi &&
            modtmp->INPmodType != INPtypelook("BSIM3") &&
            modtmp->INPmodType != INPtypelook("BSIM3v32") &&
            modtmp->INPmodType != INPtypelook("BSIM3v0") &&
            modtmp->INPmodType != INPtypelook("BSIM3v1") &&
            modtmp->INPmodType != INPtypelook("BSIM4") &&
            modtmp->INPmodType != INPtypelook("BSIM4v5") &&
            modtmp->INPmodType != INPtypelook("BSIM4v6") &&
            modtmp->INPmodType != INPtypelook("BSIM4v7") &&
            modtmp->INPmodType != INPtypelook("HiSIM2") &&
            modtmp->INPmodType != INPtypelook("HiSIMHV1") &&
            modtmp->INPmodType != INPtypelook("HiSIMHV2"))
        {
            continue;
        }

        /* parse lmin/lmax from the model line; wmin/wmax are optional.
         * parse_line returns FALSE if any requested token is absent, so call
         * it unconditionally and inspect parse_found[] individually. */
        parse_line(modtmp->INPmodLine->line, model_tokens, 4, parse_values, parse_found);

        if (!parse_found[0] || !parse_found[1])
            continue;   /* lmin/lmax are required for bin selection */

        lmin = parse_values[0]; lmax = parse_values[1];

        bool has_w_range = parse_found[2] && parse_found[3];
        if (has_w_range) {
            wmin = parse_values[2]; wmax = parse_values[3];
        }

        /* Compare the per-finger W (already w/nf above) and L against
         * the bin's wmin/wmax and lmin/lmax.  Foundry .lib files (TSMC,
         * GlobalFoundries, Samsung) author bin boundaries in POST-SHRINK
         * (EFFECTIVE) dimensions: e.g. TSMC 22nm ULP nch.1 wmax=2.5651µm
         * = drawn 3µm × shrink 0.855.  The shrink is applied upstream in
         * subckt expansion (the subcircuit's `scale` parameter multiplies
         * the device W/L) or by the model's own dimension expressions, so
         * the W/L parsed here are already EFFECTIVE — no shrink factor is
         * applied at this point.
         *
         * `m` is intentionally NOT in the calculation — it is a
         * parallel-instance multiplier (the whole device is replicated
         * m times), so per-finger geometry is unchanged by it. */
        double w_cmp = w;
        double l_cmp = l;

        if (in_range(l_cmp, lmin, lmax) && (!has_w_range || in_range(w_cmp, wmin, wmax))) {
            /* create unless model is already defined */
            if (!modtmp->INPmodfast) {
                int error = create_model(ckt, modtmp, tab);
                if (error)
                    return NULL;
            }

            *model = modtmp;
            /* Stash the bin's (lmin, lmax) for OSDIsetup's deferred-eval
             * pre-pass, which needs a representative L within the bin's
             * range to make Samsung-PDK expressions like
             * `(l==14n)*X + (l==16n)*Y` evaluate to non-zero. */
            if (is_osdi)
                osdi_defer_record_bin_range(modtmp->INPmodName, lmin, lmax);
            return NULL;
        }
    }

    /* Second pass for OSDI models: if no exact bin match was found (device W
     * is outside all bin width ranges), pick the l-matching bin whose wmin/wmax
     * is closest to the device width.  This mirrors HSPICE/Spectre behaviour. */
    {
        double    best_w_dist = -1.0;   /* negative = "no candidate yet" */
        INPmodel *best_mod    = NULL;

        for (modtmp = modtab; modtmp; modtmp = modtmp->INPnextModel) {

            if (model_name_match(name, modtmp->INPmodName) < 2)
                continue;
            if (modtmp->INPmodType < 0)
                continue;

            IFdevice *dev = ft_sim->devices[modtmp->INPmodType];
            if (dev->registry_entry == NULL)
                continue;   /* nearest-bin fallback only for OSDI models */

            parse_line(modtmp->INPmodLine->line, model_tokens, 4, parse_values, parse_found);
            if (!parse_found[0] || !parse_found[1])
                continue;
            lmin = parse_values[0]; lmax = parse_values[1];
            /* Same convention as the primary pass: the W/L parsed here
             * are already EFFECTIVE (shrink applied upstream). */
            if (!in_range(l, lmin, lmax))
                continue;

            double w_c = w;
            double w_dist;
            if (!parse_found[2] || !parse_found[3]) {
                w_dist = 0.0;
            } else {
                wmin = parse_values[2]; wmax = parse_values[3];
                if (w_c < wmin)       w_dist = wmin - w_c;
                else if (w_c > wmax)  w_dist = w_c - wmax;
                else                  w_dist = 0.0;
            }

            if (best_w_dist < 0.0 || w_dist < best_w_dist) {
                best_w_dist = w_dist;
                best_mod    = modtmp;
            }
        }

        if (best_mod) {
            fprintf(stderr,
                    "Warning: OSDI model '%s': device W=%.3g is outside all bin"
                    " width ranges; using nearest bin\n",
                    name, w);
            if (!best_mod->INPmodfast) {
                int error = create_model(ckt, best_mod, tab);
                if (error)
                    return NULL;
            }
            *model = best_mod;
        }
    }

    return NULL;
}


char *
INPgetMod(CKTcircuit *ckt, char *name, INPmodel **model, INPtables *tab)
{
    INPmodel *modtmp;

#ifdef TRACE
    printf("In INPgetMod, examining model %s ...\n", name);
#endif

    if (modtabhash) {
        modtmp = nghash_find(modtabhash, name);
        if (modtmp) {
            /* found the model in question - now instantiate if necessary */
            /* and return an appropriate pointer to it */

    /* if illegal device type */
            if (modtmp->INPmodType < 0) {
#ifdef TRACE
                printf("In INPgetMod, illegal device type for model %s ...\n", name);
#endif
                * model = NULL;
                return tprintf("Unknown device type for model %s\n", name);
            }

            /* create unless model is already defined */
            if (!modtmp->INPmodfast) {
                int error = create_model(ckt, modtmp, tab);
                if (error) {
                    *model = NULL;
                    return INPerror(error);
                }
            }

            *model = modtmp;
            return NULL;
        }
    }
#if (0)
    for (modtmp = modtab; modtmp; modtmp = modtmp->INPnextModel) {

#ifdef TRACE
        printf("In INPgetMod, comparing %s against stored model %s ...\n", name, modtmp->INPmodName);
#endif

        if (strcmp(modtmp->INPmodName, name) == 0) {
            /* found the model in question - now instantiate if necessary */
            /* and return an appropriate pointer to it */

            /* if illegal device type */
            if (modtmp->INPmodType < 0) {
#ifdef TRACE
                printf("In INPgetMod, illegal device type for model %s ...\n", name);
#endif
                *model = NULL;
                return tprintf("Unknown device type for model %s\n", name);
            }

            /* create unless model is already defined */
            if (!modtmp->INPmodfast) {
                int error = create_model(ckt, modtmp, tab);
                if (error) {
                    *model = NULL;
                    return INPerror(error);
                }
            }

            *model = modtmp;
            return NULL;
        }
    }
#endif
#ifdef TRACE
    printf("In INPgetMod, didn't find model for %s, using default ...\n", name);
#endif

    *model = NULL;
    return tprintf("Unable to find definition of model %s\n", name);
}


#ifdef CIDER
/*
 * Parse a numerical model by running through the list of original
 * input cards which make up the model
 * Given:
 * 1. First card looks like: .model modname modtype <level=val>
 * 2. Other cards look like: +<whitespace>? where ? tells us what
 * to do with the next card:
 *    '#$*' = comment card
 *    '+'   = continue previous card
 *    other = new card
 */
static int
INPparseNumMod(CKTcircuit *ckt, INPmodel *model, INPtables *tab, char **errMessage)
{
    struct card *txtCard;    /* Text description of a card */
    GENcard *tmpCard = NULL; /* Processed description of a card */
    IFcardInfo *info = NULL; /* Info about the type of card located */
    char *cardName = NULL;   /* name of a card */
    int cardNum = 0;         /* number of this card in the overall line */
    char *err = NULL;        /* Strings for error messages */
    int error;

    /* Chase down to the top of the list of actual cards */
    txtCard = model->INPmodLine->actualLine;

    /* Skip the first card if it exists since there's nothing interesting */
    /* txtCard will be empty if the numerical model is empty */
    if (txtCard)
        txtCard = txtCard->nextcard;

    /* Now parse each remaining card */
    for (; txtCard; txtCard = txtCard->nextcard) {
        char *line = txtCard->line;
        cardNum++;

        /* Skip the initial '+' and any whitespace. */
        line++;
        while (*line == ' ' || *line == '\t')
            line++;

        switch (*line) {
        case '*':
        case '$':
        case '#':
        case '\0':
        case '\n':
            /* comment or empty cards */
            info = NULL;
            continue;
        case '+':
            /* continuation card */
            if (!info) {
                err = INPerrCat(err,
                                tprintf("Error on card %d : illegal continuation \'+\' - ignored",
                                        cardNum));
                continue;
            }
            /* Skip leading '+'s */
            while (*line == '+')
                line++;
            break;
        default:
            info = NULL;
            break;
        }

        if (!info) {
            /* new command card */
            if (cardName)       /* get rid of old card name */
                FREE(cardName);
            INPgetTok(&line, &cardName, 1);        /* get new card name */
            if (*cardName) {                 /* Found a name? */
                int lastType = INPfindCard(cardName, INPcardTab, INPnumCards);
                if (lastType >= 0) {
                    /* Add card structure to model */
                    info = INPcardTab[lastType];
                    error = info->newCard(&tmpCard, model->INPmodfast);
                    if (error) {
                        FREE(cardName);
                        return error;
                    }
                    /* Handle parameter-less cards */
                } else if (cinprefix(cardName, "title", 3)) {
                    /* Do nothing */
                } else if (cinprefix(cardName, "comment", 3)) {
                    /* Do nothing */
                } else if (cinprefix(cardName, "end", 3)) {
                    /* Terminate parsing */
                    *errMessage = err;
                    FREE(cardName);
                    return 0;
                } else {
                    /* Error */
                    err = INPerrCat(err,
                                    tprintf("Error on card %d : unrecognized name (%s) - ignored",
                                            cardNum, cardName));
                }
                FREE(cardName);
            }
        }

        if (!info)
            continue;

        /* parse the rest of this line */
        while (*line) {

            int invert = FALSE;
            /* Strip leading carat from booleans */
            if (*line == '^') {
                invert = TRUE;
                line++;
            }

            char *parm;                /* name of a parameter */
            INPgetTok(&line, &parm, 1);
            if (!*parm) {
                FREE(parm);
                break;
            }

            int idx = INPfindParm(parm, info->cardParms, info->numParms);
            if (idx == E_MISSING) {
                /* parm not found */
                err = INPerrCat(err,
                                tprintf("Error on card %d : unrecognized parameter (%s) - ignored",
                                        cardNum, parm));
            } else if (idx == E_AMBIGUOUS) {
                /* parm ambiguous */
                err = INPerrCat(err,
                                tprintf("Error on card %d : ambiguous parameter (%s) - ignored",
                                        cardNum, parm));
            } else {
                IFvalue *value = INPgetValue(ckt, &line, info->cardParms[idx].dataType, tab);

                /* invert if this is a boolean entry */
                if (invert) {
                    if ((info->cardParms[idx].dataType & IF_VARTYPES) == IF_FLAG)
                        value->iValue = 0;
                    else
                        err = INPerrCat(err,
                                        tprintf("Error on card %d : non-boolean parameter (%s) - \'^\' ignored",
                                                cardNum, parm));
                }

                error = info->setCardParm(info->cardParms[idx].id, value, tmpCard);
                if (info->cardParms[idx].dataType & IF_STRING) {
                    FREE(value->sValue);
                } else if (info->cardParms[idx].dataType & IF_REALVEC) {
                    FREE(value->v.vec.rVec);
                } else if (info->cardParms[idx].dataType & IF_INTVEC) {
                    FREE(value->v.vec.iVec);
                }
                if (error)
                    return error;
            }
            FREE(parm);
        }
    }

    *errMessage = err;
    return 0;
}


/*
 * Locate the best match to a card name in an IFcardInfo table
 */
static int
INPfindCard(char *name, IFcardInfo *table[], int numCards)
{
    int length = (int) strlen(name);
    int best = E_MISSING;
    int bestMatch = 0;

    int test;

    /* compare all the names in the card table to this name */
    for (test = 0; test < numCards; test++) {
        int match = cimatch(name, table[test]->name);
        if ((match > 0) && (match == bestMatch)) {
            best = E_AMBIGUOUS;
        } else if ((match > bestMatch) && (match == length)) {
            best = test;
            bestMatch = match;
        }
    }

    return best;
}


/*
 * Locate the best match to a parameter name in an IFparm table
 */
static int
INPfindParm(char *name, IFparm *table, int numParms)
{
    int length = (int) strlen(name);
    int best = E_MISSING;
    int bestMatch = 0;
    int bestId = -1;

    int test;

    /* compare all the names in the parameter table to this name */
    for (test = 0; test < numParms; test++) {
        int match = cimatch(name, table[test].keyword);
        if ((match == length) && (match == (int) strlen(table[test].keyword))) {
            /* exact match */
            return test;
        }
        int id = table[test].id;
        if ((match > 0) && (match == bestMatch) && (id != bestId)) {
            best = E_AMBIGUOUS;
        } else if ((match > bestMatch) && (match == length)) {
            bestMatch = match;
            bestId = id;
            best = test;
        }
    }

    return best;
}

#endif /* CIDER */
