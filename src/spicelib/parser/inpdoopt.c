/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Thomas L. Quarles
Modified: 2000 AlansFixes
**********/

    /* INPdoOpts(ckt,option card)
     *  parse the options off of the given option card and add them to
     *  the given circuit
     */

#include "ngspice/ngspice.h"
#include <stdio.h>
#include "ngspice/inpdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/cpdefs.h"
#include "ngspice/fteext.h"


/* Track unknown option names that have already been warned about so that
 * large PDK libraries (which repeat the same .option lines many times) do
 * not flood stderr with identical messages. */
static struct {
    char *names[64];
    int   count;
} warned_opts;

static bool
unknown_opt_first_seen(const char *name)
{
    int i;
    for (i = 0; i < warned_opts.count; i++)
        if (strcmp(warned_opts.names[i], name) == 0)
            return false;
    if (warned_opts.count < 64)
        warned_opts.names[warned_opts.count++] = copy(name);
    return true;
}

/* Advance line past an optional '=value' token so that the value string
 * is not mistakenly parsed as the next option keyword. */
static void
skip_opt_value(char **line)
{
    while (isspace((unsigned char)**line)) (*line)++;
    if (**line != '=')
        return;
    (*line)++;
    while (isspace((unsigned char)**line)) (*line)++;
    char *endp;
    strtod(*line, &endp);
    if (endp > *line) {
        *line = endp;
    } else {
        while (**line && !isspace((unsigned char)**line)) (*line)++;
    }
}


void
INPdoOpts(
    CKTcircuit *ckt,
    JOB *anal,
    struct card *optCard,
    INPtables *tab)
{
    char *line;
    char *token;
    char *errmsg;  /* used for unimplemented-option and can't-set-option warnings */
    IFvalue *val;
    int error;
    int which;

    which = ft_find_analysis("options");

    if(which == -1) {
        optCard->error = INPerrCat(optCard->error,INPmkTemp(
                                       "error:  analysis options table not found\n"));
        return;
    }

    line = optCard->line;

    INPgetTok(&line,&token,1);    /* throw away '.option' */

    while (*line) {

        IFparm *if_parm;

        INPgetTok(&line,&token,1);

        if_parm = ft_find_analysis_parm(which, token);

        if(if_parm && !(if_parm->dataType & IF_UNIMP_MASK)) {
            errmsg = tprintf(" Warning: %s not yet implemented - ignored \n", token);
            optCard->error = INPerrCat(optCard->error,errmsg);
            val = INPgetValue(ckt,&line, if_parm->dataType, tab);
            continue;
        }

        if(if_parm && (if_parm->dataType & IF_SET)) {
            val = INPgetValue(ckt,&line, if_parm->dataType&IF_VARTYPES, tab);
            error = ft_sim->setAnalysisParm (ckt, anal, if_parm->id, val, NULL);
            if(error) {
                errmsg = tprintf("Warning:  can't set option %s\n", token);
                optCard->error = INPerrCat(optCard->error, errmsg);
            }
            continue;
        }

        /* geoshrink: obsolete.  The MOSFET W/L shrink is now applied
         * generically from the subcircuit's `scale` parameter during
         * subckt expansion (see subckt.c), matching HSPICE.  Decks that
         * still carry `.option geoshrink=<val>` are accepted but the
         * value is ignored — silently consume it so it isn't reparsed. */
        if (strcmp(token, "geoshrink") == 0) {
            while (isspace((unsigned char)*line)) line++;
            char *endp;
            (void) strtod(line, &endp);
            if (endp > line)
                line = endp;
            continue;
        }

        /* Skip '=value' so the value string is not re-parsed as an option. */
        skip_opt_value(&line);

        /* Warn only for non-numeric tokens, and only the first time each
         * unknown option name is encountered.  Do NOT set optCard->error:
         * unknown options are silently ignored, not fatal card errors.
         *
         * INPgetTok(gobble=1) consumes the '=' before returning, so
         * skip_opt_value above cannot see it and may leave a string value
         * token as the next token in line.  Numparam substitution replaces
         * string literals with 'numparm__________XXXXXXXX' placeholders;
         * these are artifacts, not real option names — discard silently. */
        if (strncmp(token, "numparm__________", 17) == 0)
            continue;
        char* ctoken = token;
        while (*ctoken && strchr("0123456789.e+-", *ctoken))
            ctoken++;
        if (*ctoken && unknown_opt_first_seen(token)) {
            /* Recognised-but-unimplemented HSPICE options get a softer
             * "Warning:" prefix so user scripts that intentionally rely
             * on these features can distinguish them from real typos.
             * Add to this list when a new known-pending HSPICE option
             * appears in foundry decks. */
            static const char *const known_hspice_pending[] = {
                "tmiflag", "modmonte", "tmipath", "etmiusrinput",
                NULL
            };
            const char *const *p = known_hspice_pending;
            bool known_pending = false;
            for (; *p; p++) {
                if (strcasecmp(token, *p) == 0) { known_pending = true; break; }
            }
            if (known_pending)
                fprintf(stderr, "Warning: unknown option %s - ignored\n", token);
            else
                fprintf(stderr, "Error: unknown option %s - ignored\n", token);
        }
    }
}
