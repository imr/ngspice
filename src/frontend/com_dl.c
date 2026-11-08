#include "ngspice/ngspice.h" /* for wl */
#include "ngspice/ftedefs.h"
#include "ngspice/devdefs.h" /* solve deps in dev.h*/
#include "../spicelib/devices/dev.h" /* for load library commands */
#include "ngspice/inpdefs.h" /* for osdi_bind */
#include "com_dl.h"


#ifdef XSPICE
void com_codemodel(wordlist *wl)
{
if (wl && wl->wl_word)
#ifdef CM_TRACE
    fprintf(stdout, "Note: loading codemodel %s\n", ww->wl_word);
#endif
    if (load_opus(wl->wl_word)) {
        fprintf(stderr, "Error: Library %s couldn't be loaded!\n", wl->wl_word);
        ft_spiniterror = TRUE;
        ft_codemodelerror = copy(wl->wl_word);
        if (ft_stricterror) /* if set in spinit */
            controlled_exit(EXIT_BAD);
    }
#ifdef CM_TRACE
    else {
        fprintf(stdout, "Codemodel %s is loaded\n", wl->wl_word);
    }
#endif
}
#endif

#ifdef OSDI
void com_osdi(wordlist *wl)
{
    wordlist *ww;
    for (ww = wl; ww; ww = ww->wl_next)
        if (load_osdi(ww->wl_word)) {
            fprintf(cp_err, "Error: Library %s couldn't be loaded!\n", ww->wl_word);
            ft_spiniterror = TRUE;
            ft_osdierror = TRUE;
            if (ft_stricterror)
                controlled_exit(EXIT_BAD);
         }
}


/* osdi_bind                       list the bindings
   osdi_bind mos <level> <module> [if <cpar>] [<par> <nval> <pval>] [ignore <par> ...]
   osdi_bind dio <level> <module> [if <cpar>] [lead <ipar>] [ignore <par> ...]
   osdi_bind mos|dio <level> none [if <cpar>]   remove the binding(s)
   Use as pre_osdi_bind inside a .control section. */
void com_osdi_bind(wordlist *wl)
{
    const char *usage =
        "Usage: osdi_bind mos <level> <module> [if <cpar>] [<par> <nval> <pval>] [ignore <par> ...]\n"
        "       osdi_bind dio <level> <module> [if <cpar>] [lead <ipar>] [ignore <par> ...]\n"
        "       osdi_bind mos|dio <level> none [if <cpar>]\n"
        "       osdi_bind\n";
    const char *kind;
    char *module, *end;
    long level;

    if (!wl) {
        INPosdiBindPrint(cp_out);
        return;
    }

    if (strcasecmp(wl->wl_word, "mos") == 0)
        kind = "mos";
    else if (strcasecmp(wl->wl_word, "dio") == 0)
        kind = "dio";
    else {
        fprintf(cp_err, "Error: osdi_bind: device kind %s is not supported, only mos and dio\n",
                wl->wl_word);
        return;
    }
    bool is_mos = strcmp(kind, "mos") == 0;

    if (!wl->wl_next || !wl->wl_next->wl_next) {
        fprintf(cp_err, "%s", usage);
        return;
    }
    level = strtol(wl->wl_next->wl_word, &end, 10);
    if (*end || level < 1 || level > 99) {
        fprintf(cp_err, "Error: osdi_bind: level %s is not a number from 1 to 99\n",
                wl->wl_next->wl_word);
        return;
    }
    wl = wl->wl_next->wl_next;
    module = wl->wl_word;
    wl = wl->wl_next;

    /* polarity parameter of mos, default: type 1 -1; dio has none */
    const char *cond = NULL, *lead = NULL;
    const char *par = NULL, *nval = NULL, *pval = NULL;
    if (is_mos) {
        par = "type";
        nval = "1";
        pval = "-1";
    }
    wordlist *ignore = NULL;

    while (wl) {
        if (strcasecmp(wl->wl_word, "if") == 0 && wl->wl_next) {
            /* selected if the card has this parameter */
            cond = wl->wl_next->wl_word;
            wl = wl->wl_next->wl_next;
        } else if (strcasecmp(wl->wl_word, "lead") == 0 && wl->wl_next && !is_mos) {
            /* instance parameter of an unlabeled value */
            lead = wl->wl_next->wl_word;
            wl = wl->wl_next->wl_next;
        } else if (strcasecmp(wl->wl_word, "ignore") == 0 && wl->wl_next) {
            ignore = wl_copy(wl->wl_next);
            wl = NULL;
        } else if (is_mos && wl->wl_next && wl->wl_next->wl_next &&
                   strcasecmp(wl->wl_word, "if") != 0 &&
                   strcasecmp(wl->wl_word, "ignore") != 0) {
            par = wl->wl_word;
            nval = wl->wl_next->wl_word;
            pval = wl->wl_next->wl_next->wl_word;
            wl = wl->wl_next->wl_next->wl_next;
        } else {
            fprintf(cp_err, "%s", usage);
            wl_free(ignore);
            return;
        }
    }

    if (strcasecmp(module, "none") == 0) {
        if (lead || ignore || (is_mos && strcmp(par, "type") != 0))
            fprintf(cp_err, "%s", usage);
        else if (!INPosdiBindRemove(kind, (int) level, cond, cond == NULL))
            fprintf(cp_err, "Warning: osdi_bind: %s level %ld%s%s was not bound\n",
                    kind, level, cond ? " if " : "", cond ? cond : "");
        wl_free(ignore);
        return;
    }

    if (INPosdiBindAdd(kind, (int) level, module, cond, lead, par, nval, pval, ignore))
        fprintf(cp_out, "Note: osdi_bind: replaced the binding of %s level %ld%s%s\n",
                kind, level, cond ? " if " : "", cond ? cond : "");
}
#endif




#ifdef DEVLIB
void com_use(wordlist *wl)
{
    wordlist *ww;
    for (ww = wl; ww; ww = ww->wl_next)
        if (load_dev(wl->wl_word))
            fprintf(cp_err, "Error: Library %s couldn't be loaded!\n", ww->wl_word);
}
#endif
