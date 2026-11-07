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
   osdi_bind mos <level> none [if <cpar>]   remove the binding(s)
   Use as pre_osdi_bind inside a .control section. */
void com_osdi_bind(wordlist *wl)
{
    const char *usage =
        "Usage: osdi_bind mos <level> <module> [if <cpar>] [<par> <nval> <pval>] [ignore <par> ...]\n"
        "       osdi_bind mos <level> none [if <cpar>]\n"
        "       osdi_bind\n";
    char *kind, *module, *end;
    long level;

    if (!wl) {
        INPosdiBindPrint(cp_out);
        return;
    }

    kind = wl->wl_word;
    if (strcasecmp(kind, "mos") != 0) {
        fprintf(cp_err, "Error: osdi_bind: device kind %s is not supported, only mos\n", kind);
        return;
    }

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

    /* condition: selected if the card has this parameter */
    const char *cond = NULL;
    if (wl && strcasecmp(wl->wl_word, "if") == 0) {
        if (!wl->wl_next) {
            fprintf(cp_err, "%s", usage);
            return;
        }
        cond = wl->wl_next->wl_word;
        wl = wl->wl_next->wl_next;
    }

    if (strcasecmp(module, "none") == 0) {
        if (wl)
            fprintf(cp_err, "%s", usage);
        else if (!INPosdiBindRemove("mos", (int) level, cond, cond == NULL))
            fprintf(cp_err, "Warning: osdi_bind: mos level %ld%s%s was not bound\n",
                    level, cond ? " if " : "", cond ? cond : "");
        return;
    }

    /* polarity parameter, default: type 1 -1 */
    const char *par = "type", *nval = "1", *pval = "-1";
    if (wl && strcasecmp(wl->wl_word, "ignore") != 0) {
        if (!wl->wl_next || !wl->wl_next->wl_next) {
            fprintf(cp_err, "%s", usage);
            return;
        }
        par = wl->wl_word;
        nval = wl->wl_next->wl_word;
        pval = wl->wl_next->wl_next->wl_word;
        wl = wl->wl_next->wl_next->wl_next;
    }

    wordlist *ignore = NULL;
    if (wl) {
        if (strcasecmp(wl->wl_word, "ignore") != 0 || !wl->wl_next) {
            fprintf(cp_err, "%s", usage);
            return;
        }
        ignore = wl_copy(wl->wl_next);
    }

    if (INPosdiBindAdd("mos", (int) level, module, cond, par, nval, pval, ignore))
        fprintf(cp_out, "Note: osdi_bind: replaced the binding of mos level %ld%s%s\n",
                level, cond ? " if " : "", cond ? cond : "");
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
