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


/* osdi_bind
     list the bindings
   osdi_bind mos|bjt|dio <level> <module> [version=<v>] [has=<par>]
                         [polarity=<par>[:<nval>:<pval>]|none]
                         [alias=<name>:<ipar>,...] [ignore=<par>,<par>,...]
     bind a model level to an OSDI module
   osdi_bind mos|bjt|dio <level> none [version=<v>] [has=<par>]
     remove the binding(s)
   Use as pre_osdi_bind inside a .control section. */

static const char *osdi_bind_usage =
    "Usage: osdi_bind mos|bjt <level> <module> [version=<v>] [has=<par>]\n"
    "                 [polarity=<par>[:<nval>:<pval>]|none] [alias=<name>:<ipar>,...] [ignore=<par>,...]\n"
    "       osdi_bind dio <level> <module> [version=<v>] [has=<par>] [alias=<name>:<ipar>,...] [ignore=<par>,...]\n"
    "       osdi_bind mos|bjt|dio <level> none [version=<v>] [has=<par>]\n"
    "       osdi_bind\n";

/* The words of the command as one string, without blanks around '='
   and after ',', split into words again */
static wordlist *
osdi_bind_words(wordlist *wl)
{
    char *line = wl_flatten(wl);
    char *dst = line, *src;
    for (src = line; *src; src++) {
        if (*src == ' ' || *src == '\t') {
            const char *next = src;
            while (*next == ' ' || *next == '\t')
                next++;
            if (*next == '=' || *next == ',' || *next == '\0' ||
                (dst > line && (dst[-1] == '=' || dst[-1] == ',' || dst[-1] == ' ')))
                continue;
            *dst++ = ' ';
        } else {
            *dst++ = *src;
        }
    }
    *dst = '\0';
    wordlist *words = NULL, *last = NULL;
    char *p = line, *tok;
    while ((tok = gettok(&p)) != NULL)
        wl_append_word(&words, &last, tok);
    tfree(line);
    return words;
}


void com_osdi_bind(wordlist *wl)
{
    if (!wl) {
        INPosdiBindPrint(cp_out);
        return;
    }

    wordlist *words = osdi_bind_words(wl);
    wordlist *w = words;
    const char *kind;
    char *module, *end;
    long level;
    const char *has = NULL, *version = NULL, *polarity = NULL;
    const char *nval = NULL, *pval = NULL;
    wordlist *ignore = NULL, *ignore_last = NULL;
    wordlist *alias = NULL, *alias_last = NULL;

    if (!w) {
        fprintf(cp_err, "%s", osdi_bind_usage);
        return;
    }
    if (strcasecmp(w->wl_word, "mos") == 0)
        kind = "mos";
    else if (strcasecmp(w->wl_word, "bjt") == 0)
        kind = "bjt";
    else if (strcasecmp(w->wl_word, "dio") == 0)
        kind = "dio";
    else {
        fprintf(cp_err, "Error: osdi_bind: device kind %s is not supported, only mos, bjt and dio\n",
                w->wl_word);
        goto done;
    }
    /* mos and bjt have a polarity, default parameter type */
    bool is_mos = strcmp(kind, "dio") != 0;
    if (is_mos)
        polarity = "type";

    if (!w->wl_next || !w->wl_next->wl_next) {
        fprintf(cp_err, "%s", osdi_bind_usage);
        goto done;
    }
    level = strtol(w->wl_next->wl_word, &end, 10);
    if (*end || level < 1 || level > 99) {
        fprintf(cp_err, "Error: osdi_bind: level %s is not a number from 1 to 99\n",
                w->wl_next->wl_word);
        goto done;
    }
    w = w->wl_next->wl_next;
    module = w->wl_word;

    /* options name=value; the command line lexer splits at commas,
       so words without '=' after ignore= or alias= continue its list */
    const char *in_list = NULL;
    for (w = w->wl_next; w; w = w->wl_next) {
        char *opt = w->wl_word;
        char *val = strchr(opt, '=');
        if (!val && in_list) {
            val = opt;
            opt = (char *) in_list;
        } else if (val) {
            *val++ = '\0';
        }
        in_list = NULL;
        if (!val || !*val) {
            fprintf(cp_err, "Error: osdi_bind: %s is not of the form name=value\n", opt);
            fprintf(cp_err, "%s", osdi_bind_usage);
            goto done;
        }
        if (strcasecmp(opt, "has") == 0) {
            has = val;
        } else if (strcasecmp(opt, "version") == 0) {
            /* without quotes */
            if (*val == '"' || *val == '\'')
                val++;
            char *q = strpbrk(val, "\"'");
            if (q)
                *q = '\0';
            version = val;
        } else if (strcasecmp(opt, "alias") == 0) {
            /* <name>:<ipar>,<name>:<ipar>,... */
            char *item = val, *next;
            for (; item; item = next) {
                next = strchr(item, ',');
                if (next)
                    *next++ = '\0';
                if (!*item)
                    continue;
                char *colon = strchr(item, ':');
                if (!colon || colon == item || !colon[1] || strchr(colon + 1, ':')) {
                    fprintf(cp_err, "Error: osdi_bind: alias %s is not <name>:<ipar>\n", item);
                    goto done;
                }
                *colon = '\0';
                wl_append_word(&alias, &alias_last, copy(item));
                wl_append_word(&alias, &alias_last, copy(colon + 1));
            }
            in_list = "alias";
        } else if (strcasecmp(opt, "polarity") == 0 && is_mos) {
            /* <par>, <par>:<nval>:<pval>, or none */
            char *c1 = strchr(val, ':');
            if (c1) {
                char *c2 = strchr(c1 + 1, ':');
                if (!c2 || c2 == c1 + 1 || !c2[1]) {
                    fprintf(cp_err, "Error: osdi_bind: polarity=%s is not <par>:<nval>:<pval>\n", val);
                    goto done;
                }
                *c1 = *c2 = '\0';
                nval = c1 + 1;
                pval = c2 + 1;
            }
            polarity = strcasecmp(val, "none") == 0 ? NULL : val;
        } else if (strcasecmp(opt, "ignore") == 0) {
            char *p;
            while ((p = strchr(val, ',')) != NULL) {
                *p = '\0';
                if (*val)
                    wl_append_word(&ignore, &ignore_last, copy(val));
                val = p + 1;
            }
            if (*val)
                wl_append_word(&ignore, &ignore_last, copy(val));
            in_list = "ignore";
        } else {
            fprintf(cp_err, "Error: osdi_bind: option %s is not known for %s\n", opt, kind);
            fprintf(cp_err, "%s", osdi_bind_usage);
            goto done;
        }
    }

    if (strcasecmp(module, "none") == 0) {
        if (alias || ignore || nval || (is_mos && (!polarity || strcmp(polarity, "type") != 0)))
            fprintf(cp_err, "Error: osdi_bind: only version= and has= are allowed with none\n");
        else if (!INPosdiBindRemove(kind, (int) level, has, version,
                                    has == NULL && version == NULL))
            fprintf(cp_err, "Warning: osdi_bind: %s level %ld%s%s was not bound\n",
                    kind, level, has ? " has=" : "", has ? has : "");
        goto done;
    }

    if (INPosdiBindAdd(kind, (int) level, module, has, version, polarity,
                       nval, pval, alias, ignore))
        fprintf(cp_out, "Note: osdi_bind: replaced the binding of %s level %ld%s%s\n",
                kind, level, has ? " has=" : "", has ? has : "");
    ignore = alias = NULL;      /* owned by the binding */

done:
    wl_free(alias);
    wl_free(ignore);
    wl_free(words);
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
