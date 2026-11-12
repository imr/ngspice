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


/* Allowed characters.
   osdi_bind is no expression parser: a name (module, has=, version=,
   polarity parameter, alias= and ignore= entries) consists of letters,
   digits and the characters below; anything else, e.g. brackets or
   operators, is rejected. Extend the list here if a real model card
   needs more. Separators (= , :) are split off before the check, the
   values of polarity=<par>:<nval>:<pval> must be numbers, quotes are
   removed before (as the ngspice command line does elsewhere). */
static const char osdi_bind_name_chars[] = "_.-";

/* Is name made of the allowed characters? Else report it. */
static bool
osdi_bind_name_ok(const char *what, const char *name)
{
    const char *c;
    for (c = name; *c; c++)
        if (!isalnum((unsigned char) *c) && !strchr(osdi_bind_name_chars, *c)) {
            fprintf(cp_err, "Error: osdi_bind: character '%c' is not allowed in %s %s\n",
                    *c, what, name);
            return FALSE;
        }
    if (!*name) {
        fprintf(cp_err, "Error: osdi_bind: %s is empty\n", what);
        return FALSE;
    }
    return TRUE;
}


/* Remove the quotes of an option value in place: alias="ab:area" */
static void
osdi_bind_unquote(char *s)
{
    char *dst = s;
    for (; *s; s++)
        if (*s != '"' && *s != '\'')
            *dst++ = *s;
    *dst = '\0';
}


/* Is s a number (integer or real)? */
static bool
osdi_bind_is_number(const char *s)
{
    char *end;
    if (!*s)
        return FALSE;
    (void) strtod(s, &end);
    return *end == '\0';
}


/* The conditions of a binding as in its listing: " version=<v> has=<par>" */
static char *
osdi_bind_conditions(const char *version, const char *has)
{
    return tprintf("%s%s%s%s", version ? " version=" : "", version ? version : "",
                   has ? " has=" : "", has ? has : "");
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
    char *module, *end, *cond;
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
    osdi_bind_unquote(module);
    if (!osdi_bind_name_ok("module", module))
        goto done;

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
        if (val)
            osdi_bind_unquote(val);
        if (!val || !*val) {
            fprintf(cp_err, "Error: osdi_bind: %s is not of the form name=value\n", opt);
            fprintf(cp_err, "%s", osdi_bind_usage);
            goto done;
        }
        if (strcasecmp(opt, "has") == 0) {
            if (!osdi_bind_name_ok("has=", val))
                goto done;
            has = val;
        } else if (strcasecmp(opt, "version") == 0) {
            if (!osdi_bind_name_ok("version=", val))
                goto done;
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
                if (!osdi_bind_name_ok("alias=", item) ||
                    !osdi_bind_name_ok("alias=", colon + 1))
                    goto done;
                wl_append_word(&alias, &alias_last, copy(item));
                wl_append_word(&alias, &alias_last, copy(colon + 1));
            }
            in_list = "alias";
        } else if (strcasecmp(opt, "polarity") == 0 && is_mos) {
            /* <par>, <par>:<nval>:<pval>, or none */
            char *c1 = strchr(val, ':');
            if (c1) {
                char *c2 = strchr(c1 + 1, ':');
                if (c1 == val || !c2 || strchr(c2 + 1, ':')) {
                    fprintf(cp_err, "Error: osdi_bind: polarity=%s is not <par>:<nval>:<pval>\n", val);
                    goto done;
                }
                *c1 = *c2 = '\0';
                nval = c1 + 1;
                pval = c2 + 1;
                if (!osdi_bind_is_number(nval) || !osdi_bind_is_number(pval)) {
                    fprintf(cp_err, "Error: osdi_bind: polarity=%s:%s:%s: the values are not numbers\n",
                            val, nval, pval);
                    goto done;
                }
            }
            if (!osdi_bind_name_ok("polarity=", val))
                goto done;
            polarity = strcasecmp(val, "none") == 0 ? NULL : val;
        } else if (strcasecmp(opt, "ignore") == 0) {
            char *item = val, *next;
            for (; item; item = next) {
                next = strchr(item, ',');
                if (next)
                    *next++ = '\0';
                if (!*item)
                    continue;
                if (!osdi_bind_name_ok("ignore=", item))
                    goto done;
                wl_append_word(&ignore, &ignore_last, copy(item));
            }
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
                                    has == NULL && version == NULL)) {
            cond = osdi_bind_conditions(version, has);
            fprintf(cp_err, "Warning: osdi_bind: %s level %ld%s was not bound\n",
                    kind, level, cond);
            tfree(cond);
        }
        goto done;
    }

    cond = osdi_bind_conditions(version, has);
    if (INPosdiBindAdd(kind, (int) level, module, has, version, polarity,
                       nval, pval, alias, ignore))
        fprintf(cp_out, "Note: osdi_bind: replaced the binding of %s level %ld%s\n",
                kind, level, cond);
    tfree(cond);
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
