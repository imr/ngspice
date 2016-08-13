/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/* Do history substitutions.  */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"

#include "com_history.h"

#ifdef HAVE_GNUREADLINE
/* Added GNU Readline Support -- Andrew Veliath <veliaa@rpi.edu> */
#include <readline/readline.h>
#include <readline/history.h>
#endif

#ifdef HAVE_BSDEDITLINE
/* SJB added edit line support 2005-05-05 */
#include <editline/readline.h>
#endif


static wordlist *dohsubst(char *string);
static wordlist *dohmod(char **string, wordlist *wl);
static wordlist *hpattern(char *buf);
static wordlist *hprefix(char *buf);
static wordlist *getevent(int num);
#if !defined(HAVE_GNUREADLINE) && !defined(HAVE_BSDEDITLINE)
static void cp_hprint(int eventhi, int eventlo, bool rev);
static void freehist(int num);
#endif
static char *dohs(char *pat, char *str);


struct histent *cp_lastone = NULL;

int  cp_maxhistlength = 10000;        /* Chris Inbody */
char cp_hat = '^';
char cp_bang = '!';
bool cp_didhsubst;

static struct histent *histlist = NULL;
static int histlength = 0;


/* First check for a ^ at the beginning of the line, and then search
 * each word for !. Following this can be any of string, number,
 * ?string, -number ; then there may be a word specifier, the same as
 * csh, and then the : modifiers. For the :s modifier, the syntax is
 * :sXoooXnnnX, where X is any character, and ooo and nnn are strings
 * not containing X.
 */

wordlist *
cp_histsubst(wordlist *wlist)
{
    wordlist *nwl, *w, *n;
    char *s, *b;

    /* Replace ^old^new with !:s^old^new. */

    cp_didhsubst = FALSE;
    if (*wlist->wl_word == cp_hat) {
        char *x = wlist->wl_word;
        wlist->wl_word = tprintf("%c%c:s%s", cp_bang, cp_bang, wlist->wl_word);
        tfree(x);
    }

    for (w = wlist; w; w = w->wl_next) {
        b = w->wl_word;
        for (s = b; *s; s++)
            if (*s == cp_bang) {
                cp_didhsubst = TRUE;
                n = dohsubst(s + 1);
                if (!n) {
                    wlist->wl_word = NULL;
                    return (wlist);
                }
                if (s > b) {
                    char *x = n->wl_word;
                    n->wl_word = tprintf("%.*s%s", (int)(s-b), b, n->wl_word);
                    tfree(x);
                }
                nwl = wl_splice(w, n);
                if (wlist == w)
                    wlist = n;
                w = nwl;
                break;
            }
    }

    return (wlist);
}


/* Do a history substitution on one word. Figure out which event is
 * being referenced, then do word selections and modifications, and
 * then stick anything left over on the end of the last word.
 */

static wordlist *
dohsubst(char *string)
{
    wordlist *wl, *nwl;
    char buf[BSIZE_SP], *s, *r = NULL, *t;

    if (*string == cp_bang) {
        if (cp_lastone) {
            wl = cp_lastone->hi_wlist;
            string++;
        } else {
            fprintf(cp_err, "0: event not found.\n");
            return (NULL);
        }
    } else {
        switch (*string) {

        case '-':
            wl = getevent(cp_event - scannum(++string));
            if (!wl)
                return (NULL);
            while (isdigit_c(*string))
                string++;
            break;

        case '?':
            (void) strcpy(buf, string + 1);
            if ((s = strchr(buf, '?')) != NULL)
                *s = '\0';
            wl = hpattern(buf);
            if (!wl)
                return (NULL);
            if (s == NULL) /* No modifiers on this one. */
                return (wl_copy(wl));
            break;

        case '\0':  /* Maybe this should be cp_event. */
            wl = wl_cons(copy("!"), NULL);
            cp_didhsubst = FALSE;
            return (wl);

        default:
            if (isdigit_c(*string)) {
                wl = getevent(scannum(string));
                if (!wl)
                    return (NULL);
                while (isdigit_c(*string))
                    string++;
            } else {
                (void) strcpy(buf, string);
                for (s = ":^$*-%"; *s; s++) {
                    t = strchr(buf, *s);
                    if (t && ((t < r) || !r)) {
                        r = t;
                        string += r - buf;
                    }
                }
                if (r)
                    *r = '\0';
                else
                    while (*string)
                        string++;
                if ((buf[0] == '\0') && cp_lastone)
                    wl = cp_lastone->hi_wlist;
                else
                    wl = hprefix(buf);
                if (!wl)
                    return (NULL);
            }
        }
    }

    if (wl == NULL) {   /* Shouldn't happen. */
        fprintf(cp_err, "Event not found.\n");
        return (NULL);
    }

    nwl = dohmod(&string, wl_copy(wl));
    if (!nwl)
        return (NULL);

    if (*string) {
        char *x;
        for (wl = nwl; wl->wl_next; wl = wl->wl_next)
            ;
        x = wl->wl_word;
        wl->wl_word = tprintf("%s%s", wl->wl_word, string);
        tfree(x);
    }

    return (nwl);
}


static wordlist *
dohmod(char **string, wordlist *wl)
{
    wordlist *w;
    char *s;
    char *r = NULL, *t;
    int numwords, eventlo, eventhi, i;
    bool globalsubst;

anothermod:
    numwords = wl_length(wl);
    globalsubst = FALSE;
    eventlo = 0;
    eventhi = numwords - 1;

    /* Now we know what wordlist we want. Take care of modifiers now. */
    r = NULL;
    for (s = ":^$*-%"; *s; s++) {
        t = strchr(*string, *s);
        if (t && ((t < r) || (r == NULL)))
            r = t;
    }

    if (!r)     /* No more modifiers. */
        return (wl);

    *string = r;
    if (**string == ':')
        (*string)++;

    switch (**string) {
    case '$':   /* Last word. */
        eventhi = eventlo = numwords - 1;
        break;
    case '*':   /* Words 1 through $ */
        if (numwords == 1)
            return (NULL);
        eventlo = 1;
        eventhi = numwords - 1;
        break;
    case '-':   /* Words 0 through ... */
        eventlo = 0;
        if (*(*string + 1))
            eventhi = scannum(*string + 1);
        else
            eventhi = numwords - 1;
        if (eventhi > numwords - 1)
            eventhi = numwords - 1;
        break;
    case 'p':   /* Print the command and don't execute it.
                 * This doesn't work quite like csh.
                 */
        wl_print(wl, cp_out);
        (void) putc('\n', cp_out);
        return (NULL);
    case 's':   /* Do a substitution. */
        for (w = wl; w; w = w->wl_next) {
            s = dohs(*string + 1, w->wl_word);
            if (s) {
                tfree(w->wl_word);
                w->wl_word = s;
                if (globalsubst == FALSE) {
                    while (**string)
                        (*string)++;
                    break;
                }
            }
        }
        /* In case globalsubst is TRUE... */
        while (**string)
            (*string)++;
        break;
    default:
        if (!isdigit_c(**string)) {
            fprintf(cp_err, "Error: %s: bad modifier.\n",
                    *string);
            return (NULL);
        }
        i = scannum(*string);
        if (i > eventhi) {
            fprintf(cp_err, "Error: bad event number %d\n",
                    i);
            return (NULL);
        }
        eventhi = eventlo = i;
        while (isdigit_c(**string))
            (*string)++;
        if (**string == '*')
            eventhi = numwords - 1;
        if (**string == '-') {
            if (!isdigit_c(*(*string + 1))) {
                eventhi = numwords - 1;
            } else {
                eventhi = scannum(++*string);
                while (isdigit_c(**string))
                    (*string)++;
            }
        }
    }

    /* Now change the word list accordingly and make another pass
     * if there is more of the substitute left.
     */

    wl = wl_range(wl, eventlo, eventhi);
    numwords = wl_length(wl);
    if (**string && *++*string)
        goto anothermod;

    return (wl);
}


/* Look for an event with a pattern in it... */

static wordlist *
hpattern(char *buf)
{
    struct histent *hi;
    wordlist *wl;

    if (*buf == '\0') {
        fprintf(cp_err, "Bad pattern specification.\n");
        return (NULL);
    }

    for (hi = cp_lastone; hi; hi = hi->hi_prev)
        for (wl = hi->hi_wlist; wl; wl = wl->wl_next)
            if (substring(buf, wl->wl_word))
                return (hi->hi_wlist);

    fprintf(cp_err, "%s: event not found.\n", buf);
    return (NULL);
}


static wordlist *
hprefix(char *buf)
{
    struct histent *hi;

    if (*buf == '\0') {
        fprintf(cp_err, "Bad pattern specification.\n");
        return (NULL);
    }

    for (hi = cp_lastone; hi; hi = hi->hi_prev)
        if (hi->hi_wlist && prefix(buf, hi->hi_wlist->wl_word))
            return (hi->hi_wlist);

    fprintf(cp_err, "%s: event not found.\n", buf);
    return (NULL);
}


/* Add a wordlist to the history list. (Done after the first parse.) Note
 * that if event numbers are given in a random order that's how they'll
 * show up in the history list.
 */

void
cp_addhistent(int event, wordlist *wlist)
{
    /* MW. This test is not needed if everything works right
       if (cp_lastone && !cp_lastone->hi_wlist)
       fprintf(cp_err, "Internal error: bad history list\n"); */

    if (cp_lastone == NULL) {
        /* MW. the begging - initialize histlength */
        histlength = 1;

        cp_lastone = histlist = TMALLOC(struct histent, 1);
        cp_lastone->hi_prev = NULL;
    } else {
        cp_lastone->hi_next = TMALLOC(struct histent, 1);
        cp_lastone->hi_next->hi_prev = cp_lastone;
        cp_lastone = cp_lastone->hi_next;
    }

    cp_lastone->hi_next = NULL;
    cp_lastone->hi_event = event;
    cp_lastone->hi_wlist = wl_copy(wlist);

#if !defined(HAVE_GNUREADLINE) && !defined(HAVE_BSDEDITLINE)
    freehist(histlength - cp_maxhistlength);
    histlength++;
#endif
}


/* Get a copy of the wordlist associated with an event. Error if out
 * of range.
 */

static wordlist *
getevent(int num)
{
    struct histent *hi;

    for (hi = histlist; hi; hi = hi->hi_next)
        if (hi->hi_event == num)
            break;

    if (hi == NULL) {
        fprintf(cp_err, "%d: event not found.\n", num);
        return (NULL);
    }

    return (wl_copy(hi->hi_wlist));
}


#if !defined(HAVE_GNUREADLINE) && !defined(HAVE_BSDEDITLINE)

/* Print out history between eventhi and eventlo.
 * This doesn't remember quoting, so 'hodedo' prints as hodedo.
 */

static void
cp_hprint(int eventhi, int eventlo, bool rev)
{
    struct histent *hi;

    if (rev) {
        for (hi = histlist; hi->hi_next; hi = hi->hi_next)
            ;
        for (; hi; hi = hi->hi_prev)
            if ((hi->hi_event <= eventhi) &&
                (hi->hi_event >= eventlo) &&
                hi->hi_wlist)
            {
                fprintf(cp_out, "%d\t", hi->hi_event);
                wl_print(hi->hi_wlist, cp_out);
                (void) putc('\n', cp_out);
            }
    } else {
        for (hi = histlist; hi; hi = hi->hi_next)
            if ((hi->hi_event <= eventhi) &&
                (hi->hi_event >= eventlo) &&
                hi->hi_wlist)
            {
                fprintf(cp_out, "%d\t", hi->hi_event);
                wl_print(hi->hi_wlist, cp_out);
                (void) putc('\n', cp_out);
            }
    }
}


/* This just gets rid of the first num entries on the history list, and
 * decrements histlength.
 */

static void
freehist(int num)
{
    struct histent *hi;

    if (num < 1)
        return;

    histlength -= num;
    hi = histlist;

    while (num-- && histlist->hi_next)
        histlist = histlist->hi_next;

    if (histlist->hi_prev) {
        histlist->hi_prev->hi_next = NULL;
        histlist->hi_prev = NULL;
    } else {
        fprintf(cp_err, "Internal error: history list mangled\n");
        exit(0);                                   /* Chris Inbody */
    }

    while (hi->hi_next) {
        wl_free(hi->hi_wlist);
        hi = hi->hi_next;
        tfree(hi->hi_prev);
    }

    wl_free(hi->hi_wlist);
    tfree(hi);
}

#endif /* !defined(HAVE_GNUREADLINE) && !defined(HAVE_BSDEDITLINE) */


/* Do a :s substitution. */

static char *
dohs(char *pat, char *str)
{
    char schar, *s, *p, buf[BSIZE_SP];
    int i = 0, plen;
    bool ok = FALSE;

    pat = copy(pat);    /* Don't want to mangle anything. */
    schar = *pat++;
    s = strchr(pat, schar);
    if (s == NULL) {
        fprintf(cp_err, "Bad substitute.\n");
        return (NULL);
    }
    *s++ = '\0';

    p = strchr(s, schar);
    if (p)
        *p = '\0';

    plen = (int) strlen(pat) - 1;
    for (i = 0; *str; str++) {
        if ((*str == *pat) && prefix(pat, str) && (ok == FALSE)) {
            for (p = s; *p; p++)
                buf[i++] = *p;
            str += plen;
            ok = TRUE;
        } else {
            buf[i++] = *str;
        }
    }
    buf[i] = '\0';

    if (ok)
        return (copy(buf));
    else
        return (NULL);
}


/* The "history" command. history [-r] [number] */

void
com_history(wordlist *wl)
{
    bool rev = FALSE;

    if (wl && eq(wl->wl_word, "-r")) {
        wl = wl->wl_next;
        rev = TRUE;
    }

#if defined(HAVE_GNUREADLINE) || defined(HAVE_BSDEDITLINE)
    /* Added GNU Readline Support -- Andrew Veliath <veliaa@rpi.edu> */
    {
        HIST_ENTRY *he;
        int i, N;

        N = (wl == NULL) ? history_length : atoi(wl->wl_word);

        if (N < 0)
            N = 0;
        if (N > history_length)
            N = history_length;

        if (rev)
            for (i = history_length; i > 0 && N; --i, --N) {
                he = history_get(i);
                if (!he)
                    return;
                fprintf(cp_out, "%d\t%s\n", i, he->line);
            }
        else
            for (i = history_length - N + 1; i <= history_length; ++i) {
                he = history_get(i);
                if (!he)
                    return;
                fprintf(cp_out, "%d\t%s\n", i, he->line);
            }
    }
#else
    if (wl == NULL)
        cp_hprint(cp_event - 1, cp_event - histlength, rev);
    else
        cp_hprint(cp_event - 1, cp_event - 1 - atoi(wl->wl_word), rev);
#endif
}
