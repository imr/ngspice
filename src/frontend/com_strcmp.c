#include "ngspice/ngspice.h"

#include "ngspice/bool.h"
#include "ngspice/wordlist.h"

#include "com_strcmp.h"
#include "variable.h"


/* This is a truly evil thing */
void
com_strcmp(wordlist *wl)
{
    char *var, *s1, *s2;
    int i;

    var = wl->wl_word;
    s1 = cp_unquote(wl->wl_next->wl_word);
    s2 = cp_unquote(wl->wl_next->wl_next->wl_word);

    i = strcmp(s1, s2);
    tfree(s1);/*DG  cp_unquote memory leak*/
    tfree(s2);
    cp_vset(var, CP_NUM, &i);
}

/* These must be more evil still. */

void com_strstr(wordlist *wl)
{
    char *var, *s1, *s2;
    int   i;

    s1 = cp_unquote(wl->wl_next->wl_word);
    s2 = cp_unquote(wl->wl_next->wl_next->wl_word);
    if (*s2) {
        var = strstr(s1, s2); // Search for s2 in s1
        if (var)
            i = (int)(var - s1); // Offset to match
        else
            i = -1;
    } else {
        i = (int)strlen(s1);     // Length
    }
    tfree(s1);
    tfree(s2);
    cp_vset(wl->wl_word, CP_NUM, &i);
}

void com_strslice(wordlist *wl)
{
    char *var, *s1, *tp, tmp;
    int   offset, length, actual;

    var = wl->wl_word;
    wl = wl->wl_next;
    s1 = cp_unquote(wl->wl_word);
    wl = wl->wl_next;
    offset = atoi(wl->wl_word);
    length = atoi(wl->wl_next->wl_word);
    actual = (int)strlen(s1);
    if (offset < 0)
        offset = actual + offset;
    if (length + offset > actual)
        length = actual - offset;
    if (length > 0 && offset >= 0) {
        tp = s1 + offset + length;
        tmp = *tp;
        *tp = '\0';
        cp_vset(var, CP_STRING, s1 + offset);
        *tp = tmp;
    } else {
        cp_vset(var, CP_STRING, "");
    }
    tfree(s1);
}
