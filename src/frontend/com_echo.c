#include <config.h>
#include <ngspice.h>
#include <wordlist.h>
#include <bool.h>

#include "quote.h"
#include "streams.h"

void
com_echo(wordlist *wlist)
{
    bool nl = TRUE;

    if (wlist && eq(wlist->wl_word, "-n")) {
        wlist = wlist->wl_next;
        nl = FALSE;
    }

    while (wlist) {
        fputs(cp_unquote(wlist->wl_word), cp_out);
        if (wlist->wl_next)
            fputs(" ", cp_out);
        wlist = wlist->wl_next;
    }
    if (nl)
        fputs("\n", cp_out);
}

