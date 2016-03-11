/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Initial lexer.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"

#include <errno.h>

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#ifdef HAVE_PWD_H
#include <sys/types.h>
#include <pwd.h>
#endif

/* MW. Linux has TIOCSTI, so we include all headers here */
#if !defined(__MINGW32__) && !defined(_MSC_VER)
#include <sys/ioctl.h>
#endif

#ifdef HAVE_SGTTY_H
#include <sys/types.h>
#include <sgtty.h>
#else
#ifdef HAVE_TERMIO_H
#include <sys/types.h>
#include <termio.h>
#else
#ifdef HAVE_TERMIOS_H
#include <sys/types.h>
#include <termios.h>
#endif
#endif
#endif

#include "ngspice/fteinput.h"
#include "lexical.h"

static void prompt(void);

extern bool cp_echo;  /* For CDHW patches: defined in variable.c */

FILE *cp_inp_cur = NULL;
int cp_event = 1;
bool cp_interactive = TRUE;
bool cp_bqflag = FALSE;
char *cp_promptstring = NULL;
char *cp_altprompt = NULL;
char cp_hash = '#';

static int numeofs = 0;


#define ESCAPE  '\033'


/* Return a list of words, with backslash quoting and '' quoting done.
 * Strings en(void) closed in "" or `` are made single words and returned,
 * but with the "" or `` still present. For the \ and '' cases, the
 * 8th bit is turned on (as in csh) to prevent them from being recognized,
 * and stripped off once all processing is done. We also have to deal with
 * command, filename, and keyword completion here.
 * If string is non-NULL, then use it instead of the fp. Escape and EOF
 * have no business being in the string.
 */

struct cp_lexer_buf
{
    int i, sz;
    char *s;
};


static inline void
push(struct cp_lexer_buf *buf, int c)
{
    if (buf->sz <= buf->i) {
        buf->sz += MAX(64, buf->sz);
        buf->s = TREALLOC(char, buf->s, buf->sz);
    }
    buf->s[buf->i++] = (char) c;
}


#define append(word)                            \
    wl_append_word(&wlist, &wlist_tail, word)


#define newword                                         \
    do {                                                \
        append(copy_substring(buf.s, buf.s + buf.i));   \
        buf.i = 0;                                      \
    } while(0)


/* CDHW Debug function */
/* CDHW used to perform function of set echo */

static void
pwlist_echo(wordlist *wlist, char *name)
{
    wordlist *wl;

    if (!cp_echo || cp_debug)
        return;

    fprintf(cp_err, "%s ", name);
    for (wl = wlist; wl; wl = wl->wl_next)
        fprintf(cp_err, "%s ", wl->wl_word);
    fprintf(cp_err, "\n");
}


static int
cp_readchar(char **string, FILE *fptr)
{
    if (*string == NULL)
        return input(fptr);

    if (**string)
        return *(*string)++;
    else
        return '\n';
}


/* CDHW */

wordlist *
cp_lexer(char *string)
{
    int c, d;
    int i;
    wordlist *wlist, *wlist_tail;
    struct cp_lexer_buf buf, linebuf;
    int paren;

    if (!cp_inp_cur)
        cp_inp_cur = cp_in;

    /* prompt for string if none is passed */
    if (!string && cp_interactive) {
        cp_ccon(TRUE);
        prompt();
    }

    wlist = wlist_tail = NULL;

    buf.sz = 0;
    buf.s = NULL;
    linebuf.sz = 0;
    linebuf.s = NULL;

nloop:
    if (wlist)
        wl_free(wlist);
    wlist = wlist_tail = NULL;
    buf.i = 0;
    linebuf.i = 0;
    paren = 0;

    for (;;) {

        c = cp_readchar(&string, cp_inp_cur);

    gotchar:

        if (string && (c == ESCAPE))
            continue;

        if ((c != EOF) && (c != ESCAPE))
            push(&linebuf, c);

        if (c != EOF)
            numeofs = 0;

        if (c != EOF)           /* Don't need to do this really. */
            c = strip(c);

        if ((c == '\\' && DIR_TERM != '\\') || (c == '\026') /* ^V */ ) {
            c = quote(cp_readchar(&string, cp_inp_cur));
            push(&linebuf, strip(c));
        }

        if ((c == '\n') && cp_bqflag)
            c = ' ';

        if ((c == EOF) && cp_bqflag)
            c = '\n';

        if ((c == cp_hash) && !cp_interactive && (linebuf.i == 1)) {
            if (string) {
                wl_free(wlist);
                tfree(buf.s);
                tfree(linebuf.s);
                return NULL;
            }
            while (((c = cp_readchar(&string, cp_inp_cur)) != '\n') && (c != EOF))
                ;
            goto nloop;
        }

        if ((c == '(') || (c == '[')) /* MW. Nedded by parse() */
            paren++;
        else if ((c == ')') || (c == ']'))
            paren--;

        switch (c) {

        case ' ':
        case '\t':
            if (buf.i > 0)
                newword;
            break;

        case '\n':
            if (buf.i)
                newword;
            if (!wlist_tail)
                append(NULL);
            goto done;

        case '\'':
            while ((c = cp_readchar(&string, cp_inp_cur)) != '\'')
            {
                if ((c == '\n') || (c == EOF) || (c == ESCAPE))
                    goto gotchar;
                push(&buf, quote(c));
                push(&linebuf, c);
            }
            push(&linebuf, '\'');
            break;

        case '"':
        case '`':
            d = c;
            push(&buf, d);
            while ((c = cp_readchar(&string, cp_inp_cur)) != d)
            {
                if ((c == '\n') || (c == EOF) || (c == ESCAPE))
                    goto gotchar;
                if (c == '\\') {
                    push(&linebuf, c);
                    c = cp_readchar(&string, cp_inp_cur);
                    push(&buf, quote(c));
                    push(&linebuf, c);
                } else {
                    push(&buf, c);
                    push(&linebuf, c);
                }
            }
            push(&buf, d);
            push(&linebuf, d);
            break;

        case '\004':
        case EOF:
            if (cp_interactive && !cp_nocc && !string) {

                if (linebuf.i == 0) {
                    if (cp_ignoreeof && (numeofs++ < 23)) {
                        fputs("Use \"quit\" to quit.\n", stdout);
                    } else {
                        fputs("quit\n", stdout);
                        cp_doquit();
                    }
                    append(NULL);
                    goto done;
                }

                push(&buf, '\0');
                push(&linebuf, '\0');

                // cp_ccom doesn't mess wlist, read only access to wlist->wl_word
                cp_ccom(wlist, buf.s, FALSE);
                (void) fputc('\r', cp_out);
                prompt();
                for (i = 0; linebuf.s[i]; i++)
#ifdef TIOCSTI
                    (void) ioctl(fileno(cp_out), TIOCSTI, linebuf.s + i);
#else
                fputc(linebuf.s[i], cp_out);  /* But you can't edit */
#endif
                goto nloop;
            }

            /* EOF during a source */
            if (cp_interactive) {
                fputs("quit\n", stdout);
                cp_doquit();
                append(NULL);
                goto done;
            }

            wl_free(wlist);
            tfree(buf.s);
            tfree(linebuf.s);
            return NULL;

        case ESCAPE:
            if (cp_interactive && !cp_nocc) {
                push(&buf, '\0');
                push(&linebuf, '\0');
                fputs("\b\b  \b\b\r", cp_out);
                prompt();
                for (i = 0; linebuf.s[i]; i++)
#ifdef TIOCSTI
                    (void) ioctl(fileno(cp_out), TIOCSTI, linebuf.s + i);
#else
                fputc(linebuf.s[i], cp_out);  /* But you can't edit */
#endif
                // cp_ccom doesn't mess wlist, read only access to wlist->wl_word
                cp_ccom(wlist, buf.s, TRUE);
                goto nloop;
            }
            goto ldefault;

        case ',':
            if ((paren < 1) && (buf.i > 0)) {
                newword;
                break;
            }
            goto ldefault;

        case ';':  /* CDHW semicolon inside parentheses is part of expression */
            if (paren > 0) {
                push(&buf, c);
                break;
            }
            goto ldefault;

        case '&':  /* va: $&name is one word */
            if ((buf.i >= 1) && (buf.s[buf.i - 1] == '$')) {
                push(&buf, c);
                break;
            }
            goto ldefault;

        case '<':
        case '>':  /* va: <=, >= are unbreakable words */
            if (string)
                if ((buf.i == 0) && (*string == '=')) {
                    push(&buf, c);
                    break;
                }
            goto ldefault;

        default:
            /* We have to remember the special case $<
             * here
             */
        ldefault:
            if ((cp_chars[c] & CPC_BRL) && (buf.i > 0))
                if ((c != '<') || (buf.s[buf.i - 1] != '$'))
                    newword;
            push(&buf, c);
            if (cp_chars[c] & CPC_BRR)
                if ((c != '<') || (buf.i < 2) || (buf.s[buf.i - 2] != '$'))
                    newword;
        }
    }

done:
    if (wlist->wl_word)
        pwlist_echo(wlist, "Command>");
    tfree(buf.s);
    tfree(linebuf.s);
    return wlist;
}


static void
prompt(void)
{
    char *s;

    if (cp_interactive == FALSE)
        return;

    if (cp_altprompt)
        s = cp_altprompt;
    else if (cp_promptstring)
        s = cp_promptstring;
    else
        s = "-> ";

    while (*s) {
        switch (strip(*s)) {
        case '!':
            fprintf(cp_out, "%d", cp_event);
            break;
        case '\\':
            if (s[1])
                (void) putc(strip(*++s), cp_out);
        default:
            (void) putc(strip(*s), cp_out);
        }
        s++;
    }

    (void) fflush(cp_out);
}
