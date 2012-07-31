/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/*
 * Initial lexer.
 */

#include "ngspice/config.h"
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

#if !defined(__MINGW32__) && !defined(_MSC_VER)
	/* MW. Linux has TIOCSTI, so we include all headers here */
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

#define NEW_BSIZE_SP 2*BSIZE_SP

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

#define append \
    {   wordlist *aux = wl_cons(NULL, NULL); \
        if (cw) \
           cw->wl_next = aux; \
        aux->wl_prev = cw; \
        cw = aux; \
        if (!wlist) \
            wlist = cw; \
    }

#define newword \
        if (delayed) append;  \
        delayed = 0;  \
        cw->wl_word = copy(buf);  \
        delayed = 1; \
        bzero(buf, NEW_BSIZE_SP); \
        i = 0;


/* CDHW Debug function */

static void
pwlist_echo(   /*CDHW used to perform function of set echo */
    wordlist *wlist,
    char *name)
{
    wordlist *wl;

    if ((!cp_echo)||cp_debug) /* cpdebug prints the same info */
        return;
    fprintf(cp_err, "%s ", name);
    for (wl = wlist; wl; wl = wl->wl_next)
        fprintf(cp_err, "%s ", wl->wl_word );
      fprintf(cp_err, "\n");
    return;
}

/* CDHW */

wordlist *
cp_lexer(char *string)
{
    int c, d;
    int i, j;
    wordlist *wlist = NULL, *cw = NULL;
    char buf[NEW_BSIZE_SP], linebuf[NEW_BSIZE_SP];
    int paren;
    int delayed;

   if (cp_inp_cur == NULL)
        cp_inp_cur = cp_in;

    if (!string && cp_interactive) {/* prompt for string if none is passed */
        cp_ccon(TRUE);
        prompt();
    }
nloop:
    wlist = cw = NULL;
    i = 0;
    j = 0;
    paren = 0;
    bzero(linebuf, NEW_BSIZE_SP);
    bzero(buf, NEW_BSIZE_SP);
    delayed = 1;
    for (;;) {
        if (string) {
            c = *string++;
            if (c == '\0')
                c = '\n';
            if (c == ESCAPE)
                c = '[';
        } else
            c = input(cp_inp_cur);

gotchar:
	if ((c != EOF) && (c != ESCAPE))
	    linebuf[j++] = (char) c;
        if (c != EOF)
            numeofs = 0;
        if (i == NEW_BSIZE_SP - 1) {
            fprintf(cp_err, "Warning: word too long.\n");
            c = ' ';
        }
        if (j == NEW_BSIZE_SP - 1) {
            fprintf(cp_err, "Warning: line too long.\n");
            if (cp_bqflag)
                c = EOF;
            else
                c = '\n';
        }
        if (c != EOF)
            c = strip(c);   /* Don't need to do this really. */
        if ((c == '\\' && DIR_TERM != '\\') || (c == '\026') /* ^V */ ) {
            c = quote(string ? *string++ : input(cp_inp_cur));
            linebuf[j++] = (char) strip(c);
        }
        if ((c == '\n') && cp_bqflag)
            c = ' ';
        if ((c == EOF) && cp_bqflag)
            c = '\n';
        if ((c == cp_hash) && !cp_interactive && (j == 1)) {
            if (delayed) append;
            delayed = 0;
            if (string)
                return (NULL);
            while (((c = input(cp_inp_cur)) != '\n') && (c != EOF))
                ;
            wlist = cw = NULL;
            goto nloop;
        }

        if ((c == '(') || (c == '[')) /* MW. Nedded by parse() */
	    paren++;
	else if ((c == ')') || (c == ']'))
	    paren--;

        switch (c) {
	case ' ':
	case '\t':
            if (i > 0) {
                newword;
            }
            break;

	case '\n':
            if (i) {
                buf[i] = '\0';
                newword;
            }
            if (delayed) append;
            delayed = 0;
                if (cw->wl_prev) {
                    cw->wl_prev->wl_next = NULL;
                    tfree(cw);
                } else {
                    cw->wl_word = NULL;
                }
            goto done;

	case '\'':
            while (((c = (string ? *string++ : 
                    input(cp_inp_cur))) != '\'')
                    && (i < NEW_BSIZE_SP - 1)) {
                if ((c == '\n') || (c == EOF) || (c == ESCAPE))
                    goto gotchar;
                else {
                    buf[i++] = (char) quote(c);
                    linebuf[j++] = (char) c;
                }
            }
            linebuf[j++] = '\'';
            break;

	case '"':
	case '`':
            d = c;
            buf[i++] = (char) d;
            while (((c = (string ? *string++ : input(cp_inp_cur)))
                    != d) && (i < NEW_BSIZE_SP - 2)) {
                if ((c == '\n') || (c == EOF) || (c == ESCAPE))
                    goto gotchar;
                else if (c == '\\') {
                    linebuf[j++] = (char) c;
                    c = (string ? *string++ :
                            input(cp_inp_cur));
                    buf[i++] = (char) quote(c);
                    linebuf[j++] = (char) c;
                } else {
                    buf[i++] = (char) c;
                    linebuf[j++] = (char) c;
                }
            }
            buf[i++] = (char) d;
            linebuf[j++] = (char) d;
            break;

	case '\004':
	case EOF:
            if (delayed) append;
            delayed = 0;
            if (cp_interactive && !cp_nocc && 
                    (string == NULL)) {
                if (j == 0) {
                    if (cp_ignoreeof && (numeofs++
                            < 23)) {
                        fputs(
                    "Use \"quit\" to quit.\n",
                            stdout);
                    } else {
                        fputs("quit\n", stdout);
                        cp_doquit();
                    }
                    goto done;
                }
                // cp_ccom doesn't mess wlist, read only access to wlist->wl_word
                cp_ccom(wlist, buf, FALSE);
                wl_free(wlist);
                (void) fputc('\r', cp_out);
                prompt();
                for (j = 0; linebuf[j]; j++)
#ifdef TIOCSTI
                    (void) ioctl(fileno(cp_out), TIOCSTI, linebuf + j);
#else
		    fputc(linebuf[j], cp_out);	/* But you can't edit */
#endif
                wlist = cw = NULL;
                goto nloop;
            } else    /* EOF during a source */
	    {
                if (cp_interactive) {
                    fputs("quit\n", stdout);
                    cp_doquit();
                    goto done;
                } else
                    return (NULL);
            }
	case ESCAPE:
            if (cp_interactive && !cp_nocc) {
                fputs("\b\b  \b\b\r", cp_out);
                prompt();
                for (j = 0; linebuf[j]; j++)
#ifdef TIOCSTI
                    (void) ioctl(fileno(cp_out), TIOCSTI, linebuf + j);
#else
		    fputc(linebuf[j], cp_out);	/* But you can't edit */
#endif
                if (delayed) append;
                delayed = 0;
                // cp_ccom doesn't mess wlist, read only access to wlist->wl_word
                cp_ccom(wlist, buf, TRUE);
                wl_free(wlist);
                wlist = cw = NULL;
                goto nloop;
            } goto ldefault; /* else continue with default ... */
	case ',':
	    if (paren < 1 && i > 0) {
		newword;
		break;
	    } goto ldefault; 
	case ';':  /*CDHW semicolon inside parentheses is part of expression CDHW*/
	    if (paren > 0) {
	        buf[i++] = (char) c;
		break;
	    } goto ldefault; 
	case '&':  /* va: $&name is one word */
	    if (i==1 && buf[i-1]=='$' && c=='&') {
	        buf[i++] = (char) c;
	        break;
	    } goto ldefault; /* else continue with default ... */
	case '<':
	case '>':  /* va: <=, >= are unbreakable words */
       if(string)
	    if (i==0 && (*string=='=')) {
	        buf[i++] = (char) c;
	        break;
	    } goto ldefault; /* else continue with default ... */
	default:
            /* We have to remember the special case $<
             * here
             */
ldefault:   if ((cp_chars[c] & CPC_BRL) && (i > 0)) {
                if ((c != '<') || (buf[i - 1] != '$')) {
                    newword;
                }
            }
            buf[i++] = (char) c;
            if (cp_chars[c] & CPC_BRR) {
                if ((c != '<') || (i < 2) ||
                        (buf[i - 2] != '$')) {
                    newword;
                }
            }
        }
    }
done:

    if (wlist->wl_word) pwlist_echo(wlist,"Command>");
    return (wlist);
}

static void
prompt(void)
{
    char *s;

    if (cp_interactive == FALSE)
        return;
    if (cp_promptstring == NULL)
        s = "-> ";
    else
        s = cp_promptstring;
    if (cp_altprompt)
        s = cp_altprompt;
    while (*s) {
        switch (strip(*s)) {
            case '!':
                fprintf(cp_out, "%d", cp_event);
                break;
            case '\\':
                if (*(s + 1))
                    (void) putc(strip(*++s), cp_out);
            default:
                (void) putc(strip(*s), cp_out);
        }
        s++;
    }
    (void) fflush(cp_out);
    return;
}
