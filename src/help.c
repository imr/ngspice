/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1986 Wayne A. Christopher, U. C. Berkeley CAD Group 
**********/

/*
 * The main routine for the help system in stand-alone mode.
 */

#include <config.h>
#include "ngspice.h"
#include "cpdefs.h"
#include "hlpdefs.h"

#include "frontend/variable.h"

#ifndef X_DISPLAY_MISSING
Widget toplevel;
#endif

FILE *cp_in, *cp_out, *cp_err;
char	*Spice_Exec_Dir	= NGSPICEBINDIR;
char	*Spice_Lib_Dir	= NGSPICEDATADIR;
char	*Def_Editor	= "vi";
int	AsciiRawFile	= 0;

char	*Bug_Addr	= "";
char	*Spice_Host	= "";
char	*Spiced_Log	= "";

/* dummy declaration so CP.a doesn't pull in lexical.o and other objects */
bool cp_interactive = FALSE;

char *hlp_filelist[] = { "ngspice", 0 };

int
main(int ac, char **av)
{
    wordlist *wl = NULL;

#ifndef X_DISPLAY_MISSING
    char *displayname;
    /* grrr, Xtk forced contortions */
    char *argv[2];
    int argc = 2;
    char buf[512];
#endif

    ivars( );

    cp_in = stdin;
    cp_out = stdout;
    cp_err = stderr;

#ifndef X_DISPLAY_MISSING

    if (cp_getvar("display", VT_STRING, buf)) {
      displayname = buf;
    } else if (!(displayname = getenv("DISPLAY"))) {
      fprintf(stderr, "Can't open X display.");
      goto out;
    }

    argv[0] = "nutmeg";
    argv[1] = displayname;
    /* initialize X toolkit */
    toplevel = XtInitialize("nutmeg", "Nutmeg", NULL, 0, &argc, argv);

#endif

out:
    if (ac > 1)
        wl = wl_build(av + 1);
    hlp_main(Help_Path, wl);

#ifndef X_DISPLAY_MISSING
    if (hlp_usex) {
	printf("Hit control-C when done.\n");		/* sigh */
	XtMainLoop();
    }
#endif

    exit(EXIT_NORMAL);
}

void
fatal(char *s)
{
    fprintf(stderr, "fatal error: %s\n", s);
    exit(1);
}

/* There is a conflict witj another cp_printword in cp/quote.c 
static void
cp_printword(s)
    char *s;
{
    printf("%s", s);
    return;
}

*/

bool
cp_getvar(char *n, int t, char *r)
{
    return (FALSE);
}

char *
cp_tildexpand(char *s)
{
	return tilde_expand(s);
}
