/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

/* Initialize io, cp_chars[], variable "history". */

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "../spicelib/analysis/com_optran.h"

#include "init.h"
#include "variable.h"


void
cp_init(void)
/* called from ft_cpinit() in cpitf.c.
   Uses global variables:
   cp_chars[128]
   cp_maxhistlength (set to 10000 in com_history.c)
   cp_curin, cp_curout, cp_curerr (defined in streams.c)
   cp_no_histsubst
*/
{
    cp_vset("history", CP_NUM, &cp_maxhistlength);

    cp_curin = stdin;
    cp_curout = stdout;
    cp_curerr = stderr;

    /* Enable history substitution */
    if (cp_getvar("histsubst", CP_BOOL, NULL, 0))
        cp_no_histsubst = FALSE;

    /* io redirection in streams.c:
       cp_in set to cp_curin etc. */
    cp_ioreset();

    /*set a variable oscompiled containing the OS at compile time
         [0], Other
         [1], MINGW for MS Windows
		 [2], Cygwin for MS Windows
         [3], FreeBSD
	     [4], OpenBSD
		 [5], Solaris
         [6], Linux
         [7], macOS
         [8], Visual Studio for MS Windows
     The variable may be used in a .control section to perform OS
     specific actions (setting fonts etc.).
     */
    int itmp;
#if OS_COMPILED == 1
    itmp = 1;
#elif OS_COMPILED == 2
    itmp = 2;
#elif OS_COMPILED == 3
    itmp = 3;
#elif OS_COMPILED == 4
    itmp = 4;
#elif OS_COMPILED == 5
    itmp = 5;
#elif OS_COMPILED == 6
    itmp = 6;
#elif OS_COMPILED == 7
    itmp = 7;
#else
    itmp = 0;
#endif
    /* not using configure.ac */
#ifdef _MSC_VER
    itmp = 8;
#endif
    cp_vset("oscompiled", CP_NUM, &itmp);

    /* To make optran the standard, call com_optran here.
    May be overridden by entry in spinit or .spiceinit or a local call
    in .control. */
    {
        wordlist* wl_optran;
        /* the default optran parameters: 1 1 1 100n 10u 0 */
        char* sbuf[7];
        sbuf[0] = "1";
        sbuf[1] = "1";
        sbuf[2] = "1";
        sbuf[3] = "100n";
        sbuf[4] = "10u";
        sbuf[5] = "0";
        sbuf[6] = NULL;
        wl_optran = wl_build((const char* const*)sbuf);
        com_optran(wl_optran);
        wl_free(wl_optran);
    }
}
