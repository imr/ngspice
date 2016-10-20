/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1985 Wayne A. Christopher, U. C. Berkeley CAD Group
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/dvec.h"
#include "ngspice/fteinp.h"
#include "miscvars.h"


bool ft_nomod = FALSE;
bool ft_nopage = FALSE;
bool ft_parsedb = FALSE;
bool ft_vecdb = FALSE;
bool ft_simdb = FALSE;
bool ft_evdb = FALSE;
bool ft_grdb = FALSE;
bool ft_gidb = FALSE;
bool ft_controldb = FALSE;
bool ft_asyncdb = FALSE;

char *ft_setkwords[] = {

    "acct",
    "addcontrol",
    "altshow",
    "appendwrite",
    "askquit",
    "brief",
    "bypass",
    "chgtol",
    "color<num>",
    "cpdebug",
    "cptime",
    "curplot",
    "curplotdate",
    "curplotname",
    "curplottitle",
    "debug",
    "defad",
    "defas",
    "defl",
    "defw",
    "device",
    "diff_abstol",
    "diff_reltol",
    "diff_vntol",
    "display",
    "dontplot",
    "dpolydegree",
    "editor",
    "filetype",
    "fourgridsize",
    "geometry",
    "geometry<num>",
    "gmin",
    "gridsize",
    "hcopydev",
    "hcopydevtype",
    "hcopyfont",
    "hcopyfontsize",
    "hcopyscale",
    "height",
    "history",
    "ignoreeof",
    "interactive",
    "itl1",
    "itl2",
    "itl3",
    "itl4",
    "itl5",
    "list",
    "lprplot5",
    "lprps",
    "maxwins",
    "modelcard",
    "modelline",
    "moremode",
    "nfreqs",
    "noasciiplotvalue",
    "nobreak",
    "noclobber",
    "node",
    "noglob",
    "nogrid",
    "nomod",
    "nonomatch",
    "nopadding",
    "nopage",
    "noparse",
    "noprintscale",
    "nosort",
    "nosubckt",
    "notrnoise",
    "numdgt",
    "opts",
    "pivrel",
    "pivtol",
    "plots",
    "pointchars",
    "polydegree",
    "polysteps",
    "program",
    "prompt",
    "rawfile",
    "rawfileprec",
    "renumber",
    "rhost",
    "rprogram",
    "slowplot",
    "sourcepath",
    "spicepath",
    "subend",
    "subinvoke",
    "substart",
    "term",
    "ticmarks",
    "tnom",
    "trtol",
    "units",
    "unixcom",
    "width",
    "x11lineararcs",
    "xbrushheight",
    "xbrushwidth",
    "xfont",

    NULL
};
