#include <ngspice.h>
#include <wordlist.h>
#include <bool.h>
#include <variable.h>
#include <hlpdefs.h>

#include "com_ghelp.h"
#include "com_help.h"
#include "variable.h"
#include "streams.h"
#include "cpextern.h"

void
com_ghelp(wordlist *wl)
{
    char *npath;
    char *path = Help_Path;
    char buf[BSIZE_SP];
    int i;

    if (cp_getvar("helppath", VT_STRING, buf))
        path = copy(buf);
    if (!path) {
        fprintf(cp_err, "Note: defaulting to old help.\n\n");
        com_help(wl);
        return;
    }
    if (!(npath = cp_tildexpand(path))) {
        fprintf(cp_err, "Note: can't find help dir %s\n", path);
        fprintf(cp_err, "Defaulting to old help.\n\n");
        com_help(wl);
        return;
    }
    path = npath;
    if (cp_getvar("helpregfont", VT_STRING, buf))
        hlp_regfontname = copy(buf);
    if (cp_getvar("helpboldfont", VT_STRING, buf))
        hlp_boldfontname = copy(buf);
    if (cp_getvar("helpitalicfont", VT_STRING, buf))
        hlp_italicfontname = copy(buf);
    if (cp_getvar("helptitlefont", VT_STRING, buf))
        hlp_titlefontname = copy(buf);
    if (cp_getvar("helpbuttonfont", VT_STRING, buf))
        hlp_buttonfontname = copy(buf);
    if (cp_getvar("helpinitxpos", VT_NUM, (char *) &i))
        hlp_initxpos = i;
    if (cp_getvar("helpinitypos", VT_NUM, (char *) &i))
        hlp_initypos = i;
    if (cp_getvar("helpbuttonstyle", VT_STRING, buf)) {
        if (cieq(buf, "left"))
            hlp_buttonstyle = BS_LEFT;
        else if (cieq(buf, "center"))
            hlp_buttonstyle = BS_CENTER;
        else if (cieq(buf, "unif"))
            hlp_buttonstyle = BS_UNIF;
        else
            fprintf(cp_err, "Warning: no such button style %s\n",
                    buf);
    }
    if (cp_getvar("width", VT_NUM, (char *) &i))
        hlp_width = i;
    if (cp_getvar("display", VT_STRING, buf))
        hlp_displayname = copy(buf);
    else if (cp_getvar("device", VT_STRING, buf))
        hlp_displayname = copy(buf);
    else
        hlp_displayname = NULL;
    hlp_main(path, wl);
    return;
}
