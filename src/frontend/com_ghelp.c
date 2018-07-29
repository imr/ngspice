#include "ngspice/ngspice.h"
#include "ngspice/wordlist.h"
#include "ngspice/bool.h"

#include "variable.h"
#include "ngspice/cpextern.h"
#include "ngspice/cpextern.h"
#include "ngspice/hlpdefs.h"

#include "com_ghelp.h"
#include "com_help.h"

void
com_ghelp(wordlist *wl)
{
#if defined(HAS_WINGUI) || defined(_MSC_VER) || defined(__MINGW32__) || defined(X_DISPLAY_MISSING) || defined(NOINTHELP)

    NG_IGNORE(wl);

    printf("Internal help is no longer avaialable!\n"
           "Please check for the actual ngspice manual at\n"
           "  http://ngspice.sourceforge.net/docs/ngspice-manual.pdf\n"
           "or for help on spice3 at\n"
           "  http://newton.ex.ac.uk/teaching/CDHW/Electronics2/userguide/\n");
    return;

#else

    char *npath;
    char *path = Help_Path;
    char buf[BSIZE_SP];

#ifndef X_DISPLAY_MISSING
    int i;
#endif /* X_DISPLAY_MISSING 1  */

    if (cp_getvar("helppath", CP_STRING, buf, sizeof(buf)))
        path = copy(buf);
    if (!path) {
        fprintf(cp_err, "Note: defaulting to old help.\n\n");
        com_help(wl);
        return;
    }
    if ((npath = cp_tildexpand(path)) == NULL) {
        fprintf(cp_err, "Note: can't find help dir %s\n", path);
        fprintf(cp_err, "Defaulting to old help.\n\n");
        com_help(wl);
        return;
    }

#ifndef X_DISPLAY_MISSING /* 1 */
    path = npath;
    if (cp_getvar("helpregfont", CP_STRING, buf, sizeof(buf)))
        hlp_regfontname = copy(buf);
    if (cp_getvar("helpboldfont", CP_STRING, buf, sizeof(buf)))
        hlp_boldfontname = copy(buf);
    if (cp_getvar("helpitalicfont", CP_STRING, buf, sizeof(buf)))
        hlp_italicfontname = copy(buf);
    if (cp_getvar("helptitlefont", CP_STRING, buf, sizeof(buf)))
        hlp_titlefontname = copy(buf);
    if (cp_getvar("helpbuttonfont", CP_STRING, buf, sizeof(buf)))
        hlp_buttonfontname = copy(buf);
    if (cp_getvar("helpinitxpos", CP_NUM, &i, 0))
        hlp_initxpos = i;
    if (cp_getvar("helpinitypos", CP_NUM, &i, 0))
        hlp_initypos = i;
    if (cp_getvar("helpbuttonstyle", CP_STRING, buf, sizeof(buf))) {
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
    if (cp_getvar("width", CP_NUM, &i, 0))
        hlp_width = i;
    if (cp_getvar("display", CP_STRING, buf, sizeof(buf)))
        hlp_displayname = copy(buf);
    else if (cp_getvar("device", CP_STRING, buf, sizeof(buf)))
        hlp_displayname = copy(buf);
    else
        hlp_displayname = NULL;
    hlp_main(path, wl);
    return;
#else
    com_help(wl);
#endif /* X_DISPLAY_MISSING 1  */

#endif
}
