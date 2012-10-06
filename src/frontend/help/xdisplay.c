/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1986 Wayne A. Christopher, U. C. Berkeley CAD Group
Modified 1999 Emmanuel Rouat
**********/

#include "ngspice/ngspice.h"
#include "ngspice/cpstd.h"
#include "ngspice/hlpdefs.h"
#include "ngspice/suffix.h"


char *hlp_boldfontname = BOLD_FONT;
char *hlp_regfontname = REG_FONT;
char *hlp_italicfontname = ITALIC_FONT;
char *hlp_titlefontname = TITLE_FONT;
char *hlp_buttonfontname = BUTTON_FONT;
char *hlp_displayname = NULL;
int hlp_initxpos = START_XPOS;
int hlp_initypos = START_YPOS;
int hlp_buttonstyle = BS_LEFT;


#ifdef X_DISPLAY_MISSING

bool
hlp_xdisplay(topic *top)
{
    NG_IGNORE(top);
    return (FALSE);
}


void
hlp_xkillwin(topic *top)
{
    NG_IGNORE(top);
}

#endif


void
hlp_xwait(topic *top, bool on)
{
    NG_IGNORE(on);
    NG_IGNORE(top);
}


void
hlp_xclosedisplay(void)
{
}


toplink *
hlp_xhandle(topic **pp)
{
    *pp = NULL;
    return (NULL);
}
