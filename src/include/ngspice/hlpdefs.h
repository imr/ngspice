/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1986 Wayne A. Christopher, U. C. Berkeley CAD Group 
Modified 1999 Emmanuel Rouat - 2000  AlansFixes
**********/

/*
 * Definitions for the help system.
 */
#include "ngspice/config.h"



#ifndef X_DISPLAY_MISSING
#  include <X11/Intrinsic.h>
#endif

typedef struct fplace {
    char *filename;
    long fpos;
    FILE *fp;
} fplace;

typedef struct button {
    char *text;
    fplace *tag;        /* Why isn't used for anything? */
    int x;
    int y;
    int width;
    int height;
} button;

struct hlp_index {
    char subject[64];
    long fpos;
};

typedef struct toplink {
    char *description;          /* really the subject */
    fplace *place;
    struct toplink *next;
    struct button button;
} toplink;

typedef struct topic {
    char *subject;
    char *title;
    fplace *place;
    wordlist *text;
    char *chartext;
    toplink *subtopics;
    toplink *seealso;
    int xposition;
    int yposition;
    struct topic *parent;
    struct topic *children;
    struct topic *next;
    struct topic *winlink;
    struct topic *readlink;
    int numlines;
    int maxcols;
    int curtopline;

#ifndef X_DISPLAY_MISSING
    Widget shellwidget, formwidget, titlewidget, buttonwidget,
        textwidget, seelabelwidget, sublabelwidget, seeboxwidget, subboxwidget;
#endif
} topic;

typedef struct handle {
    topic *parent;
    toplink *result;
} handle;

#define REG_FONT        "timrom12"
#define BOLD_FONT       "timrom12b"
#define ITALIC_FONT     "timrom12i"
#define TITLE_FONT      "accordb"
#define BUTTON_FONT     "6x10"

#define X_INCR          20
#define Y_INCR          20
#define BORDER_WIDTH    3
#define INT_BORDER      10
#define BUTTON_XPAD 4
#define BUTTON_YPAD 2

#define START_XPOS  100
#define START_YPOS  100

/* If the MAX_LINES and SCROLL_INCR are different, it is very confusing... */

#define MIN_COLS    40
#define MAX_COLS    90

#define MAX_LINES   25
#define SCROLL_INCR 25

enum {
    BS_LEFT = 0,
    BS_CENTER,
    BS_UNIF,
};

/* External symbols. */

/* help.c */

extern char *hlp_directory;

extern void hlp_main(char *path, wordlist *wl);
extern FILE *hlp_fopen(char *filename);
extern fplace *findglobalsubject(char *subject);
extern bool hlp_approvedfile(char *filename);
extern void hlp_pathfix(char *buf);


/* readhelp.c */

extern topic *hlp_read(fplace *place);
extern void hlp_free(void);
extern long findsubject(char *filename, char *subject);

/* provide.c */

extern void hlp_provide(topic *top);
extern bool hlp_usex;
extern void hlp_fixchildren(topic *parent);
extern void hlp_killfamily(topic *top);

/* xdisplay.c */

extern char *hlp_regfontname;
extern char *hlp_boldfontname;
extern char *hlp_italicfontname;
extern char *hlp_titlefontname;
extern char *hlp_buttonfontname;
extern int hlp_initxpos;
extern int hlp_initypos;
extern int hlp_buttonstyle;
extern char *hlp_displayname;
extern bool hlp_xdisplay(topic *top);
extern void hlp_xclosedisplay(void);
extern toplink *hlp_xhandle(topic **pp);
extern void hlp_xkillwin(topic *top);
extern void hlp_xwait(topic *top, bool on);


/* textdisp.c */

extern bool hlp_tdisplay(topic *top);
extern toplink *hlp_thandle(topic **parent);
extern void hlp_tkillwin(topic *top);
extern int hlp_width;

