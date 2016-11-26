/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1988 Jeffrey M. Hsu
**********/

/*
  Postscript driver
*/

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/graph.h"
#include "ngspice/ftedbgra.h"
#include "ngspice/ftedev.h"
#include "ngspice/fteinput.h"
#include "ngspice/fteext.h"

#include "postsc.h"
#include "variable.h"
#include "plotting/graphdb.h"

#define RAD_TO_DEG      (180.0 / M_PI)
#define DEVDEP(g) (*((PSdevdep *) (g)->devdep))
#define MAX_PS_LINES    1000
#define SOLID 0
#define DOTTED 1

#define gtype         graph->grid.gridtype
#define xoff          dispdev->minx
#define yoff          dispdev->miny
#define XOFF          48      /* printer left margin */
#define YOFF          48      /* printer bottom margin */
#define XTADJ         0       /* printer text adjustment x */
#define YTADJ         4       /* printer text adjustment y */

#define GRIDSIZE      420     /* printer gridsize divisible by 10, [7-2] */
#define GRIDSIZES     360     /* printer gridsize divisible by [10-8], [6-2] */

#define FONTSIZE      10      /* printer default fontsize */
#define FONTWIDTH     6       /* printer default fontwidth */
#define FONTHEIGHT    14      /* printer default fontheight */

typedef struct {
    int lastlinestyle, lastcolor; /* initial invalid value */
    int lastx, lasty, linecount;
} PSdevdep;


static char *linestyle[] = {
    "[]",           /* solid */
    "[1 2]",        /* dotted */
    "[7 7]",        /* longdashed */
    "[3 3]",        /* shortdashed */
    "[7 2 2 2]",        /* longdotdashed */
    "[3 2 1 2]",    /* shortdotdashed */
    "[8 3 2 3]",
    "[14 2]",
    "[3 5 1 5]"     /* dotdashed */
};

static FILE *plotfile;
char psfont[128], psfontsize[32], psscale[32], pscolor[32];
static int fontsize   = FONTSIZE;
static int fontwidth  = FONTWIDTH;
static int fontheight = FONTHEIGHT;
static int screenflag = 0;
static int colorflag = 0;
static int setbgcolor = 0;
static int settxcolor = -1;
static double scale;  /* Used for fine tuning */
static int xtadj;     /* text adjustment x */
static int ytadj;     /* text adjustment y */
static int hcopygraphid;


void PS_LinestyleColor(int linestyleid, int colorid);
void PS_SelectColor(int colorid);
void PS_Stroke(void);


/* Set scale, color and size of the plot */
int
PS_Init(void)
{
    char pswidth[30], psheight[30];

    if (!cp_getvar("hcopyscale", CP_STRING, psscale)) {
        scale = 1.0;
    } else {
        sscanf(psscale, "%lf", &scale);
        if ((scale <= 0) || (scale > 10))
            scale = 1.0;
    }
    dispdev->numlinestyles = NUMELEMS(linestyle);
    /* plot color */
    if (!cp_getvar("hcopypscolor", CP_NUM, &setbgcolor)) {
        /* if not set, set plot to b&w and use line styles */
        colorflag = 0;
        dispdev->numcolors = 2;

    } else {
        /* get backgroung color and set plot to color */
        colorflag = 1;
        dispdev->numcolors = 21;   /* don't know what the maximum should be */
        cp_getvar("hcopypstxcolor", CP_NUM, &settxcolor);
    }

    /* plot size */
    if (!cp_getvar("hcopywidth", CP_STRING, pswidth)) {
        dispdev->width = (int)(7.75 * 72.0 * scale);       /* (8 1/2 - 3/4) * 72 */
    } else {
        sscanf(pswidth, "%d", &(dispdev->width));
        if (dispdev->width <= 100)
            dispdev->width = 100;
        if (dispdev->width >= 10000)
            dispdev->width = 10000;
    }
    if (!cp_getvar("hcopyheight", CP_STRING, psheight)) {
        dispdev->height = dispdev->width;
    } else {
        sscanf(psheight, "%d", &(dispdev->height));
        if (dispdev->height <= 100)
            dispdev->height = 100;
        if (dispdev->height >= 10000)
            dispdev->height = 10000;
    }

    /* The following side effects have to be considered
     * when the printer is called by com_hardcopy !
     * gr_init:
     * viewportxoff = 8 * fontwidth
     * viewportyoff = 4 * fontheight
     * gr_resize_internal:
     * viewport.width  = absolute.width - 2 * viewportxoff
     * viewport.height = absolute.height - 2 * viewportyoff
     */

    if (!cp_getvar("hcopyfont", CP_STRING, psfont))
        strcpy(psfont, "Courier");
    if (!cp_getvar("hcopyfontsize", CP_STRING, psfontsize)) {
        fontsize = 10;
        fontwidth = 6;
        fontheight = 14;
        xtadj = (int)(XTADJ * scale);
        ytadj = (int)(YTADJ * scale);
    } else {
        sscanf(psfontsize, "%d", &fontsize);
        if ((fontsize < 10) || (fontsize > 14))
            fontsize = 10;
        fontwidth = (int)(0.5 + 0.6 * fontsize);
        fontheight = (int)(2.5 + 1.2 * fontsize);
        xtadj = (int)(XTADJ * scale * fontsize / 10);
        ytadj = (int)(YTADJ * scale * fontsize / 10);
    }

    screenflag = 0;
    dispdev->minx = (int)(XOFF / scale);
    dispdev->miny = (int)(YOFF / scale);

    return (0);
}


/* Plot and fill bounding box */
int
PS_NewViewport(GRAPH *graph)
{
    int x1, x2, y1, y2;
    hcopygraphid = graph->graphid;
    /* devdep initially contains name of output file */
    if ((plotfile = fopen((char*)graph->devdep, "w")) == NULL) {
        perror((char*)graph->devdep);
        graph->devdep = NULL;
        return (1);
    }

    if (graph->absolute.width) {
        /* hardcopying from the screen */

        screenflag = 1;
    }

    /* reasonable values, used in gr_ for placement */
    graph->fontwidth = (int)(fontwidth * scale); /* was 12, p.w.h. */
    graph->fontheight = (int)(fontheight * scale); /* was 24, p.w.h. */

    graph->absolute.width = dispdev->width;
    graph->absolute.height = dispdev->height;
    /* Also done in gr_init, if called . . . */
    graph->viewportxoff = 8 * fontwidth;
    graph->viewportyoff = 4 * fontheight;

    xoff = (int)(scale * XOFF);
    yoff = (int)(scale * YOFF);

    x1 = (int)(0.75 * 72);
    y1 = x1;
    x2 = (int)(graph->absolute.width + .75 * 72);
    y2 = (int)(graph->absolute.height + .75 * 72);
    /* start file off with a % */
    fprintf(plotfile, "%%!PS-Adobe-3.0 EPSF-3.0\n");
    fprintf(plotfile, "%%%%Creator: nutmeg\n");
    fprintf(plotfile, "%%%%BoundingBox: %d %d %d %d\n", x1, y1, x2, y2);

    /* ReEncoding to allow 'extended asccii'
     * thanks to http://apps.jcns.fz-juelich.de/doku/sc/ps-latin/ */
    fprintf(plotfile, "/ReEncode { %% inFont outFont encoding | -\n");
    fprintf(plotfile, "   /MyEncoding exch def\n");
    fprintf(plotfile, "      exch findfont\n");
    fprintf(plotfile, "      dup length dict\n");
    fprintf(plotfile, "      begin\n");
    fprintf(plotfile, "         {def} forall\n");
    fprintf(plotfile, "         /Encoding MyEncoding def\n");
    fprintf(plotfile, "         currentdict\n");
    fprintf(plotfile, "      end\n");
    fprintf(plotfile, "      definefont\n");
    fprintf(plotfile, "} def\n");
    fprintf(plotfile, "/%s /%sLatin1 ISOLatin1Encoding ReEncode\n", psfont, psfont);

    if (colorflag == 1) {
        /* set the background to color given in spinit (or 0) */
        PS_SelectColor(setbgcolor);
        fprintf(plotfile, "%s setrgbcolor\n", pscolor);
        fprintf(plotfile, "newpath\n");
        fprintf(plotfile, "%d %d moveto %d %d lineto\n", x1, y1, x2, y1);
        fprintf(plotfile, "%d %d lineto %d %d lineto\n", x2, y2, x1, y2);
        fprintf(plotfile, "closepath fill\n");
    }

    /* set up a reasonable font */
    fprintf(plotfile, "/%sLatin1 findfont %d scalefont setfont\n\n",
            psfont, (int) (fontsize * scale));

    graph->devdep = TMALLOC(PSdevdep, 1);
    DEVDEP(graph).lastlinestyle = -1;
    DEVDEP(graph).lastcolor = -1;
    DEVDEP(graph).lastx = -1;
    DEVDEP(graph).lasty = -1;
    DEVDEP(graph).linecount = 0;
    PS_SelectColor(0);
    graph->linestyle = -1;

    return 0;
}


int
PS_Close(void)
{
    /* in case PS_Close is called as part of an abort,
       w/o having reached PS_NewViewport */
    if (plotfile) {
        PS_Stroke();
        fprintf(plotfile, "showpage\n%%%%EOF\n");
        fclose(plotfile);
        plotfile = NULL;
    }
    /* In case of hardcopy command destroy the hardcopy graph
     * and reset currentgraph to graphid 1, if possible
     */
    if (!screenflag) {
        DestroyGraph(hcopygraphid);
        currentgraph = FindGraph(1);
    }
    return 0;
}


int
PS_Clear(void)
{
    /* do nothing */
    return 0;
}


int
PS_DrawLine(int x1, int y1, int x2, int y2)
{
    /* note: this is not extendible to more than one graph
       => will have to give NewViewport a writeable graph XXX */

    if (DEVDEP(currentgraph).linecount > MAX_PS_LINES ||
        DEVDEP(currentgraph).linecount == 0 ||
        x1 != DEVDEP(currentgraph).lastx ||
        y1 != DEVDEP(currentgraph).lasty)
    {
        PS_Stroke();
        fprintf(plotfile, "newpath\n");
        fprintf(plotfile, "%d %d moveto\n", x1 + xoff, y1 + yoff);
        DEVDEP(currentgraph).linecount += 1;
    }

    if (x1 != x2 || y1 != y2) {
        fprintf(plotfile, "%d %d lineto\n", x2 + xoff, y2 + yoff);
        DEVDEP(currentgraph).linecount += 1;
    }

    DEVDEP(currentgraph).lastx = x2;
    DEVDEP(currentgraph).lasty = y2;
    return 0;
}


int
PS_Arc(int x0, int y0, int r, double theta, double delta_theta)
{
    double x1, y1;
    double angle1, angle2;
    PS_Stroke();

    angle1 = (double) (RAD_TO_DEG * theta);
    angle2 = (double) (RAD_TO_DEG * (theta + delta_theta));
    x1 = (double) x0 + r * cos(theta);
    y1 = (double) y0 + r * sin(theta);

    fprintf(plotfile, "%f %f moveto ", x1+(double)xoff, y1+(double)yoff);
    fprintf(plotfile, "%d %d %d %f %f arc\n", x0+xoff, y0+yoff, r,
            angle1, angle2);
    fprintf(plotfile, "stroke\n");

    DEVDEP(currentgraph).linecount = 0;
    return 0;
}


int
PS_Text(char *text, int x, int y)
{
    int savedlstyle, savedcolor;

    /* set linestyle to solid
       or may get funny color text on some plotters */
    savedlstyle = currentgraph->linestyle;
    savedcolor = currentgraph->currentcolor;

    PS_SetLinestyle(SOLID);
    /* set text color to black if background is not white */
    if (setbgcolor > 0)
        PS_SetColor(0);
    else
        PS_SetColor(1);
    /* if color is given by set hcopytxpscolor=settxcolor, give it a try */
    if (settxcolor >= 0)
        PS_SetColor(settxcolor);
    /* stroke the path if there's an open one */
    PS_Stroke();
    /* move to (x, y) */
    fprintf(plotfile, "%d %d moveto\n", x + xoff + xtadj, y + yoff + ytadj);
    fprintf(plotfile, "(%s) show\n", text);

    DEVDEP(currentgraph).lastx = -1;
    DEVDEP(currentgraph).lasty = -1;

    /* restore old linestyle */

    PS_SetColor(savedcolor);
    PS_SetLinestyle(savedlstyle);
    return 0;
}


/* PS_DefineColor */
/* PS_DefineLinestyle */

int
PS_SetLinestyle(int linestyleid)
{
    /* special case
       get it when PS_Text restores a -1 linestyle */
    if (linestyleid == -1) {
        currentgraph->linestyle = -1;
        return 0;
    }
    if (linestyleid < 0 || linestyleid > dispdev->numlinestyles) {
        internalerror("bad linestyleid inside PS_SetLinestyle");
        fprintf(cp_err, "linestyleid is: %d\n", linestyleid);
        return 0;
    }
    PS_LinestyleColor(linestyleid, currentgraph->currentcolor);
    return 0;
}


int
PS_SetColor(int colorid)
{
    PS_LinestyleColor(currentgraph->linestyle, colorid);
    return 0;
}


int
PS_Update(void)
{
    fflush(plotfile);
    return 0;
}


/**************** PRIVAT FUNCTIONS OF PS FRONTEND *****************************/

void
PS_SelectColor(int colorid)           /* should be replaced by PS_DefineColor */
{
    char colorN[30] = "", colorstring[30] = "";
    char rgb[30], s_red[30] = "0x", s_green[30] = "0x", s_blue[30] = "0x";
    int  red = 0, green = 0, blue = 0, maxval = 1;
    int i;
    typedef struct { int red, green, blue;} COLOR;
    /* duplicated colors from src/frontend/plotting/x11.c in rgb-style */
    const COLOR colors[] = {{  0,   0,   0},    /*0: black */
                            {255, 255, 255},    /*1: white */
                            {255,   0,   0},    /*2: red */
                            {  0,   0, 255},    /*3: blue */
                            {255, 165,   0},    /*4: orange */
                            {  0, 255,   0},    /*5: green */
                            {255, 192, 203},    /*6: pink */
                            {165,  42,  42},    /*7: brown */
                            {240, 230, 140},    /*8: khaki */
                            {221, 160, 221},    /*9: plum */
                            {218, 112, 214},    /*10: orchid */
                            {238, 130, 238},    /*11: violet */
                            {176,  48,  96},    /*12: maroon */
                            { 64, 224, 208},    /*13: turqoise */
                            {160,  82,  45},    /*14: sienna */
                            {255, 127,  80},    /*15: coral */
                            {  0, 255, 255},    /*16: cyan */
                            {255,   0, 255},    /*17: magenta */
                            /*{255, 215,   0},    18: gold */
                            { 96,  96,  96},    /*18: gray for smith grid */
                            /*{255, 255,   0},    19: yello */
                            {150, 150, 150},    /*19: gray for smith grid */
                            {128, 128, 128}};   /*20: gray for normal grid */

    /* Extract the rgbcolor, format is: "rgb:<red>/<green>/<blue>" */
    sprintf(colorN, "color%d", colorid);
    if (cp_getvar(colorN, CP_STRING, colorstring)) {
        for (i = 0; colorstring[i]; i++)
            if (colorstring[i] == '/' || colorstring[i] == ':')
                colorstring[i] = ' ';

        sscanf(colorstring, "%s %s %s %s", rgb, &(s_red[2]), &(s_green[2]), &(s_blue[2]));

        if ((strlen(s_blue) == strlen(s_red) && strlen(s_green) == strlen(s_red))
            && (strlen(s_blue) > 2) && (strlen(s_blue) < 7)) {
            sscanf(s_red, "%x", &red);
            sscanf(s_green, "%x", &green);
            sscanf(s_blue, "%x", &blue);
            maxval = (1 << (strlen(s_blue) - 2) * 4) - 1;
            sprintf(colorstring, "%1.3f %1.3f %1.3f",
                    (double) red/maxval, (double) green/maxval, (double) blue/maxval);
            strcpy(pscolor, colorstring);
        }
    }
    if (colorid < 0 || colorid > 20) {
        internalerror("bad colorid inside PS_SelectColor");
    } else if (maxval == 1) {  /* colorN is not an rgbstring, use default color */
        sprintf(colorstring, "%1.3f %1.3f %1.3f", colors[colorid].red/255.0,
                colors[colorid].green/255.0, colors[colorid].blue/255.0);
        strcpy(pscolor, colorstring);
    }
}


void
PS_LinestyleColor(int linestyleid, int colorid)
{
    /* we have some different linestyles and colors:
       - color and linestyle we got via function call
       - color and linestyle we used last time for drawing
       - generated color and linestyle we'll use for drawing this time */
    /* these are the rules:
       DOTTED and colored ps -> color20 (used for grid) and SOLID
       color18 or 19 and black-white -> linestyle is DOTTED */

    int gencolor = 0, genstyle = 0;

    if (colorflag == 1) {
        genstyle = SOLID;
        if (linestyleid == DOTTED)
            gencolor = 20;
        else
            gencolor = colorid;
    } else { /* colorflag == 0 -> mono*/
        if ((colorid == 18) || (colorid == 19))
            genstyle = DOTTED;
        else if (linestyleid == -1)
            genstyle = 0;
        else
            genstyle = linestyleid;
    }

    /* change color if nessecary */
    if (colorflag == 1 && gencolor != DEVDEP(currentgraph).lastcolor) {
        /* if background is white, set all white line colors to black */
        if ((setbgcolor == 1) && (gencolor == 1))
            PS_SelectColor(0);
        else
            PS_SelectColor(gencolor);
        PS_Stroke();
        fprintf(plotfile, "%s setrgbcolor\n", pscolor);
        DEVDEP(currentgraph).lastcolor = gencolor;
    }
    currentgraph->currentcolor = colorid;

    /* change linestyle if nessecary */
    if (colorflag == 0 && genstyle != DEVDEP(currentgraph).lastlinestyle) {
        PS_Stroke();
        fprintf(plotfile, "%s 0 setdash\n", linestyle[genstyle]);
        DEVDEP(currentgraph).lastlinestyle = genstyle;
    }
    currentgraph->linestyle = linestyleid;
}


void
PS_Stroke(void)
{
    /* strokes an open path */
    if (DEVDEP(currentgraph).linecount > 0) {
        fprintf(plotfile, "stroke\n");
        DEVDEP(currentgraph).linecount = 0;
    }
}
