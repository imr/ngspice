/**********
Author: Jim Groves
**********/

/*
    HPGL driver
*/

/*
   1000 plotter units / inch  - 1pu = 0.025mm  1pu = 1mil

  SP - select pen
  PU - pen up (PU x,y)
  PD - pen down (PD x,y)
  LT - line type
       0   dots only at plotted points
       1   .   .    .   .    .
       2   ___     ___     ___     ___
       3   ----    ----    ----    ----
       4   ----- . ----- . ----- . -----.
       5   ---- -  ---- -  ---- -
       6   --- - - --- - - --- - - --- - -
       null - solid line
  IN - initialize
  DF - default values (PA, solid line, set 0)
  PA - plot absolute
  SI - absolute character size (SI width, height) in cm

*/

#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/graph.h"
#include "ngspice/ftedbgra.h"
#include "ngspice/ftedev.h"
#include "ngspice/fteinput.h"
#include "ngspice/fteext.h"
#include "variable.h"
#include "plotting/graphdb.h"
#include "hpgl.h"

#define RAD_TO_DEG      (180.0 / M_PI)
#define DEVDEP(g) (*((GLdevdep *) (g)->devdep))
#define MAX_GL_LINES    9999
#define SOLID 0
#define DOTTED 1

#define gtype         graph->grid.gridtype
#define xoff          dispdev->minx
#define yoff          dispdev->miny
#define XOFF          25      /* printer left margin */
#define YOFF          28      /* printer bottom margin */
#define XTADJ         0       /* printer text adjustment x */
#define YTADJ         0       /* printer text adjustment y */

#define DELXMAX       360     /* printer gridsize divisible by 10, [7-2] */
#define DELYMAX       360     /* printer gridsize divisible by [10-8], [6-2] */

#define FONTWIDTH     6       /* printer default fontwidth */
#define FONTHEIGHT    8       /* printer default fontheight */

typedef struct {
    int lastlinestyle;      /* initial invalid value */
    int lastx, lasty, linecount;
} GLdevdep;

static char *linestyle[] = {
    "",     /* solid */
    "1",    /* was 1 - dotted */
    "",     /* longdashed */
    "3",    /* shortdashed */
    "4",    /* longdotdashed */
    "5",    /* shortdotdashed */
    "1"
};

static FILE *plotfile;
extern char psscale[32];
static int fontwidth  = FONTWIDTH;
static int fontheight = FONTHEIGHT;
static int   jgmult      = 10;
static int screenflag = 0;
static double tocm = 0.0025;
static double scale;    /* Used for fine tuning */
static int hcopygraphid;


int GL_Init(void)
{
    if (!cp_getvar("hcopyscale", CP_STRING, psscale, sizeof(psscale))) {
        scale = 1.0;
    }
    else {
        sscanf(psscale, "%lf", &scale);
        if ((scale <= 0) || (scale > 10))
            scale = 1.0;
    }

    dispdev->numlinestyles = NUMELEMS(linestyle);
    dispdev->numcolors = 6;

    dispdev->width = (int)(DELXMAX * scale);
    dispdev->height = (int)(DELYMAX * scale);


    screenflag = 0;
    dispdev->minx = (int)(XOFF * 1.0);
    dispdev->miny = (int)(YOFF * 1.0);

    return 0;
}


/* devdep initially contains name of output file */
int GL_NewViewport(GRAPH *graph)
{
    hcopygraphid = graph->graphid;

    if ((plotfile = fopen((char*) graph->devdep, "w")) == NULL) {
        perror((char *) graph->devdep);
        free(graph->devdep);
        graph->devdep = NULL;
        graph->n_byte_devdep = 0;
        return 1;
    }

    if (graph->absolute.width) {
        /* hardcopying from the screen */

        screenflag = 1;

        /* scale to fit on 8 1/2 square */

    }

    /* reasonable values, used in gr_ for placement */
    graph->fontwidth = (int)(fontwidth * scale); /* was 12, p.w.h. */
    graph->fontheight = (int)(fontheight * scale); /* was 24, p.w.h. */

    graph->absolute.width = dispdev->width;
    graph->absolute.height = dispdev->height;
    /* Also done in gr_init, if called . . . */
    graph->viewportxoff = 16 * fontwidth;
    graph->viewportyoff = 8 * fontheight;

    xoff = XOFF;
    yoff = YOFF;

    /* start file off with a % */
    fprintf(plotfile, "IN;DF;PA;");
    fprintf(plotfile, "SI %f,%f;",
            tocm * jgmult * fontwidth * scale,
            tocm * jgmult * fontheight * scale);

#ifdef notdef
    if (!screenflag)
#endif
    {
        graph->devdep = TMALLOC(GLdevdep, 1);
        graph->n_byte_devdep = sizeof(GLdevdep);
    }

    DEVDEP(graph).lastlinestyle = -1;
    DEVDEP(graph).lastx = -1;
    DEVDEP(graph).lasty = -1;
    DEVDEP(graph).linecount = 0;
    graph->linestyle = -1;

    return 0;
}


int GL_Close(void)
{
    /* in case GL_Close is called as part of an abort,
       w/o having reached GL_NewViewport */
    if (plotfile) {
        if (DEVDEP(currentgraph).lastlinestyle != -1) {
            DEVDEP(currentgraph).linecount = 0;
        }
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


int GL_Clear(void)
{
    /* do nothing */

    return 0;
}


int
GL_DrawLine(int x1, int y1, int x2, int y2, bool isgrid)
{
    NG_IGNORE(isgrid);
    /* note: this is not extendible to more than one graph
       => will have to give NewViewport a writeable graph XXX */


    if (DEVDEP(currentgraph).linecount == 0
            || x1 != DEVDEP(currentgraph).lastx
            || y1 != DEVDEP(currentgraph).lasty) {
        fprintf(plotfile, "PU;PA %d , %d ;",
                jgmult * (x1 + xoff), jgmult * (y1 + yoff));
    }
    if (x1 != x2 || y1 != y2) {
        fprintf(plotfile, "PD;PA %d , %d ;",
                jgmult * (x2 + xoff), jgmult * (y2 + yoff));
        DEVDEP(currentgraph).linecount += 1;
    }

    DEVDEP(currentgraph).lastx = x2;
    DEVDEP(currentgraph).lasty = y2;
    DEVDEP(currentgraph).lastlinestyle = currentgraph->linestyle;

    return 0;
}


/* ARGSUSED */
int GL_Arc(int x0, int y0, int r, double theta, double delta_theta, bool isgrid)
{
    NG_IGNORE(isgrid);
    
    int  x1, y1, angle;

    x1 = x0 + (int)(r * cos(theta));
    y1 = y0 + (int)(r * sin(theta));

    angle = (int)(RAD_TO_DEG * delta_theta);

    fprintf(plotfile, "PU;PA %d , %d;",
            jgmult * (x1 + xoff + XTADJ), jgmult * (y1 + yoff + YTADJ));
    fprintf(plotfile, "PD;AA %d , %d, %d;",
            jgmult * (x0 + xoff + XTADJ), jgmult*(y0 + yoff + YTADJ), angle);

    DEVDEP(currentgraph).linecount = 0;

    return 0;
}


int GL_Text(const char *text, int x, int y, int angle)
{
    NG_IGNORE(angle);

    /* move to (x, y) */
    NG_IGNORE(angle);

    fprintf(plotfile, "PU;PA %d , %d;",
            jgmult * (x + xoff + XTADJ), jgmult * (y + yoff + YTADJ));
    fprintf(plotfile, "LB %s \x03", text);

    DEVDEP(currentgraph).lastx = -1;
    DEVDEP(currentgraph).lasty = -1;

    return 0;
}


int GL_SetLinestyle(int linestyleid)
{
    /* special case
       get it when GL_Text restores a -1 linestyle */
    if (linestyleid == -1) {
        currentgraph->linestyle = -1;
        return 0;
    }

    if (linestyleid < 0 || linestyleid > dispdev->numlinestyles) {
        internalerror("bad linestyleid");
        return 0;
    }

    if (currentgraph->linestyle != linestyleid) {
        fprintf(plotfile, "LT %s ;", linestyle[linestyleid]);
        currentgraph->linestyle = linestyleid;
    }

    return 0;
}


int GL_SetColor(int colorid)
{
    fprintf(plotfile, "SP %d;", colorid);

    return 0;
}


int GL_Update(void)
{
    fflush(plotfile);

    return 0;
}
