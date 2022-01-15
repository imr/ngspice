/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Copyright 2020 Giles Atkinson
Original author (postsc.c): 1988 Jeffrey M. Hsu
**********/

/* SVG driver. */

#include "ngspice/ngspice.h"
#include "ngspice/graph.h"
#include "ngspice/ftedbgra.h"
#include "ngspice/ftedev.h"
#include "ngspice/fteext.h"

#include "svg.h"
#include "plotting/graphdb.h"
#include "variable.h"

/* String and int options, if changed, change SVGxxxxx  macros below. */

#define SVG_WIDTH        0
#define SVG_HEIGHT       1
#define SVG_FONT_SIZE    2
#define SVG_FONT_WIDTH   3
#define SVG_USE_COLOR    4      /* 0 = no 1 = yes 2 = yes, with dashes */
#define SVG_STROKE_WIDTH 5
#define SVG_GRID_WIDTH   6

#define NUM_INTS 7

#define SVG_BACKGROUND  0
#define SVG_FONT_FAMILY 1
#define SVG_FONT        2

#define NUM_STRINGS     3

static struct {
    int   ints[NUM_INTS];
    char *strings[NUM_STRINGS];
} Cfg = {{1024, 768, 16, 0, 1, 2, 0}, {NULL,}};

/* Macros to examine configuration options. */

#define SVGwidth (Cfg.ints[SVG_WIDTH])
#define SVGheight (Cfg.ints[SVG_HEIGHT])
#define SVGfont_size (Cfg.ints[SVG_FONT_SIZE])
#define SVGfont_width (Cfg.ints[SVG_FONT_WIDTH])
#define SVGuse_color (Cfg.ints[SVG_USE_COLOR])
#define SVGstroke_width (Cfg.ints[SVG_STROKE_WIDTH])
#define SVGgrid_width (Cfg.ints[SVG_GRID_WIDTH])

#define SVGbackground (Cfg.strings[SVG_BACKGROUND])
#define SVGfont_family (Cfg.strings[SVG_FONT_FAMILY])
#define SVGfont (Cfg.strings[SVG_FONT])

/* svg_intopts are:
  "svgwidth", "svgheight",  "svgfont-size", "svgfont-width", "svguse-color",
  "svgstroke-width", "svggrid-width"

  svg_stropts are:
    "svgbackground", "svgfont-family", "svgfont"
*/

typedef struct {
    int  lastx, lasty;
    int  inpath;
    int  linelen;
    bool isgrid;
} SVGdevdep;

#define DEVDEP_P(g) ((SVGdevdep *)(g)->devdep)

/* Values for DEVDEP_P(g)->inpath. */

#define NOPATH 0
#define PATH   1
#define LINE   2

#define CHECK_PATH \
    if (ddp->inpath == NOPATH || ddp->linelen > 240) startpath(ddp)

static void closepath(SVGdevdep *ddp), startpath(SVGdevdep *ddp);
static void startpath_width(SVGdevdep *ddp, int width);

/* Values for 'stroke-dasharray'. */

static char *linestyles[] = {
    NULL,           /* Solid */
    "1,2",          /* Dotted */
    "7,7",          /* Long dashes */
    "3,3",          /* Short dashes */
    "7,2,2,2",      /* Long/dot dashes */
    "3,2,1,2",      /* Short/dot dashes */
    "8,3,2,3",
    "14,2",
    "3,5,1,5"       /* Short/dot, longp dashes */
};

static FILE *plotfile;
static int screenflag = 0;
static int hcopygraphid;

const char *svgcolors[] = {"black",
                        "white",
                        "red",
                        "blue",
                        "#FFA500",    /*4: orange */
                        "green",
                        "#FFC0C5",    /*6: pink */
                        "#A52A2A",    /*7: brown */
                        "#F0E68C",    /*8: khaki */
                        "#DDA0DD",    /*9: plum */
                        "#DA70D6",    /*10: orchid */
                        "#EE82EE",    /*11: violet */
                        "#B03060",    /*12: maroon */
                        "#40E0D0",    /*13: turqoise */
                        "#A0522D",    /*14: sienna */
                        "#FF7F50",    /*15: coral */
                        "cyan",
                        "magenta",
                        "#666",       /*18: gray for smith grid */
                        "#949494",    /*19: gray for smith grid */
                        "#888"};      /*20: gray for normal grid */

static char** colors;

/* Set scale, color and size of the plot */
int
SVG_Init(void)
{
    char          colorN[16], colorstring[30], strbuf[512];
    unsigned int  colorid, i;
    struct variable *vb, *va;
    bool intopts_isset = FALSE, stropts_isset = FALSE;
    unsigned int numberofcolors = NUMELEMS(svgcolors);

    /* set intopts and stropts as basic data */
    if (cp_getvar("svg_intopts", CP_LIST, &va, 0)) {
        i = 0;
        while (i < NUM_INTS && va) {
            Cfg.ints[i++] = va->va_num;
            va = va->va_next;
        }
        intopts_isset = TRUE;
    }
    if (cp_getvar("svg_stropts", CP_LIST, &vb, 0)) {
        i = 0;
        while (i < NUM_STRINGS && vb) {
            tfree(Cfg.strings[i]);
            Cfg.strings[i++] = strdup(vb->va_string);
            vb = vb->va_next;
        }
        stropts_isset = TRUE;
    }

    /* plot size */
    cp_getvar("hcopywidth", CP_NUM, &SVGwidth, 0);
    dispdev->width = SVGwidth;
    cp_getvar("hcopyheight", CP_NUM, &SVGheight, 0);
    dispdev->height = SVGheight;

    /* get linewidth information from spinit */
    if (!cp_getvar("xbrushwidth", CP_NUM, &Cfg.ints[SVG_STROKE_WIDTH], 0))
        Cfg.ints[SVG_STROKE_WIDTH] = 0;
    if (Cfg.ints[SVG_STROKE_WIDTH] < 0)
        Cfg.ints[SVG_STROKE_WIDTH] = 0;

    /* get linewidth for grid from spinit */
    if (!cp_getvar("xgridwidth", CP_NUM, &Cfg.ints[SVG_GRID_WIDTH], 0))
        Cfg.ints[SVG_GRID_WIDTH] = Cfg.ints[SVG_STROKE_WIDTH];
    if (Cfg.ints[SVG_GRID_WIDTH] < 0)
        Cfg.ints[SVG_GRID_WIDTH] = 0;

    if (cp_getvar("hcopyfont", CP_STRING, &strbuf, sizeof(strbuf))) {
        tfree(Cfg.strings[SVG_FONT]);
        Cfg.strings[SVG_FONT] = strdup(strbuf);
    }
    else if (!stropts_isset) {
        tfree(Cfg.strings[SVG_FONT]);
        Cfg.strings[SVG_FONT] = strdup("Helvetica");
    }

    if (cp_getvar("hcopyfontfamily", CP_STRING, &strbuf, sizeof(strbuf))) {
        tfree(Cfg.strings[SVG_FONT_FAMILY]);
        Cfg.strings[SVG_FONT_FAMILY] = strdup(strbuf);
    }
    else if (!stropts_isset){
        tfree(Cfg.strings[SVG_FONT_FAMILY]);
        Cfg.strings[SVG_FONT_FAMILY] = strdup("Helvetica");
    }

    cp_getvar("hcopyfontsize", CP_NUM, &Cfg.ints[SVG_FONT_SIZE], 0);

    colors = TMALLOC(char*, numberofcolors);

    /* Look for colour overrides: HTML/X11 #xxxxxx style. */
    for (colorid = 0; colorid < numberofcolors; ++colorid) {
        sprintf(colorN, "color%d", colorid);
        if (cp_getvar(colorN, CP_STRING, colorstring, sizeof(colorstring))) {
            colors[colorid] = strdup(colorstring);
            if (colorid == 0) {
                tfree(Cfg.strings[SVG_BACKGROUND]);
                Cfg.strings[SVG_BACKGROUND] = strdup(colors[0]);
            }
        }
        else {
            colors[colorid] = strdup(svgcolors[colorid]);
        }
    }

    /* Get other options. */
    if (SVGgrid_width == 0)
        SVGgrid_width = (2 * SVGstroke_width) / 3;

    if (SVGuse_color == 0) {
        /* Black and white. */
        dispdev->numcolors = 2;
    } else {
        dispdev->numcolors = NUMELEMS(svgcolors);
    }

    if (SVGuse_color == 1) {
        /* Suppress dashes in favour of color. */
        dispdev->numlinestyles = 2;
    } else {
        dispdev->numlinestyles = NUMELEMS(linestyles);
    }

    dispdev->minx = 0;
    dispdev->miny = 0;
    return 0;
}

/* Plot and fill bounding box */
int
SVG_NewViewport(GRAPH *graph)
{
    SVGdevdep *ddp;

    hcopygraphid = graph->graphid;
    if (graph->absolute.width) {
        /* hardcopying from the screen */

        screenflag = 1;
    }

    graph->absolute.width = dispdev->width;
    graph->absolute.height = dispdev->height;
    if (SVGfont_width)
        graph->fontwidth = SVGfont_width;
    else
        graph->fontwidth = (2 * SVGfont_size) / 3;  // Ugly!
    graph->fontheight = SVGfont_size;

    /* Start file off.
     * devdep initially contains name of output file. 
     */

    if ((plotfile = fopen((char*)graph->devdep, "w")) == NULL) {
        perror((char*)graph->devdep);
        graph->devdep = NULL;
        return 1;
    }

    fputs("<?xml version=\"1.0\" standalone=\"yes\"?>\n", plotfile);
    fputs("<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\"\n"
          " \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n",
          plotfile);
    fputs("<svg xmlns=\"http://www.w3.org/2000/svg\" version=\"1.1\"\n",
          plotfile);
    fprintf(plotfile,
            "  width=\"100%%\" height=\"100%%\" viewBox=\"0 0 %d %d\"\n",
            dispdev->width, dispdev->height);

    /* Style paths. */

    fputs("  style=\"fill: none;", plotfile);
    if (SVGstroke_width > 0) {
        fprintf(plotfile, " stroke-width: %d;", SVGstroke_width);
    }

    /* Style text.  */

    if (SVGfont_family) {
        fprintf(plotfile, " font-family: %s;\n", SVGfont_family);
    }
    if (SVGfont) {
        fprintf(plotfile, " font: %s;\n", SVGfont_family);
    }
    fputs("\">\n\n<!-- Creator: NGspice -->\n\n", plotfile);

    /* Fill background. */

    fprintf(plotfile,
            "<rect x=\"0\" y=\"0\" width=\"%d\" height=\"%d\" "
            "fill=\"%s\" stroke=\"none\"/>\n",
            graph->absolute.width, graph->absolute.height,
            SVGbackground ? SVGbackground : "black");

    /* Allocate and initialise per-graph data. */

    tfree(graph->devdep);

    graph->devdep = TMALLOC(SVGdevdep, 1);
    ddp = DEVDEP_P(graph);
    ddp->lastx = ddp->lasty = -1;
    return 0;
}


int
SVG_Close(void)
{
    /* Test for activity in case SVG_Close is called as part of an abort,
     * without having reached SVG_NewViewport
     */

    size_t i;

    if (plotfile) {
        closepath(DEVDEP_P(currentgraph));
        fprintf(plotfile, "</svg>\n");
        fclose(plotfile);
        plotfile = NULL;
    }

    if (colors) {
        for (i = 0; i < NUMELEMS(svgcolors); i++)
            tfree(colors[i]);
        tfree(colors);
    }

    for (i = 0; i < NUM_STRINGS; i++)
        tfree(Cfg.strings[i]);

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
SVG_Clear(void)
{
    /* do nothing */
    return 0;
}


int
SVG_DrawLine(int x1, int y1, int x2, int y2, bool isgrid)
{
    SVGdevdep *ddp;

    if (x1 == x2 && y1 == y2)
        return 0;

    ddp = DEVDEP_P(currentgraph);
    if (isgrid != ddp->isgrid) {
        closepath(ddp);
        ddp->isgrid = isgrid;
    }
    if (isgrid && ddp->inpath == NOPATH)
        startpath_width(ddp, SVGgrid_width);
    CHECK_PATH;
    if (x1 == ddp->lastx && y1 == ddp->lasty) {
        putc((ddp->inpath == LINE) ? ' ' : 'l', plotfile);
        ++ddp->linelen;
    } else {
        ddp->linelen += fprintf(plotfile, "M%d %dl", x1, dispdev->height - y1);
    }
    ddp->linelen += fprintf(plotfile, "%d %d", x2 - x1, y1 - y2);
    ddp->lastx = x2;
    ddp->lasty = y2;
    ddp->inpath = LINE;
    return 0;
}


int
SVG_Arc(int x0, int y0, int r, double theta, double delta_theta, bool isgrid)
{
    double x1, y1, x2, y2, left;
    SVGdevdep *ddp;

    /* SVG will not draw full circles, so do them in pieces. */

    if (delta_theta < 0.0) {
        theta += delta_theta;
        delta_theta = -delta_theta;
    }
    if (delta_theta > M_PI) {
        left = delta_theta - M_PI;
        if (left > M_PI)
            left = M_PI;
        delta_theta = M_PI;
    } else {
        left = 0.0;
    }

    ddp = DEVDEP_P(currentgraph);
    if (isgrid != ddp->isgrid) {
        closepath(ddp);
        ddp->isgrid = isgrid;
    }
    if (isgrid && ddp->inpath == NOPATH)
        startpath_width(ddp, SVGgrid_width);
    CHECK_PATH;

    x1 = (double) x0 + r * cos(theta);
    y1 = (double) y0 + r * sin(theta);
    x2 = (double) x0 + r * cos(theta + delta_theta);
    y2 = (double) y0 + r * sin(theta + delta_theta);

    ddp->linelen += fprintf(plotfile, "M%f %fA%d %d 0 0 0 %f %f",
                            x1, dispdev->height - y1, r, r,
                            x2, dispdev->height - y2);
    if (left != 0.0) {
        x2 = (double) x0 + r * cos(theta + M_PI + left);
        y2 = (double) y0 + r * sin(theta + M_PI + left);
        ddp->linelen += fprintf(plotfile, " %d %d 0 0 0 %f %f",
                                r, r, x2, dispdev->height - y2);
    }
    ddp->lastx = -1;
    ddp->lasty = -1;
    ddp->inpath = PATH;
    return 0;
}


int
SVG_Text(const char *text, int x, int y, int angle)
{
    SVGdevdep *ddp;

    ddp = DEVDEP_P(currentgraph);
    if (ddp->inpath != NOPATH)
        closepath(ddp);

    y = dispdev->height - y;
    fputs("<text", plotfile);
    if (angle != 0) {
        fprintf(plotfile, " transform=\"rotate(%d, %d, %d)\" ",
                -angle, x, y);
    }
    fprintf(plotfile,
            " stroke=\"none\" fill=\"%s\" font-size=\"%d\" "
            "x=\"%d\" y=\"%d\">\n%s\n</text>\n",
            colors[currentgraph->currentcolor],
            SVGfont_size,
            x, y, text);
    return 0;
}


/* SVG_DefineColor() and SVG_DefineLinestyle() are never used. */

int
SVG_SetLinestyle(int linestyleid)
{
    /* special case: get it when SVG_Text restores a -1 linestyle */

    if (linestyleid == -1) {
        currentgraph->linestyle = -1;
        return 0;
    }
    
    if (SVGuse_color == 1 && linestyleid > 1) {
        /* Caller ignores dispdev->numlinestyles.  Keep quiet,
         * but allow dotted grid.
         */
        currentgraph->linestyle = 0;
        return 0;
    }

    if (linestyleid < 0 || linestyleid > dispdev->numlinestyles) {
        internalerror("bad linestyleid inside SVG_SetLinestyle");
        fprintf(cp_err, "linestyleid is: %d\n", linestyleid);
        return 1;
    }

    if (linestyleid != currentgraph->linestyle) {
        closepath(DEVDEP_P(currentgraph));
        currentgraph->linestyle = linestyleid;
    }
    return 0;
}


int
SVG_SetColor(int colorid)
{
    if (colorid < 0 || colorid > (int)NUMELEMS(svgcolors)) {
        internalerror("bad colorid inside SVG_SelectColor");
        return 1;
    }
    if (colorid != currentgraph->currentcolor) {
        closepath(DEVDEP_P(currentgraph));
        currentgraph->currentcolor = colorid;
    }
    return 0;
}


int
SVG_Update(void)
{
    fflush(plotfile);
    return 0;
}

int
SVG_Finalize(void)
{
    fputs("\"/>\n", plotfile);
    return 0;
}

/**************** PRIVATE FUNCTIONS OF SVG FRONTEND ***************************/

/* Start a new path. */

static void startpath_width(SVGdevdep *ddp, int width)
{
    if (ddp->inpath != NOPATH)
        closepath(ddp);
    ddp->linelen = 3 +
        fprintf(plotfile, "<path stroke=\"%s\" ",
                colors[currentgraph->currentcolor]);
    if (width) {
        ddp->linelen += fprintf(plotfile, "stroke-width=\"%d\" ", width);
    }

    /* Kludgy, but allow dash style 1 (the grid) when suppressing dashes. */

    if (SVGuse_color != 1 || currentgraph->linestyle == 1) {
        ddp->linelen += fprintf(plotfile, "stroke-dasharray=\"%s\" ",
                                linestyles[currentgraph->linestyle]);
    }
    fputs("d=\"", plotfile);
    ddp->inpath = PATH;
}

static void startpath(SVGdevdep *ddp) {
    startpath_width(ddp, 0);
}

static void closepath(SVGdevdep *ddp)
{
    if (ddp->inpath != NOPATH) {
        fputs("\"/>\n", plotfile);
        ddp->inpath = NOPATH;
    }
    ddp->lastx = -1;
    ddp->lasty = -1;
}
