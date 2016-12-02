/*
 * Frame buffer for the PC using MS Windows
 * Wolfgang Muees 27.10.97
 * Holger Vogt  07.12.01
 * Holger Vogt  05.12.07
 */

#include "ngspice/ngspice.h"

#ifdef HAS_WINGUI

#include "ngspice/graph.h"
#include "ngspice/ftedev.h"
#include "ngspice/ftedbgra.h"
#include "ngspice/fteext.h"
#include "../plotting/graf.h"
#include "../plotting/graphdb.h"
#include "windisp.h"

/*
 * The ngspice.h file included above defines BOOLEAN (via bool.h) and this
 * clashes with the definition obtained from windows.h (via winnt.h).
 * However, BOOLEAN is not used by this file so we can work round this problem
 * by undefining BOOLEAN before including windows.h
 * SJB - May 2005
 */
#undef BOOLEAN

#define STRICT
#include <windows.h>
#include <windowsx.h>
#include "ngspice/suffix.h"

/* Typen */
typedef struct {      /* Extra window data */
    HWND  wnd;        /* window */
    HDC   hDC;        /* Device context of window */
    RECT  Area;       /* plot area */
    int   ColorIndex; /* Index of actual color */
    int   PaintFlag;  /* 1 with WM_PAINT */
    int   FirstFlag;  /* 1 before first update */
} tWindowData;
typedef tWindowData *tpWindowData;       /* pointer to it */

#define pWindowData(g) ((tpWindowData)(g->devdep))

LRESULT CALLBACK PlotWindowProc(HWND hwnd,     /* window procedure */
                                UINT uMsg, WPARAM wParam, LPARAM lParam);
void WPRINT_PrintInit(HWND hwnd);              /* Windows printer init */
void WaitForIdle(void);                        /* wait until no more events */

static void WIN_ScreentoData(GRAPH *graph, int x, int y, double *fx, double *fy);
static LRESULT HcpyPlot(HWND hwnd);
static LRESULT HcpyPlotBW(HWND hwnd);
static LRESULT PrintPlot(HWND hwnd);
static LRESULT PrintInit(HWND hwnd);
//static void RealClose(void);

extern HINSTANCE   hInst;         /* application instance */
extern int         WinLineWidth;  /* width of text window */
extern HWND        swString;      /* string input window of main window */
//extern struct plot *plot_cur;
extern int         DevSwitch(char *devname);
extern int         NewViewport(GRAPH *pgraph);
extern void        com_hardcopy(wordlist *wl);

/* defines */
#define RAD_TO_DEG   (180.0 / M_PI)

#ifndef M_LN10
#define M_LN10  2.30258509299404568402
#endif

#define DEF_FONTW "Arial"

/* local variables */
static int           IsRegistered = 0;             /* 1 if window class is registered */
#define NumWinColors 23                            /* predefined colors */
static COLORREF      ColorTable[NumWinColors];     /* color memory */
static char         *WindowName = "Spice Plot";    /* window name */
static WNDCLASS      TheWndClass;                  /* Plot-window class */
static HFONT         PlotFont;                     /* which font */
#define              ID_DRUCKEN      0xEFF0        /* System Menue: print */
#define              ID_DRUCKEINR    0xEFE0        /* System Menue: printer setup */
#define              ID_HARDCOPY     0xEFD0        /* System Menue: hardcopy color*/
#define              ID_HARDCOPY_BW  0xEFB0        /* System Menue: hardcopy b&w*/
#define              ID_MASK         0xFFF0;       /* System-Menue: mask */

static char         *STR_DRUCKEN   = "Printer..."; /* System menue strings */
static char         *STR_DRUCKEINR = "Printer setup...";
static char         *STR_HARDCOPY = "Postscript file, color";
static char         *STR_HARDCOPY_BW = "Postscript file, b&w";
static bool          isblack = TRUE;               /* background color of plot is black */
static bool          isblackold = TRUE;
static int           linewidth = 0;                /* linewidth of grid and plot */

/******************************************************************************
WIN_Init() makes connection to graphics. We have to determine

   dispdev->numlinestyles  (if color screen == 1)
   dispdev->numcolors
   dispdev->width          (preliminary window width)
   dispdev->height         (preliminary window height)

WIN_Init() returns 0, if no error ocurred.

WIN_Init() does not yet open a window, this happens only in WIN_NewViewport()
******************************************************************************/

int
WIN_Init(void)
{
    char colorstring[BSIZE_SP];

    /* Initialization of display descriptor */
    dispdev->width         = GetSystemMetrics(SM_CXSCREEN);
    dispdev->height        = GetSystemMetrics(SM_CYSCREEN);
    dispdev->numlinestyles = 5;   /* see implications in WinPrint! */
    dispdev->numcolors     = NumWinColors;

    /* always, user may have set color0 to white */
    /* get background color information from spinit, only "white"
       is recognized as a suitable option! */
    if (cp_getvar("color0", CP_STRING, colorstring))
        isblack = !cieq(colorstring, "white");

    /* get linewidth information from spinit */
    if (!cp_getvar("xbrushwidth", CP_NUM, &linewidth))
        linewidth = 0;
    if (linewidth < 0)
        linewidth = 0;

    /* only for the first time: */
    if (!IsRegistered) {

        isblackold = isblack;

        /* get linewidth information from spinit
         * if (!cp_getvar("xbrushwidth", CP_NUM, &linewidth))
         *     linewidth = 0;
         * if (linewidth < 0)
         *     linewidth = 0;
         */

        /* Initialize colors */
        if (isblack) {
            ColorTable[0] = RGB(  0,   0,   0);   /* black   = background */
            ColorTable[1] = RGB(255, 255, 255);   /* white   = text and grid */
        }
        else {
            ColorTable[0] = RGB(255, 255, 255);   /* white   = background */
            ColorTable[1] = RGB(  0,   0,   0);   /* black   = text and grid */
        }

        ColorTable[2]  = RGB(  0, 255,   0);   /* green   = first line */
        ColorTable[3]  = RGB(255,   0,   0);   /* red */
        ColorTable[4]  = RGB(  0,   0, 255);   /* blue */
        ColorTable[5]  = RGB(255, 255,   0);   /* yellow */
        ColorTable[6]  = RGB(255,   0, 255);   /* violett */
        ColorTable[7]  = RGB(  0, 255, 255);   /* azur */
        ColorTable[8]  = RGB(255, 128,   0);   /* orange */
        ColorTable[9]  = RGB(128,  64,   0);   /* brown */
        ColorTable[10] = RGB(128,   0, 255);   /* light violett */
        ColorTable[11] = RGB(255, 128, 128);   /* pink */

        /* 2. color bank (with different line style */
        if (isblack)
            ColorTable[12] = RGB(255, 255, 255);   /* white */
        else
            ColorTable[12] = RGB(  0,   0,   0);   /* black */

        ColorTable[13] = RGB(  0, 255,   0);   /* green */
        ColorTable[14] = RGB(255,   0,   0);   /* red */
        ColorTable[15] = RGB(  0,   0, 255);   /* blue */
        ColorTable[16] = RGB(255, 255,   0);   /* yellow */
        ColorTable[17] = RGB(255,   0, 255);   /* violett */
        ColorTable[18] = RGB(  0, 255, 255);   /* azur */
        ColorTable[19] = RGB(255, 128,   0);   /* orange */
        ColorTable[20] = RGB(128,  64,   0);   /* brown */
        ColorTable[21] = RGB(128,   0, 255);   /* light violett */
        ColorTable[22] = RGB(255, 128, 128);   /* pink */

        /* Ansii fixed font */
        PlotFont = GetStockFont(ANSI_FIXED_FONT);

        /* register window class */
        TheWndClass.lpszClassName  = WindowName;
        TheWndClass.hInstance      = hInst;
        TheWndClass.lpfnWndProc    = PlotWindowProc;
        TheWndClass.style          = CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
        TheWndClass.lpszMenuName   = NULL;
        TheWndClass.hCursor        = LoadCursor(NULL, IDC_ARROW);

        if (isblack)
            TheWndClass.hbrBackground  = GetStockObject(BLACK_BRUSH);
        else
            TheWndClass.hbrBackground  = GetStockObject(WHITE_BRUSH);

        TheWndClass.hIcon          = LoadIcon(hInst, MAKEINTRESOURCE(2));
        TheWndClass.cbClsExtra     = 0;
        TheWndClass.cbWndExtra     = sizeof(GRAPH *);

        if (!RegisterClass(&TheWndClass))
            return 1;
    }
    /* not first time */
    else if (isblackold != isblack) {
        if (isblack) {
            ColorTable[0]  = RGB(  0,   0,   0);   /* black   = background */
            ColorTable[1]  = RGB(255, 255, 255);   /* white    = text and grid */
        }
        else {
            ColorTable[0]  = RGB(255, 255, 255);   /* white   = background */
            ColorTable[1]  = RGB(  0,   0,   0);   /* black   = text and grid */
        }
        if (isblack)
            ColorTable[12] = RGB(255, 255, 255);   /* white */
        else
            ColorTable[12] = RGB(  0,   0,   0);   /* black */

        isblackold = isblack;
    }
    IsRegistered = 1;

    /* ready */
    return 0;
}


/* get pointer to graph */
/* (attach to window) */
static GRAPH *
pGraph(HWND hwnd)
{
    return (GRAPH *) GetWindowLongPtr(hwnd, 0);
}


/* return line style for plotting */
static int
LType(int ColorIndex)
{
    if (ColorIndex >= 12)
        return PS_DOT;
    else
        return PS_SOLID;
}


/* postscript hardcopy from a plot window */
/* called by SystemMenue / Postscript hardcopy */
static LRESULT
HcpyPlot(HWND hwnd)
{
    int colorval = isblack? 0 : 1;
    NG_IGNORE(hwnd);
    cp_vset("hcopypscolor", CP_NUM, &colorval);
    com_hardcopy(NULL);
    return 0;
}


static LRESULT
HcpyPlotBW(HWND hwnd)
{
    int bgcolor;
    NG_IGNORE(hwnd);
    if (cp_getvar("hcopypscolor", CP_NUM, &bgcolor))
        cp_remvar("hcopypscolor");
    com_hardcopy(NULL);
    return 0;
}


/* print a plot window */
/* called by SystemMenue / Print */
static LRESULT
PrintPlot(HWND hwnd)
{
    GRAPH *graph;
    GRAPH *temp;

    /* get pointer to graph */
    graph = pGraph(hwnd);
    if (!graph)
        return 0;

    /* switch to printer */
    /* (results in WPRINT_Init()) */
    if (DevSwitch("WinPrint"))
        return 0;

    /* Cursor = wait */
    SetCursor(LoadCursor(NULL, IDC_WAIT));

    /* copy graph */
    temp = CopyGraph(graph);
    if (!temp)
        goto PrintEND;

    /* add to the copy the new printer data */
    if (NewViewport(temp))
        goto PrintEND2;

    /* make correction to placement of grid (copy from gr_init) */
    temp->viewportxoff = temp->fontwidth  * 8;
    temp->viewportyoff = temp->fontheight * 4;

    /* print the graph */
    gr_resize(temp);

 PrintEND2:
    /* delete temporary graph */
    DestroyGraph(temp->graphid);

 PrintEND:
    /* switch back to screen */
    DevSwitch(NULL);

    /* Cursor = normal */
    SetCursor(LoadCursor(NULL, IDC_ARROW));

    return 0;
}


/* initialze printer */
static LRESULT
PrintInit(HWND hwnd)
{
    /* hand over to printer module */
    WPRINT_PrintInit(hwnd);
    return 0;
}


/* window procedure */
LRESULT CALLBACK
PlotWindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    static int x0, y0, xep, yep;
    int xe, ye, prevmix;
    static double fx0, fy0;
    double fxe, fye;
    double angle;
    char buf[BSIZE_SP];
    char buf2[128];
    char *t;
    HDC hdc;
    HPEN OldPen;
    HPEN NewPen;

    switch (uMsg) {
    case WM_SYSCOMMAND:
    {
        /* test command */
        WPARAM cmd = wParam & ID_MASK;
        switch(cmd) {
        case ID_DRUCKEN:     return PrintPlot(hwnd);
        case ID_DRUCKEINR:   return PrintInit(hwnd);
        case ID_HARDCOPY:    return HcpyPlot(hwnd);
        case ID_HARDCOPY_BW: return HcpyPlotBW(hwnd);
        }
    }
    goto WIN_DEFAULT;

    case WM_LBUTTONDOWN:
    {
        GRAPH *gr = pGraph(hwnd);
        xep = x0 = LOWORD (lParam);
        yep = y0 = HIWORD (lParam);
        /* generate x,y data from grid coordinates */
        WIN_ScreentoData(gr, x0, y0, &fx0, &fy0);
    }
    goto WIN_DEFAULT;

    case WM_MOUSEMOVE:
    {
        /* left mouse button: connect coordinate pair by dashed pair of x, y lines */
        if (wParam & MK_LBUTTON) {
            hdc = GetDC(hwnd);
            if (isblack)
                prevmix = SetROP2(hdc, R2_XORPEN);
            else
                prevmix = SetROP2(hdc, R2_NOTXORPEN);
            /* Create white dashed pen */
            NewPen = CreatePen(LType(12), 0, ColorTable[1]);
            OldPen = SelectObject(hdc, NewPen);
            /* draw lines with previous coodinates -> delete old line because of XOR */
            MoveToEx (hdc, x0, y0, NULL);
            LineTo   (hdc, x0, yep);
            LineTo   (hdc, xep, yep);
            /* get new end point */
            xe = LOWORD (lParam);
            ye = HIWORD (lParam);
            /* draw new lines */
            MoveToEx (hdc, x0, y0, NULL);
            LineTo   (hdc, x0, ye);
            LineTo   (hdc, xe, ye);
            /* restore standard color mix */
            SetROP2(hdc, prevmix);
            OldPen = SelectObject(hdc, OldPen);
            DeleteObject(NewPen);
            ReleaseDC (hwnd, hdc);
            /* restore new to previous coordinates */
            yep = ye;
            xep = xe;
        }
        /* right mouse button: create white (black) dashed box */
        else if (wParam & MK_RBUTTON) {
            hdc = GetDC (hwnd);
            if (isblack)
                prevmix = SetROP2(hdc, R2_XORPEN);
            else
                prevmix = SetROP2(hdc, R2_NOTXORPEN);
            /* Create white (black) dashed pen */
            NewPen = CreatePen(LType(12), 0, ColorTable[1]);
            OldPen = SelectObject(hdc, NewPen);
            /* draw box with previous coodinates -> delete old lines because of XOR */
            MoveToEx (hdc, x0, y0, NULL);
            LineTo   (hdc, x0, yep);
            LineTo   (hdc, xep, yep);
            LineTo   (hdc, xep, y0);
            LineTo   (hdc, x0, y0);
            /* get new end point */
            xe = LOWORD (lParam);
            ye = HIWORD (lParam);
            /* draw new box */
            MoveToEx (hdc, x0, y0, NULL);
            LineTo   (hdc, x0, ye);
            LineTo   (hdc, xe, ye);
            LineTo   (hdc, xe, y0);
            LineTo   (hdc, x0, y0);
            /* restore standard color mix */
            SetROP2(hdc, prevmix);
            OldPen = SelectObject(hdc, OldPen);
            DeleteObject(NewPen);
            ReleaseDC (hwnd, hdc);
            /* restore new to previous coordinates */
            yep = ye;
            xep = xe;
        }
    }
    goto WIN_DEFAULT;

    /* get final coordinates upon left mouse up */
    /* calculate and print out the data */
    case WM_LBUTTONUP:
    {
        GRAPH *gr = pGraph(hwnd);
        InvalidateRect (hwnd, NULL, TRUE);
        xe = LOWORD (lParam);
        ye = HIWORD (lParam);
        WIN_ScreentoData(gr, xe, ye, &fxe, &fye);

        /* print it out */
        if (xe == x0 && ye == y0) {     /* only one location */
            fprintf(stdout, "\nx0 = %g, y0 = %g\n", fx0, fy0);
            if (gr->grid.gridtype == GRID_POLAR ||
                gr->grid.gridtype == GRID_SMITH ||
                gr->grid.gridtype == GRID_SMITHGRID)
            {
                angle = RAD_TO_DEG * atan2(fy0, fx0);
                fprintf(stdout, "r0 = %g, a0 = %g\n",
                        hypot(fx0, fy0),
                        (angle > 0) ? angle : 360.0 + angle);
            }
        } else  {
            /* need to print info about two points */
            fprintf(stdout, "\nx0 = %g, y0 = %g    x1 = %g, y1 = %g\n",
                    fx0, fy0, fxe, fye);
            fprintf(stdout, "dx = %g, dy = %g\n", fxe-fx0, fye - fy0);
            if (xe != x0 && ye != y0) {
                /* add slope info if both dx and dy are zero, */
                /* because otherwise either dy/dx or dx/dy is zero, */
                /* which is uninteresting */

                fprintf(stdout, "dy/dx = %g    dx/dy = %g\n",
                        (fye - fy0) / (fxe - fx0), (fxe - fx0) / (fye - fy0));
            }
        }
        SetFocus(swString);
    }
    goto WIN_DEFAULT;

    /* get starting coordinates upon right mouse button down */
    case WM_RBUTTONDOWN:
    {
        GRAPH *gr = pGraph(hwnd);
        x0 = xep = LOWORD (lParam);
        y0 = yep = HIWORD (lParam);
        WIN_ScreentoData(gr, x0, y0, &fx0, &fy0);
    }
    goto WIN_DEFAULT;

    /* get final coordinates upon right mouse button up */
    /* copy xlimit, ylimit command into buf */
    /* start plot loop with argument buf   */
    case WM_RBUTTONUP:
    {
        GRAPH *gr = pGraph(hwnd);
        InvalidateRect (hwnd, NULL, TRUE);
        xe = LOWORD (lParam);
        ye = HIWORD (lParam);
        /* do nothing if mouse curser is not moved in both x and y */
        if ((xe == x0) || (ye == y0)) {
            SetFocus(swString);
            goto WIN_DEFAULT;
        }
        WIN_ScreentoData(gr, xe, ye, &fxe, &fye);

        strncpy(buf2, gr->plotname, sizeof(buf2));
        if ((t = strchr(buf2, ':')) != NULL)
            *t = '\0';

        if (!eq(plot_cur->pl_typename, buf2)) {
            (void) sprintf(buf,
//       "setplot %s; %s xlimit %e %e ylimit %e %e; setplot $curplot\n",
                           "setplot %s; %s xlimit %e %e ylimit %e %e\n",
                           buf2, gr->commandline, fx0, fxe, fy0, fye);
        } else {
            (void) sprintf(buf, "%s xlimit %e %e ylimit %e %e\n",
                           gr->commandline, fx0, fxe, fy0, fye);
        }

        (void) cp_evloop(buf);
        SetFocus(swString);
    }
    goto WIN_DEFAULT;

    case WM_CLOSE: /* close window */
    {
        GRAPH *g = pGraph(hwnd);

        if (g) {
            /* if g equals currentgraph, reset currentgraph. */
            if (g == currentgraph)
                currentgraph = NULL;
            DestroyGraph(g->graphid);
        }
    }
    goto WIN_DEFAULT;

    case WM_PAINT: /* replot window (e.g. after Resize) */
    {
        PAINTSTRUCT ps;
        GRAPH *g;
        tpWindowData wd;
        HDC saveDC;    /* the DC from BeginPaint is different... */
        HDC newDC;

        /* has to happen */
        newDC = BeginPaint(hwnd, &ps);
        g = pGraph(hwnd);
        if (g) {
            wd = pWindowData(g);
            if (wd) {
                if (!wd->PaintFlag && !wd->FirstFlag) {
                    /* avoid recursive call */
                    wd->PaintFlag = 1;
                    /* get window sizes */
                    GetClientRect(hwnd, &(wd->Area));
                    g->absolute.width  = wd->Area.right;
                    g->absolute.height = wd->Area.bottom;
                    /* switch DC */
                    saveDC = wd->hDC;
                    wd->hDC = newDC;

                    /* plot anew */
                    {
                        GRAPH *tmp = currentgraph;
                        currentgraph = g;
                        gr_resize(g);
                        currentgraph = tmp;
                    }

                    /* switch DC */
                    wd->hDC = saveDC;
                    /* ready */
                    wd->PaintFlag = 0;
                }
            }
        }
        /* finish */
        EndPaint(hwnd, &ps);
    }
    return 0;

    default:
    WIN_DEFAULT:
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
    }
}


/******************************************************************************
 WIN_NewViewport() creates a new window with a graph inside.

 WIN_NewViewport() returns 0 if successful

******************************************************************************/

int
WIN_NewViewport(GRAPH *graph)
{
    int      i;
    HWND     window;
    HDC      dc;
    TEXTMETRIC  tm;
    tpWindowData   wd;
    HMENU    sysmenu;

    /* test the parameters */
    if (!graph)
        return 1;

    /* initialize if not yet done */
    if (WIN_Init() != 0) {
        externalerror("Can't initialize GDI.");
        return 1;
    }

    /* allocate device dependency info */
    wd = calloc(1, sizeof(tWindowData));
    if (!wd)
        return 1;

    graph->devdep = wd;

    /* Create the window */
    i = GetSystemMetrics(SM_CYSCREEN) / 3;
    window = CreateWindow(WindowName, graph->plotname, WS_OVERLAPPEDWINDOW,
                          0, 0, WinLineWidth, i * 2 - 22, NULL, NULL, hInst, NULL);
    if (!window)
        return 1;

    /* change the background color of all windows (both new and already plotted)
       by assessing the registered window class */
    if (isblack)
        SetClassLongPtr(window, GCLP_HBRBACKGROUND, (LONG_PTR)GetStockObject(BLACK_BRUSH));
    else
        SetClassLongPtr(window, GCLP_HBRBACKGROUND, (LONG_PTR)GetStockObject(WHITE_BRUSH));


    wd->wnd = window;
    SetWindowLongPtr(window, 0, (LONG_PTR)graph);

    /* show window */
    ShowWindow(window, SW_SHOWNORMAL);

    /* get the mask */
    GetClientRect(window, &(wd->Area));

    /* get the DC */
    dc = GetDC(window);
    wd->hDC = dc;

    /* set the Color Index */
    wd->ColorIndex = 0;

    /* still no flag */
    wd->PaintFlag = 0;
    wd->FirstFlag = 1;

    /* modify system menue */
    sysmenu = GetSystemMenu(window, FALSE);
    AppendMenu(sysmenu, MF_SEPARATOR, 0, NULL);
    AppendMenu(sysmenu, MF_STRING, ID_DRUCKEN,   STR_DRUCKEN);
    AppendMenu(sysmenu, MF_STRING, ID_DRUCKEINR, STR_DRUCKEINR);
    AppendMenu(sysmenu, MF_STRING, ID_HARDCOPY, STR_HARDCOPY);
    AppendMenu(sysmenu, MF_STRING, ID_HARDCOPY_BW, STR_HARDCOPY_BW);

    /* set default parameters of DC */
    SetBkColor(dc, ColorTable[0]);
    SetBkMode(dc, TRANSPARENT );

    /* set font */
    SelectObject(dc, PlotFont);

    /* query the font parameters */
    if (GetTextMetrics(dc, &tm)) {
        graph->fontheight = tm.tmHeight;
        graph->fontwidth  = tm.tmAveCharWidth;
    }

    /* set viewport parameters */
    graph->viewport.height = wd->Area.bottom;
    graph->viewport.width  = wd->Area.right;

    /* set absolute parameters */
    graph->absolute.xpos   = 0;
    graph->absolute.ypos   = 0;
    graph->absolute.width  = wd->Area.right;
    graph->absolute.height = wd->Area.bottom;

    /* wait until the window is really there */
    WaitForIdle();

    /* ready */
    return 0;
}


/******************************************************************************
WIN_Close is essentially the counterpart to WIN_Init. unfortunately it might
happen, that WIN_Close is called during plotting, because one wants to switch
to the printer. Therefore WIN_Close is not allowed to do anything, cancelling
of the structures occurs at program termination.
******************************************************************************/

int
WIN_Close(void)
{
    return 0;
}


#if 0
static void
RealClose(void)
{
    // delete window class
    if (IsRegistered) {
        if (TheWndClass.hIcon) {
            DestroyIcon(TheWndClass.hIcon);
            TheWndClass.hIcon = NULL;
        }
        UnregisterClass(WindowName, hInst);
        IsRegistered = FALSE;
    }
}
#endif


int
WIN_Clear(void)
{
    tpWindowData wd;

    if (!currentgraph)
        return 0;

    wd = pWindowData(currentgraph);
    if (!wd)
        return 0;

    /* this is done by the window itself */
    if (!wd->PaintFlag)  /* not necessary with WM_PAINT */
        SendMessage(wd->wnd, WM_ERASEBKGND, (WPARAM) wd->hDC, 0);

    return 0;
}


int
WIN_DrawLine(int x1, int y1, int x2, int y2)
{
    tpWindowData wd;
    HPEN      OldPen;
    HPEN      NewPen;

    if (!currentgraph)
        return 0;

    wd = pWindowData(currentgraph);
    if (!wd)
        return 0;

    MoveToEx(wd->hDC, x1, wd->Area.bottom - y1, NULL);
    NewPen = CreatePen(LType(wd->ColorIndex), linewidth, ColorTable[wd->ColorIndex]);
    OldPen = SelectObject(wd->hDC, NewPen);
    LineTo(wd->hDC, x2, wd->Area.bottom - y2);
    OldPen = SelectObject(wd->hDC, OldPen);
    DeleteObject(NewPen);

    return 0;
}


int
WIN_Arc(int x0, int y0, int radius, double theta, double delta_theta)
/*
 * Notes:
 *    Draws an arc of <radius> and center at (x0,y0) beginning at
 *    angle theta (in rad) and ending at theta + delta_theta
 */
{
    tpWindowData wd;
    HPEN     OldPen;
    HPEN     NewPen;
    int   left, right, top, bottom;
    int   xs, ys, xe, ye;
    int      yb;
    int   direction;
    double   r;
    double  dx0;
    double   dy0;

    if (!currentgraph)
        return 0;

    wd = pWindowData(currentgraph);
    if (!wd)
        return 0;

    direction = AD_COUNTERCLOCKWISE;
    if (delta_theta < 0) {
        theta += delta_theta;
        delta_theta = - delta_theta;
        direction = AD_CLOCKWISE;
    }
    SetArcDirection(wd->hDC, direction);

    /* some geometric considerations in advance */
    yb     = wd->Area.bottom;
    left   = x0 - radius;
    right  = x0 + radius;
    top    = y0 + radius;
    bottom = y0 - radius;

    r = radius;
    dx0 = x0;
    dy0 = y0;
    xs = (int)(dx0 + (r * cos(theta)));
    ys = (int)(dy0 + (r * sin(theta)));
    xe = (int)(dx0 + (r * cos(theta + delta_theta)));
    ye = (int)(dy0 + (r * sin(theta + delta_theta)));

    /* plot */
    NewPen = CreatePen(LType(wd->ColorIndex), linewidth, ColorTable[wd->ColorIndex]);
    OldPen = SelectObject(wd->hDC, NewPen);
    Arc(wd->hDC, left, yb-top, right, yb-bottom, xs, yb-ys, xe, yb-ye);
    OldPen = SelectObject(wd->hDC, OldPen);
    DeleteObject(NewPen);

    return 0;
}


#if 0
int
WIN_Text_old(char *text, int x, int y, int degrees)
{
    tpWindowData wd;

    if (!currentgraph)
        return 0;

    wd = pWindowData(currentgraph);
    if (!wd)
        return 0;

    SetTextColor(wd->hDC, ColorTable[wd->ColorIndex]);
    TextOut(wd->hDC, x, wd->Area.bottom - y - currentgraph->fontheight, text, strlen(text));

    return 0;
}
#endif


int
WIN_Text(char *text, int x, int y, int angle)
{
    tpWindowData wd;
    HFONT hfont;
    LOGFONT lf;

    if (!currentgraph)
        return 0;

    wd = pWindowData(currentgraph);
    if (!wd)
        return 0;

    lf.lfHeight         = (int) (1.1 * currentgraph->fontheight);
    lf.lfWidth          = 0;
    lf.lfEscapement     = angle * 10;
    lf.lfOrientation    = angle * 10;
    lf.lfWeight         = 500;
    lf.lfItalic         = 0;
    lf.lfUnderline      = 0;
    lf.lfStrikeOut      = 0;
    lf.lfCharSet        = 0;
    lf.lfOutPrecision   = 0;
    lf.lfClipPrecision  = 0;
    lf.lfQuality        = 0;
    lf.lfPitchAndFamily = 0;

    /* set up fonts */
    if (!cp_getvar("wfont", CP_STRING, lf.lfFaceName))
        (void) lstrcpy(lf.lfFaceName, DEF_FONTW);

    if (!cp_getvar("wfont_size", CP_NUM, &(lf.lfHeight)))
        lf.lfHeight = (int) (1.3 * currentgraph->fontheight);

    hfont = CreateFontIndirect (&lf);
    SelectObject(wd->hDC, hfont);

    SetTextColor(wd->hDC, ColorTable[wd->ColorIndex]);
    TextOut(wd->hDC, x, wd->Area.bottom - y - currentgraph->fontheight, text, (int)strlen(text));

    DeleteObject(SelectObject(wd->hDC, GetStockObject(SYSTEM_FONT)));

    return 0;
}


int
WIN_DefineColor(int colorid, double red, double green, double blue)
{
    NG_IGNORE(colorid);
    NG_IGNORE(red);
    NG_IGNORE(green);
    NG_IGNORE(blue);
    return 0;
}


int
WIN_DefineLinestyle(int num, int mask)
{
    NG_IGNORE(num);
    NG_IGNORE(mask);
    return 0;
}


int
WIN_SetLinestyle(int style)
{
    NG_IGNORE(style);
    return 0;
}


int
WIN_SetColor(int color)
{
    tpWindowData wd;

    if (!currentgraph)
        return 0;

    wd = pWindowData(currentgraph);
    if (!wd)
        return 0;

    wd->ColorIndex = color % NumWinColors;

    return 0;
}


int
WIN_Update(void)
{
    tpWindowData wd;

    if (!currentgraph)
        return 0;

    wd = pWindowData(currentgraph);
    if (!wd)
        return 0;

    /* After the first run of Update() */
    /* FirstFlag again handles WM_PAINT messages. */
    /* This prevents double painting during displaying the window. */
    wd->FirstFlag = 0;
    return 0;
}


#if 0
int
WIN_DiagramReady(void)
{
    return 0;
}
#endif


void
RemoveWindow(GRAPH *dgraph)
{
    tpWindowData wd;

    wd = pWindowData(dgraph);
    if (wd)
        SendMessage(wd->wnd, WM_CLOSE, (WPARAM) wd->hDC, 0);
}


/* Function borrowed from x11.c */
static void WIN_ScreentoData(GRAPH *graph, int x, int y, double *fx, double *fy)
{
    double lmin, lmax;

    if (graph->grid.gridtype == GRID_XLOG ||
        graph->grid.gridtype == GRID_LOGLOG)
    {
        lmin = log10(graph->datawindow.xmin);
        lmax = log10(graph->datawindow.xmax);
        *fx = exp(((x - graph->viewportxoff) *
                   (lmax - lmin) / graph->viewport.width + lmin) * M_LN10);
    } else {
        *fx = (x - graph->viewportxoff) * graph->aspectratiox +
            graph->datawindow.xmin;
    }

    if (graph->grid.gridtype == GRID_YLOG ||
        graph->grid.gridtype == GRID_LOGLOG)
    {
        lmin = log10(graph->datawindow.ymin);
        lmax = log10(graph->datawindow.ymax);
        *fy = exp(((graph->absolute.height - y - graph->viewportxoff) *
                   (lmax - lmin) / graph->viewport.height + lmin) * M_LN10);
    } else {
        *fy = ((graph->absolute.height - y) - graph->viewportyoff) *
            graph->aspectratioy + graph->datawindow.ymin;
    }

}


#endif /* HAS_WINGUI */
