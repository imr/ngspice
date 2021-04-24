/*
 * Printing Routines for the IBM PC using MS Windows
 * Wolfgang Muees 27.10.97
 * Holger Vogt 07.12.01
 */

#include "ngspice/ngspice.h"

#ifdef HAS_WINGUI

#include "ngspice/graph.h"
#include "ngspice/ftedev.h"
#include "ngspice/ftedbgra.h"
#include "ngspice/fteext.h"

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
#ifdef _MSC_VER
#pragma hdrstop
#endif /* _MSC_VER */

#include "winprint.h"   /* function prototypes */

/* Typen */
typedef struct { /* Extra Printdaten */
    int ColorIndex; /* Index auf die akt. Farbe */
    int LineIndex; /* Index auf den akt. Linientyp */
} tPrintData;
typedef tPrintData *tpPrintData; /* Zeiger darauf */
#define pPrintData(g) ((tpPrintData)(g->devdep))

/* externals */

void WaitForIdle(void);                            /* wait until no events */
extern void wincolor_init(COLORREF* ColorTable, int noc);

/* lokale Variablen */
static HFONT            PlotFont = NULL;
static HFONT            OldFont  = NULL;
#define NumLines 7                                  /* number of LineStyles */
static int              LineTable[NumLines];        /* LineStyle memory */
static HDC              PrinterDC = NULL;           /* Device Context */
#define NumPrintColors 23                           /* number of pre-defined colors */
static COLORREF         ColorTable[NumPrintColors]; /* color memory */
static int              PrinterWidth  = 1000;       /* paper width */
static int              PrinterHeight = 1000;       /* paper height */
static int              xoffset;                    /* offset from left boundary */
static int              yoffset;                    /* offset from top */

/******************************************************************************
 printer initialization
******************************************************************************/

void WPRINT_PrintInit(HWND hwnd)
{
    /* Parameter-Block */
    PRINTDLG pd;

    /* Initialisieren */
    pd.lStructSize = sizeof(PRINTDLG);
    pd.hwndOwner = hwnd;
    pd.hDevMode = NULL;
    pd.hDevNames = NULL;
    pd.hDC = NULL;
    pd.Flags = PD_PRINTSETUP;
    pd.nFromPage = 1;
    pd.nToPage = 1;
    pd.nMinPage = 0;
    pd.nMaxPage = 0;
    pd.nCopies = 1;
    pd.hInstance = NULL;
    pd.lCustData = 0;
    pd.lpfnPrintHook = NULL;
    pd.lpfnSetupHook = NULL;
    pd.lpPrintTemplateName = NULL;
    pd.lpSetupTemplateName = NULL;
    pd.hPrintTemplate = NULL;
    pd.hSetupTemplate = NULL;

    /* Default-Drucker initialisieren */
    (void) PrintDlg(&pd);

    /* Speicher freigeben */
    if (pd.hDevMode) {
        GlobalFree(pd.hDevMode);
    }
    if (pd.hDevNames) {
        GlobalFree(pd.hDevNames);
    }
}

/* Abort-Procedur zum Drucken */
BOOL CALLBACK WPRINT_Abort(HDC hdc, int iError)
{
    NG_IGNORE(hdc);
    NG_IGNORE(iError);

    /* Multitasking */
    WaitForIdle();

    /* Warten */
    return TRUE;
}


/******************************************************************************
WPRINT_Init() stellt die Verbindung zur Grafik her. Dazu gehoert die Feststellung
von
    dispdev->numlinestyles
    dispdev->numcolors
    dispdev->width
    dispdev->height

WPRINT_Init() gibt 0 zurueck, falls kein Fehler auftrat.

******************************************************************************/

int WPRINT_Init(void)
{
    int    pWidth;
    int  pHeight;

    /* Printer-DC holen */
    if (!PrinterDC) {

        /* Parameter-Block */
        PRINTDLG pd;

        /* Initialisieren */
        pd.lStructSize = sizeof(PRINTDLG);
        pd.hwndOwner = NULL;
        pd.hDevMode = NULL;
        pd.hDevNames = NULL;
        pd.hDC = NULL;
        pd.Flags = PD_NOPAGENUMS | PD_NOSELECTION | PD_RETURNDC;
        pd.nFromPage = 1;
        pd.nToPage = 1;
        pd.nMinPage = 0;
        pd.nMaxPage = 0;
        pd.nCopies = 1;
        pd.hInstance = NULL;
        pd.lCustData = 0;
        pd.lpfnPrintHook = NULL;
        pd.lpfnSetupHook = NULL;
        pd.lpPrintTemplateName = NULL;
        pd.lpSetupTemplateName = NULL;
        pd.hPrintTemplate = NULL;
        pd.hSetupTemplate = NULL;

        /* Default-Drucker initialisieren */
        (void) PrintDlg( &pd);

        /* Speicher freigeben */
        if( pd.hDevMode)  GlobalFree( pd.hDevMode);
        if( pd.hDevNames) GlobalFree( pd.hDevNames);

        /* DC holen */
        PrinterDC = pd.hDC;
        if (!PrinterDC) return 1;

        /* Abmasze bestimmen */
        PrinterWidth    = GetDeviceCaps( PrinterDC, HORZRES);
        PrinterHeight   = GetDeviceCaps( PrinterDC, VERTRES);
        pWidth          = GetDeviceCaps( PrinterDC, HORZSIZE);
        pHeight         = GetDeviceCaps( PrinterDC, VERTSIZE);

        /* Mapping Mode setzen (fuer Kreise) */
        if ( pWidth > pHeight)
            /* Querformat */
            PrinterWidth = (PrinterHeight * pWidth) / pHeight;
        else
            /* Hochformat */
            PrinterHeight = (PrinterWidth * pHeight) / pWidth;

        SetMapMode( PrinterDC, MM_ISOTROPIC);
        SetWindowExtEx( PrinterDC, PrinterWidth, PrinterHeight, NULL);
        SetViewportExtEx( PrinterDC, PrinterWidth, PrinterHeight, NULL);

        /* nicht hoeher als breit zeichnen */
        if (pWidth < pHeight) {
            /* Papier im Hochformat */
            PrinterHeight = PrinterWidth;
        }

        /* Initialisierungen des Display-Descriptors */
        dispdev->width         = PrinterWidth;
        dispdev->height        = PrinterHeight;
        dispdev->numlinestyles = NumLines;
        dispdev->numcolors     = NumPrintColors;

        /* Farben initialisieren */
         wincolor_init(ColorTable, NumPrintColors);

        /* if grid/text color is white, change to black */
        if (ColorTable[1] == RGB(255, 255, 255))
            ColorTable[1] = 0;

        ColorTable[0] = RGB(255, 255, 255);

        /* LineStyles initialisieren */
        LineTable[0] = PS_SOLID;
        LineTable[1] = PS_DOT;      /* Gitter */
        LineTable[2] = PS_SOLID;    /* Erste Linie */
        LineTable[3] = PS_DOT;      /* Zweite Linie */
        LineTable[4] = PS_DASH;     /* usw */
        LineTable[5] = PS_DASHDOT;
        LineTable[6] = PS_DASHDOTDOT;

        /* Font */
        if (!PlotFont) {
            PlotFont = CreateFont( 0,0,0,0, FW_DONTCARE, FALSE, FALSE, FALSE,
                ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
                PROOF_QUALITY, FIXED_PITCH, NULL);
        }

        /* Abort-Prozedur setzen */
        SetAbortProc( PrinterDC, WPRINT_Abort);
    }
    /* fertig */
    return (0);
}


/******************************************************************************
 WPRINT_NewViewport() oeffnet den Drucker

 WPRINT_NewViewport() gibt 0 zurueck, falls erfolgreich

******************************************************************************/

int WPRINT_NewViewport(GRAPH * graph)
{

    TEXTMETRIC      tm;
    tpPrintData     pd;
    DOCINFO         di;

    /* Parameter testen */
    if (!graph) return 1;

    /* Initialisiere, falls noch nicht geschehen */
    if (WPRINT_Init() != 0) {
        externalerror("Can't initialize Printer.");
        return(1);
    }

    /* Device dep. Info allocieren */
    pd = calloc(1, sizeof(tPrintData));
    if (!pd) return 1;
    graph->devdep = pd;

    /* Setze den Color-Index */
    pd->ColorIndex = 0;

    /* Font setzen */
    OldFont = SelectObject( PrinterDC, PlotFont);

    /* Font-Parameter abfragen */
    if (GetTextMetrics( PrinterDC, &tm)) {
        graph->fontheight = tm.tmHeight;
        graph->fontwidth  = (int)(tm.tmAveCharWidth * 1.2);
    }

    /* Setze den Linien-Index */
    pd->LineIndex = 0;

    /* Viewport-Parameter setzen */
    graph->viewport.height  = PrinterHeight;
    graph->viewport.width   = PrinterWidth;

    /* Absolut-Parameter setzen */
    graph->absolute.xpos    = 0;
    graph->absolute.ypos    = 0;
    graph->absolute.width   = (int)(PrinterWidth * .9);
    graph->absolute.height  = (int)(PrinterHeight * .9);

    xoffset = (int)(PrinterWidth * .05);
    yoffset = (int)(PrinterHeight * -.05);

    /* Druckauftrag anmelden */
    di.cbSize       = sizeof( DOCINFO);
    di.lpszDocName  = graph->plotname;
    di.lpszOutput   = NULL;
    if (StartDoc( PrinterDC, &di) <= 0) return 1;
    if (StartPage( PrinterDC) <= 0) return 1;

    /* titel drucken */
    if (graph->plotname) {
        UINT align;
        align = GetTextAlign( PrinterDC);
        SetTextAlign( PrinterDC, TA_RIGHT | TA_TOP | TA_NOUPDATECP);
#ifdef EXT_ASC
        TextOut(PrinterDC, PrinterWidth - graph->fontwidth, -1*yoffset, graph->plotname,
            (int)strlen(graph->plotname));
#else
        /* wtext needs not to be NULL-terminated */
        const int n_byte_wide = (int) strlen(graph->plotname);
        wchar_t * const wtext = TMALLOC(wchar_t, n_byte_wide);
        MultiByteToWideChar(CP_UTF8, 0, graph->plotname, -1, wtext,
                n_byte_wide);
        TextOutW(PrinterDC, PrinterWidth - graph->fontwidth -xoffset, -1*yoffset, wtext,
                n_byte_wide);
        txfree(wtext);
#endif
        SetTextAlign(PrinterDC, align);
    }

    /* fertig */
    return(0);
}

int WPRINT_Close(void)
{
    if (PrinterDC) {
        EndPage(PrinterDC);
        EndDoc(PrinterDC);
        if (OldFont) {
            SelectObject(PrinterDC, OldFont);
            OldFont = NULL;
        }
        DeleteObject(PlotFont);
        DeleteDC(PrinterDC);
        PrinterDC = NULL;
    }
    return 0;
}


int WPRINT_Clear(void)
{
    return 0;
}


int WPRINT_DrawLine(int x1, int y1, int x2, int y2, bool isgrid)
{
    NG_IGNORE(isgrid);

    x1 = x1 + xoffset;
    x2 = x2 + xoffset;
    y1 = y1 + yoffset;
    y2 = y2 + yoffset;

    tpPrintData  pd;
    HPEN         OldPen;
    HPEN         NewPen;
    int          ColIndex;

    if (!currentgraph) return 0;
    pd = pPrintData(currentgraph);
    if (!pd) return 0;

    /* Farben/Dicke */
    ColIndex = pd->ColorIndex;

    MoveToEx(PrinterDC, x1, PrinterHeight - y1, NULL);
    if (isgrid)
        NewPen = CreatePen( LineTable[pd->LineIndex], 0, ColorTable[ColIndex] );
    else
        NewPen = CreatePen( LineTable[pd->LineIndex], 10, ColorTable[ColIndex] );
    OldPen = SelectObject(PrinterDC, NewPen);
    LineTo(PrinterDC, x2, PrinterHeight - y2);
    OldPen = SelectObject(PrinterDC, OldPen);
    DeleteObject( NewPen);
    return (0);
}


int WPRINT_Arc(int x0, int y0, int radius, double theta, double delta_theta)
     /*
      * Notes:
      *    Draws an arc of <radius> and center at (x0,y0) beginning at
      *    angle theta (in rad) and ending at theta + delta_theta
      */
{
    tpPrintData pd;
    HPEN    OldPen;
    HPEN    NewPen;
    int     left, right, top, bottom;
    int     xs, ys, xe, ye;
    int     yb;
    int     direction;
    int     ColIndex;
    double  r;
    double  dx0;
    double  dy0;

    if (!currentgraph) return 0;
    pd = pPrintData(currentgraph);
    if (!pd) return 0;

    x0 = x0 + xoffset;
    y0 = y0 + yoffset;

    ColIndex = pd->ColorIndex;

    direction = AD_COUNTERCLOCKWISE;
    if (delta_theta < 0) {
        theta = theta + delta_theta;
        delta_theta = - delta_theta;
        direction = AD_CLOCKWISE;
    }
    SetArcDirection( PrinterDC, direction);

    /* Geometrische Vorueberlegungen */
    yb      = PrinterHeight;
    left    = x0 - radius;
    right   = x0 + radius;
    top     = y0 + radius;
    bottom  = y0 - radius;

    r = radius;
    dx0 = x0;
    dy0 = y0;
    xs = (int)(dx0 + (r * cos(theta)));
    ys = (int)(dy0 + (r * sin(theta)));
    xe = (int)(dx0 + (r * cos(theta + delta_theta)));
    ye = (int)(dy0 + (r * sin(theta + delta_theta)));

    /* Zeichnen */
    NewPen = CreatePen( LineTable[pd->LineIndex], 10, ColorTable[ColIndex] );
    OldPen = SelectObject(PrinterDC, NewPen);
    Arc( PrinterDC, left, yb-top, right, yb-bottom, xs, yb-ys, xe, yb-ye);
    OldPen = SelectObject(PrinterDC, OldPen);
    DeleteObject( NewPen);

    return 0;
}

int WPRINT_Text(char * text, int x, int y, int degrees)
{
    tpPrintData pd;
    int ColIndex;
    NG_IGNORE(degrees);

    if (!currentgraph) return 0;
    pd = pPrintData(currentgraph);
    if (!pd) return 0;

    x = x + xoffset;
    y = y + yoffset;

    ColIndex = pd->ColorIndex;

    SetTextColor( PrinterDC, ColorTable[ColIndex]);

#ifdef EXT_ASC
    TextOut(PrinterDC, x, PrinterHeight - y - currentgraph->fontheight, text, (int)strlen(text));
#else
    /* wtext needs not to be NULL-terminated */
    const int n_byte_wide = (int) strlen(text);
    wchar_t * const wtext = TMALLOC(wchar_t, n_byte_wide);
    MultiByteToWideChar(CP_UTF8, 0, text, -1, wtext, n_byte_wide);
    TextOutW(PrinterDC, x, PrinterHeight - y - currentgraph->fontheight,
            wtext, n_byte_wide);
    txfree(wtext);
#endif
    return (0);
}


int WPRINT_DefineColor(int colorid, double red, double green, double blue)
{
    /* nix */
    NG_IGNORE(colorid);
    NG_IGNORE(red);
    NG_IGNORE(green);
    NG_IGNORE(blue);
    return 0;
}

int WPRINT_DefineLinestyle(int num, int mask)
{
    /* nix */
    NG_IGNORE(num);
    NG_IGNORE(mask);
    return 0;
}

int WPRINT_SetLinestyle(int style)
{
    tpPrintData pd;
    if (!currentgraph) {
        return 0;
    }
    pd = pPrintData(currentgraph);
    if (!pd) {
        return 0;
    }

    pd->LineIndex = style % NumLines;
    return 0;
}

int WPRINT_SetColor(int color)
{
    tpPrintData pd;
    if (!currentgraph) {
        return 0;
    }
    pd = pPrintData(currentgraph);
    if (!pd) {
        return 0;
    }

    pd->ColorIndex = color;
    return 0;
}

int WPRINT_Update(void)
{
    return 0;
}

int WPRINT_DiagramReady(void)
{
    return 0;
}

#endif /* HAS_WINGUI */

