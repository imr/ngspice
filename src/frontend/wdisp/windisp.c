/*
 * Frame buffer for the IDM PC using MS Windows
 * Wolfgang Muees 27.10.97
 * Holger Vogt  07.12.01
 */

#define STRICT
#include "ngspice.h"

#ifdef HAS_WINDOWS

#include "graph.h"
#include "ftedev.h"
#include "ftedbgra.h"

/*
 * The ngspice.h file included above defines BOOLEAN (via bool.h) and this
 * clashes with the definition obtained from windows.h (via winnt.h).
 * However, BOOLEAN is not used by this file so we can work round this problem
 * by undefining BOOLEAN before including windows.h
 * SJB - May 2005
 */
#undef BOOLEAN

#pragma warn -dup 		// wegen Redefinition von NUMCOLORS
#include <windows.h>
#include <windowsx.h>
#include "suffix.h"
#pragma hdrstop

// Typen
typedef struct {									// Extra Fensterdaten
	HWND	wnd;		// Fenster
	HDC		hDC;		// Device context des Fensters
	RECT    Area;		// Zeichenflaeche
	int		ColorIndex;	// Index auf die akt. Farbe
	int		PaintFlag;	// 1 bei WM_PAINT
	int		FirstFlag;	// 1 vor dem ersten Update
} tWindowData;
typedef tWindowData * tpWindowData;				// Zeiger darauf
#define pWindowData(g) ((tpWindowData)(g->devdep))

// Forwards
LRESULT CALLBACK PlotWindowProc( HWND hwnd, 		// Fensterprozedur
	UINT uMsg, WPARAM wParam, LPARAM lParam);

// externals
extern HINSTANCE		hInst;						// Instanz der Applikation
extern int				WinLineWidth;				// Breite des Textfensters
void WPRINT_PrintInit( HWND hwnd);					// Windows Drucker Init
void WaitForIdle(void);							// Warte, bis keine Events da

// lokale Variablen
static int			 	IsRegistered = 0;        	// 1 wenn Fensterkl. reg.
#define NumWinColors 	23       					// vordef. Farben
static COLORREF 		ColorTable[NumWinColors];	// Speicher fuer die Farben
static char *			WindowName = "Spice Plot";  // Fenstername
static WNDCLASS 		TheWndClass;				// Plot-Fensterklasse
static HFONT			PlotFont;					// Font-Merker
#define 				ID_DRUCKEN 	    0xEFF0		// System-Menu: drucken
#define				ID_DRUCKEINR    0xEFE0		// System-Menu: Druckereinrichtung
static const int		ID_MASK		  = 0xFFF0;		// System-Menu: Maske
static char *			STR_DRUCKEN   = "Drucken...";	// System-Menu-Strings
static char *			STR_DRUCKEINR = "Druckereinrichtung...";

/******************************************************************************
WIN_Init() stellt die Verbindung zur Grafik her. Dazu gehoert die Feststellung
von
	dispdev->numlinestyles		(bei Farbschirmen == 1)
	dispdev->numcolors
	dispdev->width              (vorlaeufig, Bildschirmbreite)
	dispdev->height             (vorlaeufig, Bildschirmhoehe)

WIN_Init() gibt 0 zurueck, falls kein Fehler auftrat.

WIN_Init() macht noch kein Fenster auf, dies geschieht erst in WIN_NewViewport()
******************************************************************************/

int WIN_Init( )
{
	// Initialisierungen des Display-Descriptors
	dispdev->width         = GetSystemMetrics( SM_CXSCREEN);
	dispdev->height        = GetSystemMetrics( SM_CYSCREEN);
	dispdev->numlinestyles = 5;	// siehe Auswirkungen in WinPrint!
	dispdev->numcolors     = NumWinColors;

	// nur beim ersten Mal:
	if (!IsRegistered) {

		// Farben initialisieren
		ColorTable[0] = RGB(  0,  0,  0);	// Schwarz 	= Hintergrund
		ColorTable[1] = RGB(255,255,255);	// Weisz 	= Beschriftung und Gitter
		ColorTable[2] = RGB(  0,255,  0);   // Gruen		= erste Linie
		ColorTable[3] = RGB(255,  0,  0);   // Rot
		ColorTable[4] = RGB(  0,  0,255);   // Blau
		ColorTable[5] = RGB(255,255,  0);   // Gelb
		ColorTable[6] = RGB(255,  0,255);   // Violett
		ColorTable[7] = RGB(  0,255,255);   // Azur
		ColorTable[8] = RGB(255,128,  0);   // Orange
		ColorTable[9] = RGB(128, 64,  0);   // braun
		ColorTable[10]= RGB(128,  0,255);   // Hellviolett
		ColorTable[11]= RGB(255,128,128);   // Rosa
		// 2. Farb-Bank (mit anderem Linientyp
		ColorTable[12]= RGB(255,255,255);	// Weisz
		ColorTable[13]= RGB(  0,255,  0);   // Gruen
		ColorTable[14]= RGB(255,  0,  0);   // Rot
		ColorTable[15]= RGB(  0,  0,255);   // Blau
		ColorTable[16]= RGB(255,255,  0);   // Gelb
		ColorTable[17]= RGB(255,  0,255);   // Violett
		ColorTable[18]= RGB(  0,255,255);   // Azur
		ColorTable[19]= RGB(255,128,  0);   // Orange
		ColorTable[20]= RGB(128, 64,  0);   // braun
		ColorTable[21]= RGB(128,  0,255);   // Hellviolett
		ColorTable[22]= RGB(255,128,128);   // Rosa

		// Ansii fixed font
		PlotFont = GetStockFont( ANSI_FIXED_FONT);

		// Fensterklasse registrieren
		TheWndClass.lpszClassName 	= WindowName;
		TheWndClass.hInstance     	= hInst;
		TheWndClass.lpfnWndProc   	= PlotWindowProc;
		TheWndClass.style 			= CS_OWNDC | CS_HREDRAW | CS_VREDRAW;
		TheWndClass.lpszMenuName  	= NULL;
		TheWndClass.hCursor       	= LoadCursor(NULL, IDC_ARROW);
		TheWndClass.hbrBackground   = GetStockObject( BLACK_BRUSH);
		TheWndClass.hIcon         	= LoadIcon(hInst, MAKEINTRESOURCE(2));
		TheWndClass.cbClsExtra    	= 0;
		TheWndClass.cbWndExtra    	= sizeof(GRAPH *);
		if (!RegisterClass(&TheWndClass)) return 1;
	}
	IsRegistered = 1;

	// fertig
	return (0);
}

// Zeiger auf den Graphen gewinnen
// (wird an das Fenster angehaengt)
static GRAPH * pGraph( HWND hwnd)
{
	return (GRAPH *) GetWindowLong( hwnd, 0);
}

// Linientyp zurueckgeben zum Zeichnen
static int LType( int ColorIndex)
{
	if (ColorIndex >= 12)
		return PS_DOT;
	else
		return PS_SOLID;
}
 
// Drucke ein Plotfenster
// Aufruf durch SystemMenue / Drucken
LRESULT PrintPlot( HWND hwnd)
{
	GRAPH * graph;
	GRAPH * temp;

	// Zeiger auf die Grafik holen
	graph = pGraph( hwnd);
	if (!graph) return 0;

	// Umschalten auf den Drucker
	// (hat WPRINT_Init() zur Folge)
	if (DevSwitch("WinPrint")) return 0;

	// Cursor = warten
	SetCursor( LoadCursor( NULL, IDC_WAIT));

	// Graphen kopieren
	temp = CopyGraph(graph);
	if (!temp) goto PrintEND;

	// in die Kopie die neuen Daten des Druckers einspeisen
	if (NewViewport(temp)) goto PrintEND2;

	// Lage des Gitters korrigieren (Kopie aus gr_init)
	temp->viewportxoff = temp->fontwidth  * 8;
	temp->viewportyoff = temp->fontheight * 4;

	// dies druckt den Graphen
	gr_resize(temp);

PrintEND2:
	// temp. Graphen loeschen
	DestroyGraph(temp->graphid);

PrintEND:
	// zurueckschalten auf den Bildschirm
	DevSwitch(NULL);

	// Cursor = normal
	SetCursor( LoadCursor( NULL, IDC_ARROW));

	return 0;
}

// Druckerinitialisierung
LRESULT PrintInit( HWND hwnd)
{
	// weitergeben an das Drucker-Modul
	WPRINT_PrintInit(hwnd);
	return 0;
}

// Fensterprozedur
LRESULT CALLBACK PlotWindowProc( HWND hwnd,
	UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg) {
	case WM_SYSCOMMAND:
		{
			// Kommando testen
			int cmd = wParam & ID_MASK;
			switch(cmd) {
			case ID_DRUCKEN:		return PrintPlot( hwnd);
			case ID_DRUCKEINR:      return PrintInit( hwnd);
			}
		}
		goto WIN_DEFAULT;

	case WM_CLOSE:	// Fenster schlieszen
		{
			GRAPH * g = pGraph( hwnd);
			if (g)
				DestroyGraph(g->graphid);
		}
		goto WIN_DEFAULT;

	case WM_PAINT:	// Fenster neuzeichnen (z.B. nach Resize)
		{
			PAINTSTRUCT ps;
			GRAPH * g;
			tpWindowData wd;
			HDC saveDC;		// der DC aus BeginPaint ist anders...
			HDC newDC;

			// muss passieren
			newDC = BeginPaint( hwnd, &ps);
			g = pGraph( hwnd);
			if (g) {
				wd = pWindowData(g);
				if (wd) {
					if (!wd->PaintFlag && !wd->FirstFlag) {
						// rekursiven Aufruf verhindern
						wd->PaintFlag = 1;
						// Fenstermasze holen
						GetClientRect( hwnd, &(wd->Area));
						g->absolute.width  = wd->Area.right;
						g->absolute.height = wd->Area.bottom;
						// DC umschalten
						saveDC = wd->hDC;
						wd->hDC = newDC;
						// neu zeichnen
						gr_resize(g);
						// DC umschalten
                       wd->hDC = saveDC;
						// fertig
						wd->PaintFlag = 0;
					}
				 }
			}
			// beenden
			EndPaint( hwnd, &ps);
		}
		return 0;

	default:
WIN_DEFAULT:
		return DefWindowProc( hwnd, uMsg, wParam, lParam);
	}
}


/******************************************************************************
 WIN_NewViewport() erstellt ein neues Fenster mit einem Graphen drin.

 WIN_NewViewport() gibt 0 zurueck, falls erfolgreich

******************************************************************************/

int WIN_NewViewport( GRAPH * graph)
{
	int 			i;
	HWND 			window;
	HDC 			dc;
	HDC 			textDC;
	HFONT 			font;
	TEXTMETRIC 		tm;
	tpWindowData 	wd;
	HMENU			sysmenu;

	// Parameter testen
	if (!graph) return 1;

	// Initialisiere, falls noch nicht geschehen
	if (WIN_Init() != 0) {
		externalerror("Can't initialize GDI.");
		return(1);
	}

	// Device dep. Info allocieren
	wd = calloc(1, sizeof(tWindowData));
	if (!wd) return 1;
	graph->devdep = (char *)wd;

	// Create the window
	i = GetSystemMetrics( SM_CYSCREEN) / 3;
	window = CreateWindow( WindowName, graph->plotname, WS_OVERLAPPEDWINDOW,
		0, 0, WinLineWidth, i * 2 - 22, NULL, NULL, hInst, NULL);
	if (!window) return 1;
	wd->wnd = window;
	SetWindowLong( window, 0, (long)graph);

	// Zeige das Fenster
	ShowWindow( window, SW_SHOWNORMAL);

	// Hole die Masze
	GetClientRect( window, &(wd->Area));

	// Hole den DC
	dc = GetDC( window);
	wd->hDC = dc;

	// Setze den Color-Index
	wd->ColorIndex = 0;

	// noch kein Zeichnen
	wd->PaintFlag = 0;
	wd->FirstFlag = 1;

	// System-Menu modifizieren
	sysmenu = GetSystemMenu( window, FALSE);
	AppendMenu( sysmenu, MF_SEPARATOR, 0, NULL);
	AppendMenu( sysmenu, MF_STRING, ID_DRUCKEN,   STR_DRUCKEN);
	AppendMenu( sysmenu, MF_STRING, ID_DRUCKEINR, STR_DRUCKEINR);

	// Default-Parameter des DC setzen
	SetBkColor( dc, ColorTable[0]);
	SetBkMode(  dc, TRANSPARENT );

	// Font setzen
	SelectObject( dc, PlotFont);

	// Font-Parameter abfragen
	if (GetTextMetrics( dc, &tm)) {
		graph->fontheight = tm.tmHeight;
		graph->fontwidth  = tm.tmAveCharWidth;
	}

	// Viewport-Parameter setzen
	graph->viewport.height 	= wd->Area.bottom;
	graph->viewport.width  	= wd->Area.right;

	// Absolut-Parameter setzen
	graph->absolute.xpos 	= 0;
	graph->absolute.ypos 	= 0;
	graph->absolute.width 	= wd->Area.right;
	graph->absolute.height 	= wd->Area.bottom;

	// Warten, bis das Fenster wirklich da ist
   WaitForIdle();

	// fertig
	return(0);
}

/******************************************************************************
WIN_Close ist eigentlich das Gegenstueck zu WIN_Init. Dummerweise kann es
passieren, dasz (waehrend gerade ein Plot dargestellt wird) WIN_Close aufgerufen
wird, um auf einen Drucker umzuschalten. Deswegen darf WIN_Close nichts machen,
sondern das Aufloesen der Strukturen erfolgt bei Programmende.
******************************************************************************/

int WIN_Close()
{
	return (0);
}

void RealClose(void)
{
	// Fensterklasse loeschen
	if (IsRegistered) {
		if (TheWndClass.hIcon) {
			DestroyIcon( TheWndClass.hIcon);
			TheWndClass.hIcon = NULL;
		}
		UnregisterClass( WindowName, hInst);
		IsRegistered = FALSE;
	}
}
#pragma exit RealClose

int WIN_Clear()
{
	tpWindowData wd;
	if (!currentgraph) return 0;
	wd = pWindowData(currentgraph);
	if (!wd) return 0;

	// das macht das Fenster selbst
	if (!wd->PaintFlag)	// bei WM_PAINT unnoetig
		SendMessage( wd->wnd, WM_ERASEBKGND, (WPARAM) wd->hDC, 0);

	return 0;
}


int WIN_DrawLine(int x1, int y1, int x2, int y2)
{
	tpWindowData wd;
	HPEN 		 OldPen;
	HPEN    	 NewPen;

	if (!currentgraph) return 0;
	wd = pWindowData(currentgraph);
	if (!wd) return 0;

	MoveToEx(wd->hDC, x1, wd->Area.bottom - y1, NULL);
	NewPen = CreatePen( LType(wd->ColorIndex), 0, ColorTable[wd->ColorIndex] );
	OldPen = SelectObject(wd->hDC, NewPen);
	LineTo(wd->hDC, x2, wd->Area.bottom - y2);
	OldPen = SelectObject(wd->hDC, OldPen);
	DeleteObject( NewPen);

	return (0);
}


int WIN_Arc(int x0, int y0, int radius, double theta1, double theta2)
	 /*
	  * Notes:
	  *    Draws an arc of <radius> and center at (x0,y0) beginning at
	  *    angle theta1 (in rad) and ending at theta2
	  */
{
	tpWindowData wd;
	HPEN   	OldPen;
	HPEN   	NewPen;
	int		left, right, top, bottom;
	int		xs, ys, xe, ye;
	int    	yb;
	int		direction;
	double	temp;
	double	r;
	double  dx0;
	double	dy0;

	if (!currentgraph) return 0;
	wd = pWindowData(currentgraph);
	if (!wd) return 0;

	direction = AD_COUNTERCLOCKWISE;
	if (theta1 > theta2) {
		temp   = theta1;
		theta1 = theta2;
		theta2 = temp;
		direction = AD_CLOCKWISE;
	}
	SetArcDirection( wd->hDC, direction);

	// Geometrische Vorueberlegungen
	yb   	= wd->Area.bottom;
	left 	= x0 - radius;
	right 	= x0 + radius;
	top 	= y0 + radius;
	bottom 	= y0 - radius;

	r = radius;
	dx0 = x0;
	dy0 = y0;
	xs = (dx0 + (r * cos(theta1)));
	ys = (dy0 + (r * sin(theta1)));
	xe = (dx0 + (r * cos(theta2)));
	ye = (dy0 + (r * sin(theta2)));

	// Zeichnen
	NewPen = CreatePen( LType(wd->ColorIndex), 0, ColorTable[wd->ColorIndex] );
	OldPen = SelectObject(wd->hDC, NewPen);
	Arc( wd->hDC, left, yb-top, right, yb-bottom, xs, yb-ys, xe, yb-ye);
	OldPen = SelectObject(wd->hDC, OldPen);
	DeleteObject( NewPen);

	return 0;
}

int WIN_Text( char * text, int x, int y, int degrees)
{
	tpWindowData wd;
	if (!currentgraph) return 0;
	wd = pWindowData(currentgraph);
	if (!wd) return 0;

	SetTextColor( wd->hDC, ColorTable[wd->ColorIndex]);
	TextOut( wd->hDC, x, wd->Area.bottom - y - currentgraph->fontheight, text, strlen(text));

	return (0);
}


int WIN_DefineColor(int red, int green, int blue, int num)
{
	// nix
	return (0);
}

int WIN_DefineLinestyle(int num, int mask)
{
	// nix
	return (0);
}

int WIN_SetLinestyle(int style)
{
	// nix
	return (0);
}

int WIN_SetColor( int color)
{
	tpWindowData wd;
	if (!currentgraph) return 0;
	wd = pWindowData(currentgraph);
	if (!wd) return 0;

	wd->ColorIndex = color % NumWinColors;

	return (0);
}

int WIN_Update()
{
	tpWindowData wd;
	if (!currentgraph) return 0;
	wd = pWindowData(currentgraph);
	if (!wd) return 0;

	// Nach dem ersten absolvieren von Update() werden durch
	// FirstFlag wieder WM_PAINT-Botschaften bearbeitet.
	// Dies verhindert doppeltes Zeichnen beim Darstellen des Fensters.
	wd->FirstFlag = 0;
	return 0;
}

int WIN_DiagramReady()
{
	return 0;
}

#endif /* HAS_WINDOWS */
