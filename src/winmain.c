/* Hauptprogramm fuer Spice 3F5 unter Windows95
	Autor: Wolfgang Muees
	Stand: 28.10.97
	Autor: Holger Vogt
	Stand: 01.05.2000
	Stand: 12.12.2001
 $Id$
*/
#include "config.h"
#ifdef HAS_WINDOWS

#define STRICT				// strikte Typpruefung
#define WIN32_LEAN_AND_MEAN
#include <windows.h>    		// normale Windows-Aufrufe
#include <windowsx.h>			// Win32 Message Cracker
//#include <ctl3d.h>			// 3D-Steuerelemente
#include <stdio.h>			// sprintf und co
#include <stdlib.h>			// exit-codes
#include <stdarg.h>			// var. argumente
#include <assert.h>			// assert-macro
// #include <shellapi.h>		// shellexecute
//#include <dir.h>			// Verzeichnis-Funktionen hvogt 09.12.01
//#include <dos.h>			// argc, argv


#include <errno.h>
#include <dirent.h>

#ifdef _MSC_VER
/* Microsoft VC++ specific stuff */
#pragma hdrstop
#endif /* _MSC_VER */

#include <signal.h>
#include <ctype.h>

#include "bool.h"			// bool defined as unsigned char
/* Constants */
#define TBufSize 2048			// Groesze des Textbuffers
#define CR VK_RETURN			// Carriage Return
#define LF 10				// Line Feed
#define SE 0				// String termination
#define BorderSize 8			// Umrandung des Stringfeldes
#define SBufSize 100			// Groesze des Stringbuffers
#define IOBufSize 4096			// Groesze des printf-Buffers
#define HistSize 20			// Zeilen History-Buffer
#define StatusHeight 25			// Hoehe des Status Bars
#define StatusFrame 2			// Abstand Statusbar / StatusElement
#define StatusElHeight (StatusHeight - 2 * StatusFrame)
#define SourceLength 400		// Platz fuer Source File Name
#define AnalyseLength 100		// Platz fuer Analyse

/* Types */
typedef char SBufLine[SBufSize+1];	// Eingabezeile

/* Global variables */
HINSTANCE 		hInst;			/* Application instance */
int 			WinLineWidth = 640;	/* Window width */
HWND       	hwMain;				/* Main Window of the application */
HWND       	twText;				/* Text window */
HWND			swString;		// Eingabezeile
HWND			hwStatus;		// Status-Balken
HWND			hwSource;		// Anzeige des Source-Namens
HWND			hwAnalyse;		// Anzeige des Analyse-Fensters
static int		nReturnCode	= 0;	// Rueckgabewert von WinMain
static int		nShowState;		// Anzeigemodus des Hauptfensters
static WNDCLASS hwMainClass;			/* Class definition for the main window */
static LPCTSTR	hwClassName  = "SPICE_TEXT_WND";/* Class name of the main window */
static LPCTSTR hwWindowName = PACKAGE_STRING;	/* main window displayed name */
static WNDCLASS twTextClass;			/* Class definition for the text box */
static LPCTSTR twClassName  = "SPICE_TEXT_BOX";	/* Class name for the text box */
static LPCTSTR twWindowName = "TextOut";	/* text box name */
static size_t	TBufEnd = 0;			// Zeigt auf \0
static char TBuffer [TBufSize+1];		// Textbuffer
static SBufLine SBuffer;			// Eingabebuffer
static WNDCLASS swStringClass;			// Klassendefinition des Stringfensters
static LPCTSTR swClassName  = "SPICE_STR_IN";	// Klassenname der Texteingabe
static LPCTSTR swWindowName = "StringIn";	// Name des Fensters
static char CRLF [] = { CR, LF, SE} ;		// CR/LF
static WNDCLASS hwElementClass;			// Klassendefinition der Statusanzeigen
static LPCTSTR hwElementClassName = "ElementClass";
static LPCTSTR hwSourceWindowName = "SourceDisplay";
static LPCTSTR hwAnalyseWindowName = "AnalyseDisplay";
static int RowHeight = 16;		// Hoehe einer Textzeile
static int LineHeight = 25;		// Hoehe der Eingabezeile
static int VisibleRows = 10;		// Anzahl der sichtbaren Zeilen im Textfenster
static BOOL DoUpdate = FALSE;	// Textfenster updaten
static WNDPROC swProc = NULL;		// originale Stringfenster-Prozedur
static WNDPROC twProc = NULL;		// originale Textfenster-Prozedur
static SBufLine HistBuffer[HistSize]; 	// History-Buffer fuers Stringfenster
static int HistIndex = 0;		// History-Verwaltung
static int HistPtr   = 0;		// History-Verwaltung

extern bool oflag;			// falls 1, Output ueber stdout in File umgeleitet
extern FILE *flogp;  // siehe xmain.c, hvogt 14.6.2000
//int argc; 
//char *argv[];
// Forward-Definition von main()
int xmain( int argc, char * argv[]/*, char * env[]*/);
// forward der Update-Funktion
void DisplayText( void);


// --------------------------<History-Verwaltung>------------------------------

// Alle Puffer loeschen und Zeiger auf den Anfang setzen
void HistoryInit(void)
{
	int i;
	HistIndex = 0;
	HistPtr = 0;
	for ( i = 0; i < HistSize; i++)
		HistBuffer[i][0] = SE;
}

// Erste Zeile des Buffers loeschen; alles rueckt auf
void HistoryScroll(void)
{
	memmove( &(HistBuffer[0]), &(HistBuffer[1]), sizeof(SBufLine) * (HistSize-1));
	HistBuffer[HistSize-1][0] = SE;
	if (HistIndex) HistIndex--;
	if (HistPtr)   HistPtr--;
}

// Neue Eingabezeile in den History-Buffer schreiben
void HistoryEnter( char * newLine)
{
	if (!newLine || !*newLine) return;
	if (HistPtr == HistSize) HistoryScroll();
	strcpy( HistBuffer[HistPtr], newLine);
	HistPtr++;
	HistIndex = HistPtr;
}

// Mit dem Index eine Zeile zurueckgehen und den dort stehenden Eintrag zurueckgeben
char * HistoryGetPrev(void)
{
	if (HistIndex) HistIndex--;
	return &(HistBuffer[HistIndex][0]);
}

// Mit dem Index eine Zeile vorgehen und den dort stehenden Eintrag zurueckgeben
char * HistoryGetNext(void)
{
	if (HistIndex < HistPtr) HistIndex++;
	if (HistIndex == HistPtr) HistIndex--;
	return &(HistBuffer[HistIndex][0]);
}

// ---------------------------<Message Handling>-------------------------------

// Warte, bis keine Messages mehr zu bearbeiten sind
void WaitForIdle(void)
{
	MSG m;
	// arbeite alle Nachrichten ab
	while ( PeekMessage(  &m, NULL, 0, 0, PM_REMOVE)) {
		TranslateMessage( &m);
		DispatchMessage(  &m);
	}
}

// ---------------------------<Message Handling>-------------------------------

// Warte, bis keine Messages mehr zu bearbeiten sind,
// dann warte auf neue Message (Input handling ohne Dauerloop)
void WaitForMessage(void)
{
	MSG m;
	// arbeite alle Nachrichten ab
	while ( PeekMessage(  &m, NULL, 0, 0, PM_REMOVE)) {
		TranslateMessage( &m);
		DispatchMessage(  &m);
	}
	WaitMessage();
}

// -----------------------------<Stringfenster>--------------------------------

// Loeschen des Stringfensters
void ClearInput(void)
{
	// Darstellen
	Edit_SetText( swString, "");
}

// ---------------------------<SourceFile-Fenster>-----------------------------

// Neuer Text ins Sourcefile-Fenster
void SetSource( char * Name)
{
	if (hwSource) {
		SetWindowText( hwSource, Name);
		InvalidateRgn( hwSource, NULL, TRUE);
	}
}

// ------------------------------<Analyse-Fenster>-----------------------------

// Neuer Text ins Analysefenster
static int OldPercent = -2;
void SetAnalyse( char * Analyse, int Percent)
{
	char s[128];

	if (Percent == OldPercent) return;
   OldPercent = Percent;
	if (hwAnalyse) {
		if (Percent < 0)
			sprintf( s, "--ready--");
		else
			sprintf( s, "%s : %3u%%", Analyse, Percent);
		SetWindowText( hwAnalyse, s);
		InvalidateRgn( hwAnalyse, NULL, TRUE);
       WaitForIdle();
	}
}

// ------------------------------<Textfenster>---------------------------------

// Anpassen des Scrollers im Textfenster
// Stellt gleichzeitig den Text neu dar
void AdjustScroller(void)
{
	int LineCount;
	int FirstLine;
	int MyFirstLine;
	LineCount = Edit_GetLineCount( twText);
	FirstLine = Edit_GetFirstVisibleLine( twText);
	MyFirstLine = LineCount - VisibleRows;
	if (MyFirstLine < 0 ) MyFirstLine = 0;
	Edit_Scroll( twText, MyFirstLine - FirstLine, 0);
	// Das wars
	DoUpdate = FALSE;
}

// Loeschen einer Zeile im Textbuffer
void _DeleteFirstLine(void)
{
	char * cp = strchr( TBuffer, LF);
	if (!cp) {
		// Buffer leeren
		TBufEnd = 0;
		TBuffer[TBufEnd] = SE;
		return;
	}
	cp++;
	TBufEnd -= cp - TBuffer;
	memmove( TBuffer, cp, TBufEnd);
	TBuffer[TBufEnd] = SE;
}

// Anfuegen eines chars an den TextBuffer
void AppendChar( char c)
{
	// Textbuffer nicht zu grosz werden lassen
	while ((TBufEnd+4) >= TBufSize)
		_DeleteFirstLine();
	// Zeichen anfuegen
	TBuffer[TBufEnd++] = c;
	TBuffer[TBufEnd] = SE;
	DoUpdate = TRUE;
	// Sobald eine Zeile zuende, im Textfenster anzeigen
	if (c == LF)
	   DisplayText();
}

// Anfuegen eines Strings an den TextBuffer
void AppendString( const char * Line)
{
	size_t i;
	if (!Line) return;

	// Zeilenlaenge bestimmen
	i = strlen(Line);
	// Textbuffer nicht zu grosz werden lassen
	while ((i+TBufEnd+3) >= TBufSize)
		_DeleteFirstLine();
	// Zeile dranhaengen
	strcpy( &TBuffer[TBufEnd], Line);
	TBufEnd += i;
	DoUpdate = TRUE;
}

// Text neu darstellen
void DisplayText( void)
{
	// Darstellen
	Edit_SetText( twText, TBuffer);
	// Scroller updaten, neuen Text darstellen
	AdjustScroller();
}
/*
// Anfuegen einer Zeile an den Textbuffer
void AppendLine( const char * Line)
{
	if (!Line) return;

	// String anhaengen
	AppendString( Line);

	// CRLF anhaengen
	AppendString( CRLF);
}
*/
// -----------------------------------<User-IO>--------------------------------

// Lese ein Zeichen ein
int w_getch(void)
{
	int c;

	// Sind noch Zeichen da?
	c = SBuffer[0];
	if (!c) {
		// Alte Informationen darstellen
		if (DoUpdate)
			DisplayText();
		// Focus setzen
		SetFocus( swString);
		// Cursor = normal
		SetCursor( LoadCursor( NULL, IDC_IBEAM));
		// Analyse ist fertig
       SetAnalyse( NULL, -1);
		// Warten auf die Eingabe
		do {
			WaitForMessage();
			c = SBuffer[0];
		} while ( !c );
		// Zeichen an die Ausgabe anhaengen
		AppendString( SBuffer);
		// Cursor = warten
		SetCursor( LoadCursor( NULL, IDC_WAIT));
	}
	// Zeichen abholen
	memmove( &SBuffer[0], &SBuffer[1], SBufSize);
	return c;
}

// Gebe ein Zeichen aus
int w_putch( int c)
{
	if (c)
		AppendChar( (char)c );
	return c;
}

// -------------------------------<Fensterprozeduren>--------------------------

// Hauptfenster veraendert seine Groesze
#ifdef _MSC_VER
/* Microsoft VC++ specific stuff */
#pragma warn -par
#endif /* _MSC_VER */
void Main_OnSize(HWND hwnd, UINT state, int cx, int cy)
{
	int h = cy - LineHeight - StatusHeight;

	// Textfenster expandieren
	MoveWindow( twText, 0, 0, cx, h , TRUE);
	VisibleRows = (h / RowHeight) -1;
	AdjustScroller();

	// Stringfenster expandieren
	MoveWindow( swString, 0, h, cx, LineHeight, TRUE);

	// StatusElemente expandieren
	h = cy - LineHeight + StatusFrame -1;
	MoveWindow( hwSource, StatusFrame, h, SourceLength, StatusElHeight, TRUE);
	MoveWindow( hwAnalyse, 3 * StatusFrame + SourceLength, h, AnalyseLength,
		StatusElHeight, TRUE);
}


// Schreibe einen Befehl in den Spice-Kommandobuffer
void PostSpiceCommand( const char * const cmd)
{
	strcpy( SBuffer, cmd);
	strcat( SBuffer, CRLF);
}

// HauptfensterProzedur
#ifdef _MSC_VER
/* Microsoft VC++ specific stuff */
#pragma warn -eff
#endif /* _MSC_VER */
LRESULT CALLBACK MainWindowProc( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
/*	UINT i; */
	
	switch (uMsg) {

	case WM_SYSCOLORCHANGE:
//		Ctl3dColorChange();
		goto DEFAULT_AFTER;

	case WM_CLOSE:
		// den Spice-Befehl zum Beenden des Programms in den Textbuffer schreiben
		PostSpiceCommand( "quit");
		// Unterbrechen, falls Simulation schon laeuft, 30.4.2000 hvogt 
		raise (SIGINT);   
		return 0;

/*	//gedacht fuer ctrl C , geht noch nicht
	case WM_KEYDOWN:
		i = (UINT) wParam;
		if ((i == 0x63) && (GetKeyState(VK_CONTROL) < 0)) {
		// Interrupt zum Unterbrechen (interaktiv) 
		// oder Beenden (Batch) des Programms ausloesen
		   raise (SIGINT);
		   return 0;
		}		*/
		
/*	//gedacht fuer ctrl C , geht noch nicht
	case WM_CHAR:
		i = (char) wParam;
		if ((i == "c") && (GetKeyState(VK_CONTROL) < 0)) {
		// Interrupt zum Unterbrechen (interaktiv) 
		// oder Beenden (Batch) des Programms ausloesen
		   raise (SIGINT);
		   return 0;
		}   */

	case WM_SIZE:
		HANDLE_WM_SIZE( hwnd, wParam, lParam, Main_OnSize);
		goto DEFAULT_AFTER;

	default:
DEFAULT_AFTER:
		return DefWindowProc( hwnd, uMsg, wParam, lParam);
	}
}

// StringfensterProzedur
LRESULT CALLBACK StringWindowProc( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	char c;
	UINT i;

	switch (uMsg) {

	case WM_KEYDOWN:
		i = (UINT) wParam;
		if ((i == VK_UP) || (i == VK_DOWN)) {
			// alten Text neu setzen
			SetWindowText( hwnd, i == VK_UP? HistoryGetPrev(): HistoryGetNext());
			// Cursor ans Ende der Zeile
			CallWindowProc( swProc, hwnd, uMsg, (WPARAM) VK_END, lParam);
			return 0;
		}
		if ( i == VK_ESCAPE) {
			ClearInput();
			return 0;
		}


		goto DEFAULT;

	case WM_CHAR:
			c = (char) wParam;
			if (c == CR) {
				GetWindowText( hwnd, SBuffer, SBufSize);
				HistoryEnter( SBuffer);
				strcat( SBuffer, CRLF);
				ClearInput();
				return 0;
			}
			if (c == VK_ESCAPE)
				return 0;
	default:
DEFAULT:
		return CallWindowProc( swProc, hwnd, uMsg, wParam, lParam);
	}
}

// TextfensterProzedur
LRESULT CALLBACK TextWindowProc( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	unsigned char c;
	UINT i;

	switch (uMsg) {

	case WM_KEYDOWN:
		i = (UINT) wParam;
		if ((i == VK_UP) || (i == VK_DOWN) || (i == VK_ESCAPE)) {
			// Leite um ins String-Fenster
			SetFocus( swString);
			return SendMessage( swString, uMsg, wParam, lParam);
		}
		goto DEFAULT_TEXT;

	case WM_CHAR:
		c = (unsigned char) wParam;
		if ((c == CR) || ( c >= ' ') || ( c == VK_ESCAPE)) {
			// Leite um ins String-Fenster
			SetFocus( swString);
			return SendMessage( swString, uMsg, wParam, lParam);
		}
	default:
DEFAULT_TEXT:
		return CallWindowProc( twProc, hwnd, uMsg, wParam, lParam);
	}
}


void Element_OnPaint(HWND hwnd)
{
	PAINTSTRUCT ps;
	RECT r;
	RECT s;
	HGDIOBJ o;
	char buffer[128];
	int i;

	// Vorbereiten
	HDC hdc = BeginPaint( hwnd, &ps);
	GetClientRect( hwnd, &r);

	// Rahmen zeichnen
	o = GetStockObject( GRAY_BRUSH);
	s.left  	= r.left;
	s.right		= r.right;
	s.top		= r.top;
	s.bottom	= r.top+1;
	FillRect( hdc, &s, o);

	s.right     = r.left+1;
	s.bottom	= r.bottom;
	FillRect( hdc, &s, o);

	o = GetStockObject( WHITE_BRUSH);
	s.right		= r.right;
	s.top		= r.bottom-1;
	FillRect( hdc, &s, o);

	s.left		= r.right-1;
	s.top		= r.top;
	FillRect( hdc, &s, o);

	// Inhalt zeichnen
	buffer[0] = '\0';
	i = GetWindowText( hwnd, buffer, 127);
	s.left 		= r.left+1;
	s.right		= r.right-1;
	s.top		= r.top+1;
	s.bottom	= r.bottom-1;
	o = GetStockObject( LTGRAY_BRUSH);
	FillRect( hdc, &s, o);
	SetBkMode( hdc, TRANSPARENT);
	ExtTextOut( hdc, s.left+1, s.top+1, ETO_CLIPPED, &s, buffer, i, NULL);

	// Ende
	EndPaint( hwnd, &ps);
}


// ElementfensterProzedur
LRESULT CALLBACK ElementWindowProc( HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
	switch (uMsg) {

	case WM_PAINT:
		HANDLE_WM_PAINT(hwnd, wParam, lParam, Element_OnPaint);
		return 0;

	default:
		return DefWindowProc( hwnd, uMsg, wParam, lParam);
	}
}


#define SPACE			' '
#define QUOTE			'\"'
#define DELIMITER		26		/* for the lack of anything better */
#define DELIMITERSTRING	"\26"

/*
	This function converts a string into an argc/argv represenation.
	INPUT:
		cmdline		-	a string
	OUTPUT:
		argc		-	the number of equivalent argv strings
					which is also the number of strings in argv
		argv		-	the argv given the input string which
					consists of seperate strings for each argument
	RETURNS:
		0  on success
		-1 on failure
*/
int MakeArgcArgv(char *cmdline,int *argc,char ***argv)
{
	char  *pC1;			/*	a temporary character pointer */
	char  *pC2;			/*	a temporary character pointer */
	char  *pWorkString=NULL;		/*	a working copy of cmdline */
	int    i;				/*	a loop counter */
	int    j;				/*	a loop counter */
	int    quoteflag=0;			/*	for the finite state machine parsing cmdline */
	int    numargs=1;			/*	the number of command line arguments, later
						copied to *argc */
	char **tmpargv;			/*	the temporary argv, later copied to *argv */
	int    status = ERROR_SUCCESS; 	/* status */
	char   buffer[MAX_PATH+1];


	/* make sure we aren't dealing with any NULL pointers */
	if (	(NULL == argc)
	|| (NULL == argv))
	{
		status = -1;
		goto outahere;
	}
	*argc = 0;		/* set the count to zero to start */
	*argv = NULL;	/* set the pointer to NULL to start */
	/*	if the string passed in was a NULL pointer, consider this
		to be an empty command line and give back only
		an argc of 1 and an argv[0] */
	if (NULL != cmdline)
	{
	/*	make a copy of the string so that we can modify it
			without messing up the original */
		pWorkString = strdup(cmdline);
		if (NULL == pWorkString)
			return -1; /* memory allocation error */
		/*	Now, to make sure we don't have any quoted arguments
			with spaces in them, replace all spaces except those
			between " marks with our own special delimiter for
			strtok */
		/* trim all the whitespace off the end of the string. */
		for (i=(signed)strlen(pWorkString)-1; i >=0; i--)
			if (isspace(pWorkString[i]))
				pWorkString[i] = '\0';
			else
				break;
		/*	If we still have a string left, parse it for all
			the arguments. */
		if (strlen(pWorkString))
		{
			/*	This could probably be done with strtok as well
				but strtok is destructive if I wanted to look for " \""
				and I couldn't tell what delimiter that I had bumped
				against */
			for (i=0; i < (signed)strlen(pWorkString); i++)
			{
				switch (pWorkString[i])
				{
				case SPACE:
					if (!quoteflag)
					{
						pWorkString[i] = DELIMITER;  /* change space to delimiter */
						numargs++;
					}
					break;
				case QUOTE:
					quoteflag = !quoteflag; /* turns on and off as we pass quotes */
					break;
				}
			}
			/*	Now, we should have ctrl-Zs everywhere that
				there used to be a space not protected by
				quote marks.  We should also have the number
				of command line arguments that were in the
				command line (not including argv[0] which should
				be the program name).  We should add one more
				to numargs to take into account argv[0].  */
			numargs++;
		}
	}
	/* malloc an argv */
	tmpargv = (char**)malloc(numargs * sizeof(char *));
	if (NULL == tmpargv)
	{
		status = -1;
		goto outahere;
	}
	/*	you can put your program name here or find an API to give it
		to you if you want.  I am just giving a name to fill the space */
	GetModuleFileName(NULL, buffer, sizeof(buffer));


//	tmpargv[0] = strdup("ngspice");
	tmpargv[0] = buffer;
	
	pC1 = NULL;
	/*	Now actually strdup all the arguments out of the sting
		and store them in the argv */
	for (i=1; i < numargs; i++)
	{
		if (NULL == pC1)
			pC1 = pWorkString;
		for (j=0; j < (signed)strlen(pC1); j++)
		{
			if (DELIMITER == pC1[j])
			{
				pC1[j] = '\0';
				pC2    = &pC1[j+1];
				break;
			}
		}
		tmpargv[i] = strdup(pC1);
		if (NULL == tmpargv[i])
		{
			status = -1;
			goto outahere;
		}
		pC1 = pC2;
	}
	/*	copy the working values over to the arguments */
	*argc = numargs;
	*argv = tmpargv;
outahere:
	/*	free the working string if one was allocated */
	if (pWorkString)
		free(pWorkString);
	/* return status */
	return status;
}



/* Main entry point for our Windows application */
#ifdef _MSC_VER
/* Microsoft VC++ specific stuff */
#pragma warn -par
#endif /* _MSC_VER */
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int nCmdShow)
{
	int i;
	int status;
	
	int argc;
	char **argv;

	// globale Variablen fuellen
	hInst = hInstance;
	nShowState = nCmdShow;

	// Textbuffer initialisieren
	TBufEnd = 0;
	TBuffer[TBufEnd] = SE;
	SBuffer[0] = SE;
	HistoryInit();

	// 3D-Elemente registrieren
//	Ctl3dRegister( hInstance);
//	Ctl3dAutoSubclass( hInstance);

	// Hauptfensterklasse definieren
	hwMainClass.style 			= CS_HREDRAW | CS_VREDRAW;
	hwMainClass.lpfnWndProc		= MainWindowProc;
	hwMainClass.cbClsExtra		= 0;
	hwMainClass.cbWndExtra		= 0;
	hwMainClass.hInstance		= hInst;
	hwMainClass.hIcon 			= LoadIcon( hInst, MAKEINTRESOURCE(1));
	hwMainClass.hCursor			= LoadCursor( NULL, IDC_ARROW);
	hwMainClass.hbrBackground 	= GetStockObject( LTGRAY_BRUSH);
	hwMainClass.lpszMenuName 	= NULL;
	hwMainClass.lpszClassName 	= hwClassName;
	if (!RegisterClass( &hwMainClass)) goto THE_END;

	// Textfensterklasse definieren
	if (!GetClassInfo( NULL, "EDIT", &twTextClass)) goto THE_END;
	twProc = twTextClass.lpfnWndProc;
	twTextClass.lpfnWndProc		= TextWindowProc;
	twTextClass.hInstance		= hInst;
	twTextClass.lpszMenuName 	= NULL;
	twTextClass.lpszClassName 	= twClassName;
	if (!RegisterClass( &twTextClass)) goto THE_END;

	// Stringfensterklasse definieren
	if (!GetClassInfo( NULL, "EDIT", &swStringClass)) goto THE_END;
	swProc = swStringClass.lpfnWndProc;
	swStringClass.lpfnWndProc   = StringWindowProc;
	swStringClass.hInstance		= hInst;
	swStringClass.lpszMenuName 	= NULL;
	swStringClass.lpszClassName = swClassName;
	if (!RegisterClass( &swStringClass)) goto THE_END;

	// StatusElementklasse definieren
	hwElementClass.style 			= CS_HREDRAW | CS_VREDRAW;
	hwElementClass.lpfnWndProc		= ElementWindowProc;
	hwElementClass.cbClsExtra		= 0;
	hwElementClass.cbWndExtra		= 0;
	hwElementClass.hInstance		= hInst;
	hwElementClass.hIcon 			= NULL;
	hwElementClass.hCursor			= LoadCursor( NULL, IDC_ARROW);
	hwElementClass.hbrBackground 	= GetStockObject( LTGRAY_BRUSH);
	hwElementClass.lpszMenuName 	= NULL;
	hwElementClass.lpszClassName 	= hwElementClassName;
	if (!RegisterClass( &hwElementClass)) goto THE_END;

	// Hauptfenster kreieren
	i = GetSystemMetrics( SM_CYSCREEN) / 3;
	hwMain = CreateWindow( hwClassName, hwWindowName, WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN,
		0, i * 2, GetSystemMetrics( SM_CXSCREEN), i, NULL, NULL, hInst, NULL);
	if (!hwMain) goto THE_END;

	// Textfenster kreieren
	twText = CreateWindowEx(WS_EX_NOPARENTNOTIFY, twClassName, twWindowName,
		ES_LEFT | ES_MULTILINE | ES_READONLY | WS_CHILD | WS_BORDER | WS_VSCROLL,
		20,20,300,100, hwMain, NULL, hInst, NULL);
	if (!twText) goto THE_END;
	// Ansii fixed font
	{
		HDC textDC;
		HFONT font;
		TEXTMETRIC tm;
		font = GetStockFont( ANSI_FIXED_FONT);
		SetWindowFont( twText, font, FALSE);
		textDC = GetDC( twText);
		if (textDC) {
			SelectObject( textDC, font);
			if (GetTextMetrics( textDC, &tm)) {
				RowHeight = tm.tmHeight;
				WinLineWidth = 90 * tm.tmAveCharWidth;
			}
			ReleaseDC( twText, textDC);
		}
	}

	// Stringfenster kreieren
	swString = CreateWindowEx(WS_EX_NOPARENTNOTIFY, swClassName, swWindowName,
		ES_LEFT | WS_CHILD | WS_BORDER, 20,20,300,100, hwMain, NULL, hInst, NULL);
	if (!swString) goto THE_END;
	{
		HDC stringDC;
		TEXTMETRIC tm;
		stringDC = GetDC( swString);
		if (stringDC) {
			if (GetTextMetrics( stringDC, &tm))
				LineHeight = tm.tmHeight + tm.tmExternalLeading + BorderSize;
			ReleaseDC( swString, stringDC);
		}
	}

	// Sourcefenster kreieren
	hwSource = CreateWindowEx(WS_EX_NOPARENTNOTIFY, hwElementClassName,
		hwSourceWindowName, WS_CHILD, 0,0, SourceLength, StatusElHeight, hwMain,
		NULL, hInst, NULL);
	if (!hwSource) goto THE_END;


	// Analysefenster kreieren
	hwAnalyse = CreateWindowEx(WS_EX_NOPARENTNOTIFY, hwElementClassName,
		hwAnalyseWindowName, WS_CHILD, 0,0, AnalyseLength, StatusElHeight, hwMain,
		NULL, hInst, NULL);
	if (!hwAnalyse) goto THE_END;


	// Hauptfenster mit Unterfenstern sichtbar machen
	if (WinLineWidth > 600) WinLineWidth = 600;
	MoveWindow( hwMain, 0, ((GetSystemMetrics( SM_CYSCREEN) / 3) * 2 - 22), WinLineWidth,
				GetSystemMetrics( SM_CYSCREEN) / 3, FALSE);
	ShowWindow( hwMain,   nShowState);
	ShowWindow( twText,   SW_SHOWNORMAL);
	ShowWindow( swString, SW_SHOWNORMAL);
	ShowWindow( hwSource, SW_SHOWNORMAL);
	ShowWindow( hwAnalyse,SW_SHOWNORMAL);
	ClearInput();
	DisplayText();
	SetSource( "");
//   SetAnalyse( NULL, -1);
        SetAnalyse(" ", 0);
	UpdateWindow( hwMain);
	SetFocus( swString);

        status = MakeArgcArgv(lpszCmdLine,&argc,&argv);


	// Warten, bis alles klar ist
	WaitForIdle();

	// Ab nach main()
	nReturnCode = xmain(argc, argv/*, _environ*/);
	

THE_END:
	// 3D abschalten
//	Ctl3dUnregister( hInstance);

	// terminate
	return nReturnCode;
}


// -----------------------------------<User-IO>--------------------------------

/* Eigentlich wollte ich die Standard-Streams durch einen Hook in der Library umleiten,
	aber so etwas gibt es anscheinend nicht. Deswegen musz ich praktisch alle
	IO-Funktionen umdefinieren (siehe wstdio.h). Leider geht das nicht bei allen.
	Man schaue also nach, bevor man eine Funktion benutzt!
*/

int f_f_l_u_s_h( FILE * __stream)
{
	if (((__stream == stdout) && (oflag == FALSE)) || (__stream == stderr))
		return 0;
	else
		return fflush(__stream);
}

int fg_e_t_c( FILE * __stream)
{
	if (__stream == stdin) {
		int c;
		do {
			c = w_getch();
		} while( c == CR);
		return c;
	} else
		return fgetc(__stream);
}

int f_g_e_t_p_o_s( FILE * __stream, fpos_t * __pos)
{
	int result;
	if ((__stream == stdin) || ((__stream == stdout) && (oflag == FALSE)) || (__stream == stderr)) {
		assert(FALSE);
		return 0;
	} else
		result = fgetpos(__stream, __pos);
	return result;
}

char * fg_e_t_s(char * __s, int __n, FILE * __stream)
{
	if (__stream == stdin) {
		int i = 0;
		int c;
		while ( i < (__n-1)) {
			c = w_getch();
			if (c == LF) {
				__s[i++] = LF;
				break;
			}
			if (c != CR)
				__s[i++] = (char)c;
		}
		__s[i] = SE;
		return __s;
	} else
		return fgets( __s, __n, __stream);
}

int fp_u_t_c(int __c, FILE * __stream)
{
	if ((oflag == FALSE) && ((__stream == stdout) || (__stream == stderr))) {
		if ( __c == LF)
			w_putch( CR);
		return w_putch(__c);
//   Ausgabe in Datei *.log  14.6.2000
	} else if ((oflag == TRUE) && ((__stream == stdout) || __stream == stderr)) {
		return fputc( __c, flogp);
	} else
		return fputc( __c, __stream);
}

int fp_u_t_s(const char * __s, FILE * __stream)
{
//	if (((__stream == stdout) && (oflag == FALSE)) || (__stream == stderr)) {    hvogt 14.6.2000
	if ((__stream == stdout) || (__stream == stderr)) {

		int c = SE;
		if (!__s) return EOF;
		do {
			if (*__s) {
				c = *__s++;
				fp_u_t_c(c, __stream);
			} else
				return c;
		} while (TRUE);
	} else
		return fputs( __s, __stream);
}

int fp_r_i_n_t_f(FILE * __stream, const char * __format, ...)
{
	int result;
	char s [IOBufSize];
	va_list args;
	va_start(args, __format);

//	if (((__stream == stdout) && (oflag == FALSE)) || (__stream == stderr)) {
	if ((__stream == stdout) || (__stream == stderr)) {

		s[0] = SE;
		result = vsprintf( s, __format, args);
		fp_u_t_s( s, __stream);
	} else
		result = vfprintf( __stream, __format, args);

	va_end(args);
	return result;
}

int f_c_l_o_s_e( FILE * __stream)
{
	if ((__stream == stdin) || ((__stream == stdout) && (oflag == FALSE)) || (__stream == stderr)) {
		assert(FALSE);
		return 0;
	}
	return fclose( __stream);
}

size_t f_r_e_a_d(void * __ptr, size_t __size, size_t __n, FILE * __stream)
{
//	if ((__stream == stdin) || ((__stream == stdout) && (oflag == FALSE)) || (__stream == stderr)) {
	if (((__stream == stdout) && (oflag == FALSE)) || (__stream == stderr)) {
		assert(FALSE);
		return 0;
	}
	
	if (__stream == stdin) {
		int i = 0;
		int c;
		char s [IOBufSize];
		while ( i < (__size * __n - 1)) {
			c = w_getch();
			if (c == LF) {
//				s[i++] = LF;
				break;
			}
			if (c != CR)
				s[i++] = (char)c;
		}
//		s[i] = SE;
		__ptr = &s[0];
		return (int)(i/__size);
	}	
	return fread( __ptr, __size, __n, __stream);
}

FILE * f_r_e_o_p_e_n(const char * __path, const char * __mode, FILE * __stream)
{
	if ((__stream == stdin)/* || ((__stream == stdout) && (oflag == FALSE)) || (__stream == stderr)*/) {
		assert(FALSE);
		return 0;
	}
	return freopen( __path, __mode, __stream);
}

int fs_c_a_n_f(FILE * __stream, const char * __format, ...)
{
	int result;
	va_list args;
	va_start(args, __format);
	if ((__stream == stdin) || ((__stream == stdout) && (oflag == FALSE)) || (__stream == stderr)) {
		assert(FALSE);
		return 0;
	}
	result = fscanf( __stream, __format, args);
	va_end(args);
	return result;
}

int f_s_e_e_k(FILE * __stream, long __offset, int __whence)
{
	if ((__stream == stdin) || ((__stream == stdout) && (oflag == FALSE)) || (__stream == stderr)) {
		assert(FALSE);
		return 0;
	}
	return fseek( __stream, __offset, __whence);
}

int f_s_e_t_p_o_s(FILE * __stream, const fpos_t *__pos)
{
	if ((__stream == stdin) || ((__stream == stdout) && (oflag == FALSE)) || (__stream == stderr)) {
		assert(FALSE);
		return 0;
	}
	return fsetpos( __stream, __pos);
}

long f_t_e_l_l(FILE * __stream)
{
	if ((__stream == stdin) || ((__stream == stdout) && (oflag == FALSE)) || (__stream == stderr)) {
		assert(FALSE);
		return 0;
	}
	return ftell( __stream);
}

size_t f_w_r_i_t_e(const void * __ptr, size_t __size, size_t __n, FILE * __stream)
{
//	p_r_i_n_t_f("entered fwrite, size %d, n %d \n", __size, __n);
	if (__stream == stdin)  {
		assert(FALSE);
//		p_r_i_n_t_f("False \n");
		return 0;
	}
	if ((__stream == stdout) || (__stream == stderr)) {
		const char * __s = __ptr;
		int c = SE;
		int i = 0;
//		char *out;

//		p_r_i_n_t_f("test1 %s\n", __s);

		if (!__s) return EOF;
		for (i = 0; i< (__size * __n); i++) {
			if (*__s) {
				c = *__s++;
				fp_u_t_c(c, __stream);
			} else
				break;
		};
//		f_r_e_a_d(out, __size, __n, __stream);
//		p_r_i_n_t_f("test2 %s", out);
		return (int)(i/__size);
	}
//	p_r_i_n_t_f("test3 %s\n", __ptr);
	return fwrite( __ptr, __size, __n, __stream);
}

char * g_e_t_s(char * __s)
{
	return fg_e_t_s( __s, 10000, stdin);
}

void p_e_r_r_o_r(const char * __s)
{
	const char * cp;
/*	cp = _strerror( __s);
	fp_u_t_s( cp, stderr);  */
	cp = strerror(errno);
	fp_r_i_n_t_f(stderr, "%s: %s\n", __s, cp);
	// nur als Test fuer NT
//	fp_u_t_s("Test fuer NT: perror, weiter mit RETURN\n", stderr);
//	fg_e_t_c(stdin);
}

int p_r_i_n_t_f(const char * __format, ...)
{
	int result;
	char s [IOBufSize];
	va_list args;
	va_start(args, __format);

	s[0] = SE;
	result = vsprintf( s, __format, args);
	fp_u_t_s( s, stdout);
	va_end(args);
	return result;
}

int p_u_t_s(const char * __s)
{
	return fp_u_t_s( __s, stdout);
}

int s_c_a_n_f(const char * __format, ...)
{
	assert( FALSE);
	return FALSE;
}

int ung_e_t_c(int __c, FILE * __stream)
{
	assert( FALSE);
	return FALSE;
}

int vfp_r_i_n_t_f(FILE * __stream, const char * __format, void * __arglist)
{
	int result;
	char s [IOBufSize];

	s[0] = SE;
//	if (((__stream == stdout) && (oflag == FALSE)) || (__stream == stderr)) {
	if ((__stream == stdout) || (__stream == stderr)) {

		result = vsprintf( s, __format, __arglist);
		fp_u_t_s( s, stdout);
	} else
		result = vfprintf( __stream, __format, __arglist);
	return result;
}

/*int vfs_c_a_n_f(FILE * __stream, const char * __format, void * __arglist)
{
	if (__stream == stdin) {
		assert(FALSE);
		return 0;
	}
	return vfscanf( __stream, __format, __arglist);
}
*/
int vp_r_i_n_t_f(const char * __format, void * __arglist)
{
	int result;
	char s [IOBufSize];

	s[0] = SE;
	result = vsprintf( s, __format, __arglist);
	fp_u_t_s( s, stdout);
	return result;
}

/*int vs_c_a_n_f(const char * __format, void * __arglist)
{
	assert( FALSE);
	return FALSE;
} */

int r_e_a_d(int fd, char * __buf, int __n)
{
	if (fd == 0) {
		int i = 0;
		int c;
		char s [IOBufSize];
		while ( i < __n ) {
			c = w_getch();
			if (c == LF) {
//				s[i++] = LF;
				break;
			}
			if (c != CR)
				s[i++] = (char)c;
		}
//		s[i] = SE;
		__buf = &s[0];
		return (i);
	} 
	else {
	   return read(fd, __buf, __n);
	}
}
int g_e_t_c(FILE * __fp)
{
	return fg_e_t_c( __fp);
}

int g_e_t_char(void)
{
	return fg_e_t_c( stdin);
}

int p_u_t_char(const int __c)
{
	return fp_u_t_c( __c, stdout);
}

int p_u_t_c(const int __c, FILE * __fp)
{
	return fp_u_t_c( __c, __fp);
}

int f_e_o_f(FILE * __fp)
{
	if ((__fp == stdin) || (__fp == stdout) || (__fp == stderr)) {
		assert(FALSE);
		return 0;
	}
	return feof( __fp);
}

int f_e_r_r_o_r(FILE * __fp)
{
	if ((__fp == stdin) || (__fp == stdout) || (__fp == stderr)) {
		assert(FALSE);
		return 0;
	}
	return ferror( __fp);
}

int fg_e_t_char(void)
{
	return fg_e_t_c( stdin);
}

int fp_u_t_char(int __c)
{
	return fp_u_t_c( __c, stdout);
}

// --------------------------<Verfuegbarer Speicher>----------------------------
/*
size_t _memavl(void)
{
	MEMORYSTATUS ms;
	DWORD sum;
	ms.dwLength = sizeof(MEMORYSTATUS);
	GlobalMemoryStatus( &ms);
	sum = ms.dwAvailPhys + ms.dwAvailPageFile;
	return (size_t) sum;
}
*/
// ---------------------<Aufruf eines anderen Programms>-----------------------

int system( const char * command)
{
	// info-Bloecke
	STARTUPINFO si;
	PROCESS_INFORMATION pi;
	DWORD ExitStatus;

	// Datenstrukturen fuellen
	memset( &si, 0, sizeof( STARTUPINFO));
	si.cb = sizeof( STARTUPINFO);
	memset( &pi, 0, sizeof( PROCESS_INFORMATION));

	// starte den neuen Prozess
	if (!CreateProcess(
		NULL,	// address of module name
		(char *) command,	// address of command line
		NULL,	// address of process security attributes
		NULL,	// address of thread security attributes
		FALSE,	// new process inherits handles
		NORMAL_PRIORITY_CLASS,	// creation flags
		NULL,	// address of new environment block
		NULL,	// address of current directory name
		&si,	// address of STARTUPINFO
		&pi 	// address of PROCESS_INFORMATION
	)) return -1;

	// dieses Handle musz da sein
	if (!pi.hProcess) return -1;

	do {
		// Multitasking ermoeglichen
		WaitForIdle();
		// hole mir den Exit-Code des Prozesses
		if (!GetExitCodeProcess( pi.hProcess, &ExitStatus)) return -1;
		// solange er existiert
	} while( ExitStatus == STILL_ACTIVE);

	// Handles freigeben
	if (pi.hThread)  CloseHandle( pi.hThread);
	if (pi.hProcess) CloseHandle( pi.hProcess);

	// fertig
	return 0;
} // system Windows95

#endif /* HAS_WINDOWS */

