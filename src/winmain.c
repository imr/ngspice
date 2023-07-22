/* Main program for ngspice under Windows OS
   Autor: Wolfgang Muees
   Stand: 28.10.97
   Copyright: Holger Vogt
   Stand: 09.01.2018
   Stand: 20.07.2019
   Stand: 07.12.2019
   Modified BSD license
*/

#include "ngspice/config.h"

#ifdef HAS_WINGUI

#ifndef  _WIN32
#define _WIN32
#endif

#define STRICT              // strict type checking
#define WIN32_LEAN_AND_MEAN
#include <windows.h>        // standard Windows calls
#include <windowsx.h>       // Win32 message cracker
#include <stdio.h>          // sprintf and co
#include <stdlib.h>         // exit codes
#include <stdarg.h>         // var. arguments
#include <assert.h>         // assert macro
#include "ngspice/stringutil.h" // copy
#include <io.h>             // _read
#include <errno.h>
#include <signal.h>
#include <ctype.h>
#include <sys/types.h>
#include <sys/timeb.h>
#ifdef __MINGW32__
#include <tchar.h>
#include <stdio.h>
#endif

#include "hist_info.h" /* history management */
#include "ngspice/bool.h"   /* bool defined as unsigned char */
#include "misc/misc_time.h" /* timediff */
#include "ngspice/memory.h" /* TMALLOC */
#include "winmain.h"

/* Constants */
#define TBufSize 65536       // size of text buffer
#define CR VK_RETURN        // Carriage Return
#define VK_EOT 0x1A         // End of Transmission, should emulate ctrl-z
#define LF 10               // Line Feed
#define SE 0                // String termination
#define BorderSize 8        // Umrandung des Stringfeldes
#define SBufSize 300        // Groesze des Stringbuffers
#define IOBufSize 16348      // Groesze des printf-Buffers
#define HIST_SIZE   20  /* Max # commands held in history */
#define N_BYTE_HIST_BUF 512 /* Initial size of history buffer in bytes */
#define StatusHeight 25         // Hoehe des Status Bars
#define StatusFrame 2           // Abstand Statusbar / StatusElement
#define StatusElHeight (StatusHeight - 2 * StatusFrame)
#define SourceLength 500        // Platz fuer Source File Name
#define AnalyseLength 100       // Platz fuer Analyse
#define QuitButtonLength 60

/* Define the macro below to create a larger main window that is useful
 * for seeing debug output that is generated before the window can be
 * resized */
//#define BIG_WINDOW_FOR_DEBUGGING

/* macro to ignore unused variables and parameters */
#define NG_IGNORE(x)  (void)x

#define QUIT_BUTTON_ID 2
#define STOP_BUTTON_ID 3

/* Types */
typedef char SBufLine[SBufSize + 1];  // Eingabezeile

/* Global variables */
HINSTANCE       hInst;              /* Application instance */
int             WinLineWidth = 690; /* Window width */
HWND            hwMain;             /* Main Window of the application */
HWND            twText;             /* Text window */
HWND            swString;           /* input string */
HWND            hwStatus;           /* status bar */
HWND            hwSource;           /* display of source name */
HWND            hwAnalyse;          /* analysis window */
HWND            hwQuitButton;       /* End button */
HWND            hwStopButton;       /* Pause button */
static int      nReturnCode = 0;    /* WinMain return value */
static int      nShowState;         /* Display mode of main window */
#ifdef EXT_ASC
static WNDCLASS hwMainClass;        /* Class definition for the main window */
static LPCTSTR  hwClassName  = "SPICE_TEXT_WND";/* Class name of the main window */
static LPCTSTR hwWindowName = PACKAGE_STRING;   /* main window displayed name */
static WNDCLASS twTextClass;                    /* Class definition for the text box */
static LPCTSTR twClassName  = "SPICE_TEXT_BOX"; /* Class name for the text box */
static LPCTSTR twWindowName = "TextOut";        /* text box name */
static WNDCLASS swStringClass;                  /* Class definition of string window */
static LPCTSTR swClassName  = "SPICE_STR_IN";   /* Class name of text input */
static LPCTSTR swWindowName = "StringIn";       /* Window name */
static WNDCLASS hwElementClass;                 /* Class definition of status displays */
static LPCTSTR hwElementClassName = "ElementClass";
static LPCTSTR hwSourceWindowName = "SourceDisplay";
static LPCTSTR hwAnalyseWindowName = "AnalyseDisplay";
#else
static WNDCLASSW hwMainClassW;        /* Class definition for the main window */
static LPCWSTR  hwClassNameW = L"SPICE_TEXT_WND";/* Class name of the main window */
static LPCWSTR hwWindowNameW = L"ngspice 26";   /* main window displayed name */
static WNDCLASSW twTextClassW;                    /* Class definition for the text box */
static LPCWSTR twClassNameW = L"SPICE_TEXT_BOX"; /* Class name for the text box */
static LPCWSTR twWindowNameW = L"TextOut";        /* text box name */
static WNDCLASSW swStringClassW;                  /* Class definition of string window */
static LPCWSTR swClassNameW = L"SPICE_STR_IN";   /* Class name of text input */
static LPCWSTR swWindowNameW = L"StringIn";       /* Window name */
static WNDCLASSW hwElementClassW;                 /* Class definition of status displays */
static LPCWSTR hwElementClassNameW = L"ElementClass";
static LPCWSTR hwSourceWindowNameW = L"SourceDisplay";
static LPCWSTR hwAnalyseWindowNameW = L"AnalyseDisplay";
#endif

static size_t   TBufEnd = 0;                    /* Pointer to \0 */
static char TBuffer[TBufSize + 1];              /* Text buffer */
static SBufLine SBuffer;                        /* Input buffer */
static char CRLF[] = {CR, LF, SE};              /* CR/LF */
static int RowHeight = 16;             /* Height of line of text */
static int LineHeight = 25;            /* Height of input line */
static int VisibleRows = 10;           /* Number of visible lines in text window */
static BOOL DoUpdate = FALSE;          /* Update text window */
static WNDPROC swProc = NULL;          /* original string window procedure */
static WNDPROC twProc = NULL;          /* original text window procedure */
static HFONT efont;                    /* Font for element windows */
static HFONT tfont;                    /* Font for text window */
static HFONT sfont;                    /* Font for string window */

extern bool ft_nginfo; /* some additional info printed */
extern bool ft_ngdebug; /* some additional debug info printed */
extern bool ft_batchmode;
extern FILE *flogp;     /* definition see xmain.c, stdout redirected to file */

extern void cp_doquit(void);
extern void cp_evloop(char*);

static struct History_info *init_history(void);

void UpdateMainText(void);

// ---------------------------<Message Handling>-------------------------------

// Warte, bis keine Messages mehr zu bearbeiten sind
void
WaitForIdle(void)
{
    MSG m;
    // arbeite alle Nachrichten ab
    while (PeekMessage(&m, NULL, 0, 0, PM_REMOVE)) {
        TranslateMessage(&m);
        DispatchMessage(&m);
    }
}


// ---------------------------<Message Handling>-------------------------------

// Warte, bis keine Messages mehr zu bearbeiten sind,
// dann warte auf neue Message (Input handling ohne Dauerloop)
static void
WaitForMessage(void)
{
    MSG m;
    // arbeite alle Nachrichten ab
    while (PeekMessage(&m, NULL, 0, 0, PM_REMOVE)) {
        TranslateMessage(&m);
        DispatchMessage(&m);
    }
    WaitMessage();
}


// -----------------------------<Stringfenster>--------------------------------

// Loeschen des Stringfensters
static void
ClearInput(void)
{
    // Darstellen
    Edit_SetText(swString, "");
}

// ---------------------------<SourceFile-Fenster>-----------------------------

/* New text to Source file window */
void
SetSource(char *Name)
{
    if (hwSource) {
#ifdef EXT_ASC
        SetWindowText(hwSource, Name);
#else
        wchar_t *NameW;
        NameW = TMALLOC(wchar_t, 2 * strlen(Name) + 1);
        MultiByteToWideChar(CP_UTF8, 0, Name, -1, NameW, 2 * (int)strlen(Name) + 1);
        SetWindowTextW(hwSource, NameW);
        tfree(NameW);
#endif
        InvalidateRgn(hwSource, NULL, TRUE);
	}
}


// ------------------------------<Analyse-Fenster>-----------------------------

/* New progress report into analysis window.
   Update only every DELTATIME milliseconds */

#define DELTATIME 150

void
SetAnalyse(char *Analyse,   /* in: analysis type */
           int DecaPercent) /* in: 10 times the progress [%] */
{
    static int OldPercent = -2;     /* Previous progress value */
    static char OldAn[128];         /* Previous analysis type */
    char s[128], t[128];            /* outputs to analysis window and task bar */
    static struct timeb timebefore; /* previous time stamp */
    struct timeb timenow;           /* actual time stamp */
    int diffsec, diffmillisec;      /* differences actual minus prev. time stamp */

    WaitForIdle();

    OldAn[127] = '\0';

    if (((DecaPercent == OldPercent) && !strcmp(OldAn, Analyse)) || !strcmp(Analyse, "or"))
        return;

    /* get actual time */
    ftime(&timenow);
    timediff(&timenow, &timebefore, &diffsec, &diffmillisec);

    OldPercent = DecaPercent;
    /* output only into hwAnalyse window and if time elapsed is larger than
       DELTATIME given value, or if analysis has changed, else return */
    if (hwAnalyse && ((diffsec > 0) || (diffmillisec > DELTATIME) || strcmp(OldAn, Analyse))) {
        if (DecaPercent < 0) {
            sprintf(s, "   -- ready --");
            sprintf(t, "%s", PACKAGE_STRING);
        }
        else if (DecaPercent == 0) {
            sprintf(s, " %s", Analyse);
            sprintf(t, "%s   %s", PACKAGE_STRING, Analyse);
        }
        else if (!strcmp(Analyse, "shooting")) {
            sprintf(s, " %s: %d", Analyse, DecaPercent);
            sprintf(t, "%s   %d", PACKAGE_STRING, DecaPercent);
        }
        else {
            sprintf(s, " %s: %3.1f%%", Analyse, (double)DecaPercent/10.);
            sprintf(t, "%s   %3.1f%%", PACKAGE_STRING, (double)DecaPercent/10.);
        }
        timebefore.dstflag = timenow.dstflag;
        timebefore.millitm = timenow.millitm;
        timebefore.time = timenow.time;
        timebefore.timezone = timenow.timezone;
        /* info when previous analysis period has finished */
        if (strcmp(OldAn, Analyse)) {
            if ((ft_nginfo || ft_ngdebug) && (strcmp(OldAn, "")))
                win_x_printf("%s finished after %4.2f seconds.\n", OldAn, seconds());
            strncpy(OldAn, Analyse, 127);
        }

#ifdef EXT_ASC
        SetWindowText(hwAnalyse, s);
        SetWindowText(hwMain, t);
#else
        wchar_t sw[256];
        wchar_t tw[256];
        MultiByteToWideChar(CP_UTF8, 0, s, -1, sw, 256);
        MultiByteToWideChar(CP_UTF8, 0, t, -1, tw, 256);
        /* Analysis window */
        SetWindowTextW(hwAnalyse, sw);
        /* ngspice task bar */
        SetWindowTextW(hwMain, tw);
#endif
        InvalidateRgn(hwAnalyse, NULL, TRUE);
        UpdateWindow(hwAnalyse);
        InvalidateRgn(hwMain, NULL, TRUE);
        UpdateWindow(hwMain);
    }
}


// ------------------------------<Textfenster>---------------------------------

// Anpassen des Scrollers im Textfenster
// Stellt gleichzeitig den Text neu dar
static void
AdjustScroller(void)
{
    int LineCount;
    int FirstLine;
    int MyFirstLine;
    LineCount = Edit_GetLineCount(twText);
    FirstLine = Edit_GetFirstVisibleLine(twText);
    MyFirstLine = LineCount - VisibleRows;

    if (MyFirstLine < 0 )
        MyFirstLine = 0;

    Edit_Scroll(twText, (WPARAM) MyFirstLine - FirstLine, 0);
    // Das wars
    DoUpdate = FALSE;
}


// Loeschen einer Zeile im Textbuffer
static void
_DeleteFirstLine(void)
{
    char *cp = strchr(TBuffer, LF);
    if (!cp) {
        // Buffer leeren
        TBufEnd = 0;
        TBuffer[TBufEnd] = SE;
        return;
    }
    cp++;
    TBufEnd -= (size_t)(cp - TBuffer);
    memmove(TBuffer, cp, TBufEnd);
    TBuffer[TBufEnd] = SE;
}

/* Compare old system time with current system time.
   If difference is larger than ms milliseconds, return TRUE.
   If time is less than the delay time (in milliseconds), return TRUE. */
static bool
CompareTime(int ms, int delay)
{
    static __int64 prevfileTime64Bit;
    static __int64 startfileTime64Bit;
    /* conversion: time in ms -> 100ns */
    __int64 reftime = ms * 10000;
    __int64 delaytime = delay * 10000;
    FILETIME newtime;
    /* get time in 100ns units */
    GetSystemTimeAsFileTime(&newtime);
    ULARGE_INTEGER theTime;
    theTime.LowPart = newtime.dwLowDateTime;
    theTime.HighPart = newtime.dwHighDateTime;
    __int64 fileTime64Bit = theTime.QuadPart;
    __int64 difffileTime64Bit = fileTime64Bit - prevfileTime64Bit;
    /* Catch the delay start time */
    if ((startfileTime64Bit) == 0) {
        startfileTime64Bit = fileTime64Bit;
    }
    if ((fileTime64Bit - startfileTime64Bit) < delaytime)
        return TRUE;
    if ((difffileTime64Bit) > reftime) {
        prevfileTime64Bit = fileTime64Bit;
        return TRUE;
    }
    else
        return FALSE;
}

// Add a char to the text buffer
static void
AppendChar(char c)
{
    // Limit the text buffer size to TBufSize
    while ((TBufEnd + 4) >= TBufSize)
        _DeleteFirstLine();
    // Add character
    TBuffer[TBufEnd++] = c;
    TBuffer[TBufEnd] = SE;
    DoUpdate = TRUE;

    /* If line is complete, and waiting time has passed, show it in text window.
       If time is less than delay time, always show the line (useful during start-up) */
    if (c == LF && CompareTime(30, 500)) {
        DisplayText();
        WaitForIdle();
    }
}


// Anfuegen eines Strings an den TextBuffer
static void
AppendString(const char *Line)
{
    size_t i;
    if (!Line)
        return;

    // Zeilenlaenge bestimmen
    i = strlen(Line);
    // Textbuffer nicht zu grosz werden lassen
    while ((i + TBufEnd + 3) >= TBufSize)
        _DeleteFirstLine();
    // Zeile dranhaengen
    strcpy(&TBuffer[TBufEnd], Line);
    TBufEnd += i;
    DoUpdate = TRUE;
}


// Text neu darstellen
static void
DisplayText(void)
{
    // Show text
#ifdef EXT_ASC
    Edit_SetText(twText, TBuffer);
#else
    wchar_t *TWBuffer;
    TWBuffer = TMALLOC(wchar_t, 2 * strlen(TBuffer) + 1);
    if (MultiByteToWideChar(CP_UTF8, 0, TBuffer, -1, TWBuffer, 2 * (int)strlen(TBuffer) + 1) == 0)
        swprintf(TWBuffer, 2 * strlen(TBuffer), L"UTF-8 to UTF-16 conversion failed with 0x%x\n%hs could not be converted\n", GetLastError(), TBuffer);
    SetWindowTextW(twText, TWBuffer);
    tfree(TWBuffer);
#endif
    // Scroller updaten, neuen Text darstellen
    AdjustScroller();
}


// Anfuegen einer Zeile an den Textbuffer
#if 0
void
AppendLine(const char *Line)
{
    if (!Line)
        return;

    // String anhaengen
    AppendString(Line);

    // CRLF anhaengen
    AppendString(CRLF);
}
#endif


// -----------------------------------<User-IO>--------------------------------

// Lese ein Zeichen ein
static int
w_getch(void)
{
    int c;

    // Sind noch Zeichen da?
    c = SBuffer[0];
    if (!c) {
        // Alte Informationen darstellen
        if (DoUpdate)
            DisplayText();
        // Focus setzen
        SetFocus(swString);
        // Cursor = normal
        SetCursor(LoadCursor(NULL, IDC_IBEAM));
        // Analyse ist fertig
        SetAnalyse("", -1);
        // Warten auf die Eingabe
        do {
            WaitForMessage();
            c = SBuffer[0];
        } while (!c);
        // Zeichen an die Ausgabe anhaengen
        AppendString(SBuffer);
        // Cursor = warten
        SetCursor(LoadCursor(NULL, IDC_WAIT));
    }

    /* Shift out the character being returned. After the entire
     * contents of the buffer is read, it first byte is '\0' from
     * the null termination of the buffer.
     *
     * Inefficient way to process the string, but it should work */
    (void) memmove(SBuffer, SBuffer + 1, sizeof SBuffer - 1);
    return c;
}


// Gebe ein Zeichen aus
static int
w_putch(int c)
{
    if (c)
        AppendChar((char) c);
    return c;
}


/* -------------------------------<Window procedures>-------------------------- */

/* Main window changes size */
static void
Main_OnSize(HWND hwnd, UINT state, int cx, int cy)
{
    int h = cy - LineHeight - StatusHeight;

    NG_IGNORE(hwnd);
    NG_IGNORE(state);

    /* Expand text window */
    MoveWindow(twText, 0, 0, cx, h , TRUE);
    VisibleRows = (h / RowHeight) - 1;
    AdjustScroller();

    /* Expand string window */
    MoveWindow(swString, 0, h, cx, LineHeight, TRUE);

    /* Expand Status Elements */
    h = cy - LineHeight + StatusFrame - 2;
    int statbegin = 3 * StatusFrame + 2 * QuitButtonLength + AnalyseLength + 5;
    MoveWindow(hwSource, StatusFrame, h, cx - statbegin - BorderSize, StatusElHeight, TRUE);
    MoveWindow( hwAnalyse, cx - statbegin, h, AnalyseLength, StatusElHeight, TRUE);
    MoveWindow( hwQuitButton, cx - StatusFrame - QuitButtonLength - 1,
       h + 1, QuitButtonLength, StatusElHeight, TRUE);
    MoveWindow(hwStopButton, cx - StatusFrame - QuitButtonLength - QuitButtonLength - 3,
        h + 1, QuitButtonLength, StatusElHeight, TRUE);
}


/* Write a command into the command buffer */
static void
PostSpiceCommand(const char * const cmd)
{
    strcpy(SBuffer, cmd);
    strcat(SBuffer, CRLF);
}


/* Main Window procedure */
static LRESULT CALLBACK
MainWindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg) {

        /* command issued by pushing the "Quit" button */
    case WM_COMMAND:
        if (HIWORD(wParam) == BN_CLICKED) {
            if (ft_batchmode && LOWORD(wParam) == QUIT_BUTTON_ID &&
                (MessageBox(NULL, "Do you want to quit ngspice?", "Quit", MB_OKCANCEL | MB_ICONERROR) == IDCANCEL))
                goto DEFAULT_AFTER;
            if (ft_batchmode && LOWORD(wParam) == STOP_BUTTON_ID &&
                (MessageBox(NULL, "Stop in Batch Mode is not available!", "Stop", MB_OK) == IDOK))
                goto DEFAULT_AFTER;
        }
        if (LOWORD(wParam) == QUIT_BUTTON_ID)
            SendMessage(GetParent((HWND)lParam), WM_CLOSE, 0, 0);
        if (LOWORD(wParam) == STOP_BUTTON_ID)
            SendMessage(GetParent((HWND)lParam), WM_USER, 0, 0);
        /* write all achieved so far to log file */
        if (flogp)
            win_x_fflush(flogp);
        goto DEFAULT_AFTER;

    case WM_CLOSE:
        cp_doquit();
        /* continue if the user declined the 'quit' command */
        return 0;

    case WM_USER:
        cp_evloop(NULL);
        goto DEFAULT_AFTER;

    case WM_SIZE:
        HANDLE_WM_SIZE(hwnd, wParam, lParam, Main_OnSize);
        goto DEFAULT_AFTER;

    default:
    DEFAULT_AFTER:
#ifdef EXT_ASC
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
#else
        return DefWindowProcW(hwnd, uMsg, wParam, lParam);
#endif
    }
}


/* Procedure for string (input) window */
static LRESULT CALLBACK
StringWindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    static struct History_info **pp_hi; /* handle to history */

    switch (uMsg) {
    case WM_CREATE:
        /* Get access to history information */
#ifdef EXT_ASC
        pp_hi = (struct History_info **)
                ((LPCREATESTRUCT) lParam)->lpCreateParams;
#else
        pp_hi = (struct History_info **)
                ((LPCREATESTRUCTW) lParam)->lpCreateParams;
#endif
        break;
    case WM_KEYDOWN: {
        const UINT i = (UINT) wParam;
        if ((i == VK_UP) || (i == VK_DOWN)) {
            /* Set old text to new */
#ifdef EXT_ASC
            SetWindowText(hwnd, (i == VK_UP) ?
                    history_get_prev(*pp_hi, NULL) :
                    history_get_next(*pp_hi, NULL));
            /* Put cursor to end of line */
            CallWindowProc(swProc, hwnd, uMsg, (WPARAM) VK_END, lParam);
#else
            const char *newtext = (i == VK_UP) ?
                    history_get_prev(*pp_hi, NULL) :
                    history_get_next(*pp_hi, NULL);
            wchar_t *newtextW;
            newtextW = TMALLOC(wchar_t, 2 * strlen(newtext) + 1);
            MultiByteToWideChar(
                    CP_UTF8, 0, newtext, -1, newtextW, 2 * (int) strlen(newtext) + 1);
            SetWindowTextW(swString, newtextW);
            tfree(newtextW);
            /* Put cursor to end of line */
            CallWindowProcW(swProc, hwnd, uMsg, (WPARAM) VK_END, lParam);
#endif
            return 0;
        }
        if (i == VK_ESCAPE) {
            ClearInput();
            return 0;
        }
        break;
    }
    case WM_CHAR: {
        const char c = (char) wParam;
        if (c == CR) {
            /* Get text from the window. Must leave space for crlf
             * that is appended. -1 accounts for NULL as follows:
             * The last argument to GetWindowText is the size of the
             * buffer for writing the string + NULL. The NULL will be
             * overwritten by the strcpy below, so it should not be
             * counted in the size needed for the CRLF string. */
#ifdef EXT_ASC
            const int n_char_returned = GetWindowText(
                    hwnd, SBuffer, sizeof SBuffer - (sizeof CRLF - 1));
#else
            wchar_t *WBuffer = TMALLOC(wchar_t, sizeof(SBuffer));
            /* for utf-8 the number of characters is not the number of bytes returned */
            GetWindowTextW(hwnd, WBuffer, sizeof SBuffer - (sizeof CRLF - 1));
            WideCharToMultiByte(CP_UTF8, 0, WBuffer,
                     -1, SBuffer, sizeof SBuffer - 1, NULL, FALSE);
            /* retrive here the number of bytes returned */
            const int n_char_returned = (int)strlen(SBuffer);
            tfree(WBuffer);
#endif
            unsigned int n_char_prev_cmd;

            /* Add the command to the history if it is different from the
             * previous one. This avoids filling the buffer with the same
             * command and allows faster scrolling through the commands.
             * history_get_newest() is called rather than history_get_prev()
             * since the current return position may not be the last one
             * and the position should not be changed. */
            const char *cmd_prev = history_get_newest(
                    *pp_hi, &n_char_prev_cmd);
            if ((int) n_char_prev_cmd != n_char_returned ||
                    strcmp(SBuffer, cmd_prev) != 0) {
                /* Different, so add */
                history_add(pp_hi, n_char_returned, SBuffer);
            }
            else {
                history_reset_pos(*pp_hi);
            }

            strcpy(SBuffer + n_char_returned, CRLF);
            ClearInput();
            return 0;
        }
        if (c == VK_ESCAPE)
            return 0;
        /* ctrl-z ends input from string window (like a console input),
           FIXME: not yet working */
        if (c == VK_EOT) {
//                strcat(SBuffer, "&#004");
            SBuffer[0] = c; // '\004';
            SBuffer[1] = '\n';
            return 0;
        }
        /* ctrl-c interrupts simulation */
        if (c == VK_CANCEL) {
            raise (SIGINT);
            return 0;
        }
    }
    } /* end of switch over handled messages */

    /* Fowrard to be processed further by swProc */
#ifdef EXT_ASC
        return CallWindowProc(swProc, hwnd, uMsg, wParam, lParam);
#else
        return CallWindowProcW( swProc, hwnd, uMsg, wParam, lParam);
#endif
}


/* Procedure for text window */
static LRESULT CALLBACK
TextWindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    unsigned char c;
    UINT i;

    switch (uMsg) {

    case WM_KEYDOWN:
        i = (UINT) wParam;
        if ((i == VK_UP) || (i == VK_DOWN) || (i == VK_ESCAPE)) {
            /* redirect input into string window */
            SetFocus(swString);
            return SendMessage(swString, uMsg, wParam, lParam);
        }
        goto DEFAULT_TEXT;

    case WM_CHAR:
        c = (unsigned char) wParam;
        if ((c == CR) || (c >= ' ') || (c == VK_ESCAPE)) {
            /* redirect input into string window */
            SetFocus(swString);
            return SendMessage(swString, uMsg, wParam, lParam);
        }
        /* ctrl-c interrupts simulation */
        if (c == VK_CANCEL) {
            raise (SIGINT);
            return 0;
        }
    default:
    DEFAULT_TEXT:
#ifdef EXT_ASC
        return CallWindowProc(twProc, hwnd, uMsg, wParam, lParam);
#else
        return CallWindowProcW( twProc, hwnd, uMsg, wParam, lParam);
#endif
    }
}


static void
Element_OnPaint(HWND hwnd)
{
    PAINTSTRUCT ps;
    RECT r;
    RECT s;
    HGDIOBJ o;
#ifdef EXT_ASC
    char buffer[128];
#else
    wchar_t bufferW[256];
#endif
    int i;

    /* Prepare */
    HDC hdc = BeginPaint(hwnd, &ps);
    GetClientRect(hwnd, &r);

    /* Draw frame */
    o = GetStockObject(GRAY_BRUSH);
    s.left      = r.left;
    s.right     = r.right;
    s.top       = r.top;
    s.bottom    = r.top + 1;
    FillRect(hdc, &s, o);

    s.right     = r.left + 1;
    s.bottom    = r.bottom;
    FillRect(hdc, &s, o);

    o = GetStockObject(WHITE_BRUSH);
    s.right     = r.right;
    s.top       = r.bottom - 1;
    FillRect(hdc, &s, o);

    s.left      = r.right - 1;
    s.top       = r.top;
    FillRect(hdc, &s, o);

    /* Draw contents */
#ifdef EXT_ASC
    buffer[0] = '\0';
    i = GetWindowText(hwnd, buffer, 127);
    s.left      = r.left + 1;
    s.right     = r.right - 1;
    s.top       = r.top + 1;
    s.bottom    = r.bottom - 1;
    o = GetStockObject(LTGRAY_BRUSH);
    FillRect(hdc, &s, o);
    SetBkMode(hdc, TRANSPARENT);
    ExtTextOut(hdc, s.left + 1, s.top + 1, ETO_CLIPPED, &s, buffer, (unsigned)i, NULL);
#else
    bufferW[0] = '\0';
    i = GetWindowTextW(hwnd, bufferW, 255);
    s.left = r.left + 1;
    s.right = r.right - 1;
    s.top = r.top + 1;
    s.bottom = r.bottom - 1;
    o = GetSysColorBrush(COLOR_BTNFACE);
    FillRect(hdc, &s, o);
    SetBkMode(hdc, TRANSPARENT);
    SelectObject(hdc, efont);
    ExtTextOutW(hdc, s.left + 1, s.top + 1, ETO_CLIPPED, &s, bufferW, (unsigned)i, NULL);
#endif
    /* End */
    EndPaint(hwnd, &ps);
}


/* Procedure for element window */
static LRESULT CALLBACK
ElementWindowProc(HWND hwnd, UINT uMsg, WPARAM wParam, LPARAM lParam)
{
    switch (uMsg) {

    case WM_PAINT:
        HANDLE_WM_PAINT(hwnd, wParam, lParam, Element_OnPaint);
        return 0;

    default:
#ifdef EXT_ASC
        return DefWindowProc(hwnd, uMsg, wParam, lParam);
#else
        return DefWindowProcW( hwnd, uMsg, wParam, lParam);
#endif
    }
}


#define SPACE           ' '
#define QUOTE           '\"'
#define DELIMITER       26      /* for the lack of anything better */
#define DELIMITERSTRING "\26"

/*
  This function converts a string into an argc/argv represenation.
  INPUT:
  cmdline     -   a string
  OUTPUT:
  argc        -   the number of equivalent argv strings
  which is also the number of strings in argv
  argv        -   the argv given the input string which
  consists of seperate strings for each argument
  RETURNS:
  0  on success
  -1 on failure
*/

static int
MakeArgcArgv(char *cmdline, int *argc, char ***argv)
{
    char  *pC1;                     /* a temporary character pointer */
    char  *pWorkString = NULL;      /* a working copy of cmdline */
    int    i;                       /* a loop counter */
    int    quoteflag = 0;           /* for the finite state machine parsing cmdline */
    bool   firstspace = TRUE;       /* count only the first space */
    int    numargs = 1;             /* the number of command line arguments,
                                       later copied to *argc */
    char **tmpargv;                 /* the temporary argv, later copied to *argv */
    int    status = ERROR_SUCCESS;  /* status */
    char   buffer[MAX_PATH + 1];
    char deli[2];


    /* make sure we aren't dealing with any NULL pointers */
    if ((NULL == argc) || (NULL == argv))
    {
        status = -1;
        goto outahere;
    }

    *argc = 0;      /* set the count to zero to start */
    *argv = NULL;   /* set the pointer to NULL to start */

    /*  if the string passed in was a NULL pointer, consider this
        to be an empty command line and give back only
        an argc of 1 and an argv[0] */
    if (NULL != cmdline)
    {
        /*  make a copy of the string so that we can modify it
            without messing up the original */
        pWorkString = copy(cmdline);
        if (NULL == pWorkString)
            return -1; /* memory allocation error */
        /*  Now, to make sure we don't have any quoted arguments
            with spaces in them, replace all spaces except those
            between " marks with our own special delimiter for
            strtok */
        /* trim all the whitespace off the end of the string. */
        for (i = (signed)strlen(pWorkString) - 1; i >= 0; i--)
            if (isspace((unsigned char) pWorkString[i]))
                pWorkString[i] = '\0';
            else
                break;
#if defined(__CYGWIN__)
        /* for CYGWIN: trim off the leading white space delivered by lpszCmdLine. */
        pWorkString = rlead(pWorkString);
#endif
        /*  If we still have a string left, parse it for all
            the arguments. */
        if (strlen(pWorkString))
        {
            /*  This could probably be done with strtok as well
                but strtok is destructive if I wanted to look for " \""
                and I couldn't tell what delimiter that I had bumped
                against */
            for (i = 0; i < (signed)strlen(pWorkString); i++)
                switch (pWorkString[i])
                {
                case SPACE:
                    if (!quoteflag) {
                        pWorkString[i] = DELIMITER;  /* change space to delimiter */
                        if (firstspace) /* count only the first space */
                            numargs++;
                        firstspace = FALSE;
                    }
                    break;
                case QUOTE:
                    quoteflag = !quoteflag; /* turns on and off as we pass quotes */
                    break;
                default:
                    firstspace = TRUE;
                    break;
                }

            /*  Now, we should have ctrl-Zs everywhere that
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
    tmpargv = (char**) malloc((unsigned)numargs * sizeof(char *));
    if (NULL == tmpargv)
    {
        status = -1;
        goto outahere;
    }
    /*  API to give the program name */
    GetModuleFileName(NULL, buffer, sizeof(buffer));

    tmpargv[0] = copy(buffer); /* add program name to argv */

    deli[0] = DELIMITER;
    deli[1] = '\0'; /* delimiter for strtok */

    pC1 = NULL;
    /*  Now actually strdup all the arguments out of the string
        and store them in the argv */
    for (i = 1; i < numargs; i++) {
        if (NULL == pC1)
            pC1 = pWorkString;

        if (i == 1)
            tmpargv[i] = copy(strtok(pC1, deli));
        else
            tmpargv[i] = copy(strtok(NULL, deli));
    }

    /*  copy the working values over to the arguments */
    *argc = numargs;
    *argv = tmpargv;

 outahere:
    /*  free the working string if one was allocated */
    if (pWorkString)
        free(pWorkString);

    return status;
}



/* Main entry point for our Windows application */
#ifdef EXT_ASC
int WINAPI
WinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPSTR lpszCmdLine, _In_ int nCmdShow)
#else
int WINAPI
wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE hPrevInstance, _In_ LPWSTR wlpszCmdLine, _In_ int nCmdShow)
#endif
{
    int ix, iy; /* width and height of screen */
    int status;

    int argc;
    char **argv;

    /* Initialize history info to a maximum of HIST_SIZE commands.
     * The initial buffer for storage is N_BYTE_HIST_BUF bytes. */
    struct History_info *p_hi = init_history();
    if (p_hi == (struct History_info *) NULL) {
        goto THE_END;
    }

    RECT wsize; /* size of usable window */

    NG_IGNORE(hPrevInstance);

#ifndef EXT_ASC
    /* convert wchar to utf-8 */
    char lpszCmdLine[1024];
    WideCharToMultiByte(CP_UTF8, 0, wlpszCmdLine, -1, lpszCmdLine, 1023, NULL, FALSE);
#endif
    /* fill global variables */
    hInst = hInstance;
    nShowState = nCmdShow;

    /* Initialize text buffer */
    TBufEnd = 0;
    TBuffer[TBufEnd] = SE;
    SBuffer[0] = SE;

    /* Define main window class */
#ifdef EXT_ASC
    hwMainClass.style           = CS_HREDRAW | CS_VREDRAW;
    hwMainClass.lpfnWndProc     = MainWindowProc;
    hwMainClass.cbClsExtra      = 0;
    hwMainClass.cbWndExtra      = 0;
    hwMainClass.hInstance       = hInst;
    hwMainClass.hIcon           = LoadIcon(hInst, MAKEINTRESOURCE(101));
    hwMainClass.hCursor         = LoadCursor(NULL, IDC_ARROW);
    hwMainClass.hbrBackground   = GetStockObject(LTGRAY_BRUSH);
    hwMainClass.lpszMenuName    = NULL;
    hwMainClass.lpszClassName   = hwClassName;

    if (!RegisterClass(&hwMainClass))
        goto THE_END;
#else
    hwMainClassW.style = CS_HREDRAW | CS_VREDRAW;
    hwMainClassW.lpfnWndProc = MainWindowProc;
    hwMainClassW.cbClsExtra = 0;
    hwMainClassW.cbWndExtra = 0;
    hwMainClassW.hInstance = hInst;
    hwMainClassW.hIcon = LoadIconW(hInst, MAKEINTRESOURCEW(101));
    hwMainClassW.hCursor = LoadCursorW(NULL, MAKEINTRESOURCEW(32512));
    hwMainClassW.hbrBackground = GetStockObject(LTGRAY_BRUSH);
    hwMainClassW.lpszMenuName = NULL;
    hwMainClassW.lpszClassName = hwClassNameW;
    if (!RegisterClassW(&hwMainClassW))
        goto THE_END;
#endif

    /* Define text window class */
#ifdef EXT_ASC
    if (!GetClassInfo(NULL, "EDIT", &twTextClass))
        goto THE_END;

    twProc = twTextClass.lpfnWndProc;
    twTextClass.lpfnWndProc     = TextWindowProc;
    twTextClass.hInstance       = hInst;
    twTextClass.lpszMenuName    = NULL;
    twTextClass.lpszClassName   = twClassName;

    if (!RegisterClass(&twTextClass))
        goto THE_END;
#else
    if (!GetClassInfoW(NULL, L"EDIT", &twTextClassW)) goto THE_END;
    twProc = twTextClassW.lpfnWndProc;
    twTextClassW.lpfnWndProc = TextWindowProc;
    twTextClassW.hInstance = hInst;
    twTextClassW.lpszMenuName = NULL;
    twTextClassW.lpszClassName = twClassNameW;
    if (!RegisterClassW(&twTextClassW))
        goto THE_END;
#endif

    /* Define string window class */
#ifdef EXT_ASC
    if (!GetClassInfo(NULL, "EDIT", &swStringClass))
        goto THE_END;

    swProc = swStringClass.lpfnWndProc;
    swStringClass.lpfnWndProc   = StringWindowProc;
    swStringClass.hInstance     = hInst;
    swStringClass.lpszMenuName  = NULL;
    swStringClass.lpszClassName = swClassName;

    if (!RegisterClass(&swStringClass))
        goto THE_END;
#else
    if (!GetClassInfoW(NULL, L"EDIT", &swStringClassW)) goto THE_END;
    swProc = swStringClassW.lpfnWndProc;
    swStringClassW.lpfnWndProc = StringWindowProc;
    swStringClassW.hInstance = hInst;
    swStringClassW.lpszMenuName = NULL;
    swStringClassW.lpszClassName = swClassNameW;
    if (!RegisterClassW(&swStringClassW))
        goto THE_END;
#endif

    /* Define status element class */
#ifdef EXT_ASC
    hwElementClass.style            = CS_HREDRAW | CS_VREDRAW;
    hwElementClass.lpfnWndProc      = ElementWindowProc;
    hwElementClass.cbClsExtra       = 0;
    hwElementClass.cbWndExtra       = 0;
    hwElementClass.hInstance        = hInst;
    hwElementClass.hIcon            = NULL;
    hwElementClass.hCursor          = LoadCursor(NULL, IDC_ARROW);
    hwElementClass.hbrBackground    = GetStockObject(LTGRAY_BRUSH);
    hwElementClass.lpszMenuName     = NULL;
    hwElementClass.lpszClassName    = hwElementClassName;

    if (!RegisterClass(&hwElementClass))
        goto THE_END;
#else
    hwElementClassW.style = CS_HREDRAW | CS_VREDRAW;
    hwElementClassW.lpfnWndProc = ElementWindowProc;
    hwElementClassW.cbClsExtra = 0;
    hwElementClassW.cbWndExtra = 0;
    hwElementClassW.hInstance = hInst;
    hwElementClassW.hIcon = NULL;
    hwElementClassW.hCursor = LoadCursorW(NULL, MAKEINTRESOURCEW(32512));
    hwElementClassW.hbrBackground = GetStockObject(LTGRAY_BRUSH);
    hwElementClassW.lpszMenuName = NULL;
    hwElementClassW.lpszClassName = hwElementClassNameW;
    if (!RegisterClassW(&hwElementClassW))
        goto THE_END;
#endif

    /* Font for element status windows (source, analysis, Quit button) */
    efont = CreateFontW(16, 6, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE,
        ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        ANTIALIASED_QUALITY, VARIABLE_PITCH, L"");
/*    efont = CreateFontW(16, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE,
            DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            NONANTIALIASED_QUALITY, VARIABLE_PITCH, L"Segoe UI");*/
/*    efont = CreateFontW(15, 0, 0, 0, FW_MEDIUM, FALSE, FALSE, FALSE,
            ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            NONANTIALIASED_QUALITY, FIXED_PITCH | FF_MODERN, L"Courier");*/
    if (!efont)
        efont = GetStockFont(ANSI_FIXED_FONT);

#ifdef EXT_ASC
    SystemParametersInfo(SPI_GETWORKAREA, 0, &wsize, 0);
#else
    SystemParametersInfoW(SPI_GETWORKAREA, 0, &wsize, 0);
#endif
    iy = wsize.bottom;
    ix = wsize.right;
#ifndef BIG_WINDOW_FOR_DEBUGGING
    const int iyt = iy / 3; /* height of screen divided by 3 */
#ifdef EXT_ASC
    hwMain = CreateWindow(hwClassName, hwWindowName, WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN,
                           0, iyt * 2, ix, iyt, NULL, NULL, hInst, NULL);
#else
    hwMain = CreateWindowW(hwClassNameW, hwWindowNameW, WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN,
                           0, iyt * 2, ix, iyt, NULL, NULL, hInst, NULL);
#endif
#else
#ifdef EXT_ASC
    hwMain = CreateWindow(hwClassName, hwWindowName, WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN,
                          0, 0, ix, iy, NULL, NULL, hInst, NULL);
#else
    hwMain = CreateWindowW(hwClassName, hwWindowName, WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN,
                          0, 0, ix, iy, NULL, NULL, hInst, NULL);
#endif
#endif

    if (!hwMain)
        goto THE_END;

    /* Create text window */
#ifdef EXT_ASC
    twText = CreateWindowEx(WS_EX_NOPARENTNOTIFY, twClassName, twWindowName,
                            ES_LEFT | ES_MULTILINE | ES_READONLY | WS_CHILD | WS_BORDER | WS_VSCROLL,
                            20, 20, 300, 100, hwMain, NULL, hInst, NULL);
#else
    twText = CreateWindowExW(WS_EX_NOPARENTNOTIFY, twClassNameW, twWindowNameW,
        ES_LEFT | ES_MULTILINE | ES_READONLY | WS_CHILD | WS_BORDER | WS_VSCROLL,
        20,20,300,100, hwMain, NULL, hInst, NULL);
#endif
    if (!twText)
        goto THE_END;

#ifdef EXT_ASC
    {
        HDC textDC;
        TEXTMETRIC tm;
        tfont = GetStockFont(ANSI_FIXED_FONT);
        SetWindowFont(twText, tfont, FALSE);
        textDC = GetDC(twText);
        if (textDC) {
            SelectObject(textDC, tfont);
            if (GetTextMetrics(textDC, &tm)) {
                RowHeight = tm.tmHeight;
                WinLineWidth = 90 * tm.tmAveCharWidth;
            }
            ReleaseDC(twText, textDC);
        }
    }
#else
    {
	    HDC textDC;
        TEXTMETRICW tm;
        tfont = CreateFontW(15, 0, 0, 0, FW_MEDIUM, FALSE, FALSE,
            FALSE, ANSI_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
            NONANTIALIASED_QUALITY, FIXED_PITCH | FF_MODERN, L"Courier");
        /* Ansi fixed font */
        if(!tfont)
            tfont = GetStockFont(ANSI_FIXED_FONT);
        SetWindowFont( twText, tfont, FALSE);
        textDC = GetDC( twText);
        if (textDC) {
            SelectObject( textDC, tfont);
            if (GetTextMetricsW( textDC, &tm)) {
                RowHeight = tm.tmHeight;
                WinLineWidth = 90 * tm.tmAveCharWidth;
            }
            ReleaseDC( twText, textDC);
        }
    }
#endif

    /* Create string window for input. Give a handle to history info to
     * the window for saving and retrieving commands */
    /* Font for element status windows (source, analysis) */
    sfont = CreateFontW(16, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS,
        ANTIALIASED_QUALITY, VARIABLE_PITCH, L"");
    /* Ansi fixed font */
    if(!sfont)
        sfont = GetStockFont(ANSI_FIXED_FONT);
#ifdef EXT_ASC
    swString = CreateWindowEx(WS_EX_NOPARENTNOTIFY, swClassName, swWindowName,
            ES_LEFT | WS_CHILD | WS_BORDER |
                    ES_AUTOHSCROLL, /* Allow text to scroll */
            20, 20, 300, 100, hwMain, NULL, hInst, &p_hi);
    if (!swString) {
        goto THE_END;
    }

    {
        HDC stringDC;
        TEXTMETRIC tm;
        stringDC = GetDC(swString);
        if (stringDC) {
            if (GetTextMetrics(stringDC, &tm))
                LineHeight = tm.tmHeight + tm.tmExternalLeading + BorderSize;
            ReleaseDC(swString, stringDC);
        }
	}
#else
    swString = CreateWindowExW(WS_EX_NOPARENTNOTIFY, swClassNameW, swWindowNameW,
        ES_LEFT | WS_CHILD | WS_BORDER |
                ES_AUTOHSCROLL, /* Allow text to scroll */
        20, 20, 300, 100, hwMain, NULL, hInst, &p_hi);
    if (!swString)
        goto THE_END;
    {
        HDC stringDC;
        TEXTMETRICW tm;
        stringDC = GetDC(swString);
        if (stringDC) {
            SelectObject(stringDC, sfont);
            if (GetTextMetricsW(stringDC, &tm))
                LineHeight = tm.tmHeight + tm.tmExternalLeading + BorderSize;
            ReleaseDC(swString, stringDC);
        }
	}
#endif

    /* Element windows */
    /* Create source window */
#ifdef EXT_ASC
    hwSource = CreateWindowEx(WS_EX_NOPARENTNOTIFY, hwElementClassName, hwSourceWindowName,
                              WS_CHILD,
                              0, 0, SourceLength, StatusElHeight, hwMain, NULL, hInst, NULL);
#else
    hwSource = CreateWindowExW(WS_EX_NOPARENTNOTIFY, hwElementClassNameW, hwSourceWindowNameW,
                               WS_CHILD,
                               0, 0, SourceLength, StatusElHeight, hwMain, NULL, hInst, NULL);
#endif
    if (!hwSource)
        goto THE_END;

    SetWindowFont(hwSource, efont, FALSE);

    /* Create analysis window */
#ifdef EXT_ASC
    hwAnalyse = CreateWindowEx(WS_EX_NOPARENTNOTIFY, hwElementClassName, hwAnalyseWindowName,
                               WS_CHILD,
                               0, 0, AnalyseLength, StatusElHeight, hwMain, NULL, hInst, NULL);
#else
    hwAnalyse = CreateWindowExW(WS_EX_NOPARENTNOTIFY, hwElementClassNameW, hwAnalyseWindowNameW,
                                WS_CHILD,
                                0,0, AnalyseLength, StatusElHeight, hwMain, NULL, hInst, NULL);
#endif
    if (!hwAnalyse)
        goto THE_END;

    SetWindowFont(hwAnalyse, efont, FALSE);

    /* Create "Quit" button */
#ifdef EXT_ASC
    hwQuitButton = CreateWindow("BUTTON", "Quit", WS_CHILD | BS_PUSHBUTTON, 0, 0, QuitButtonLength,
                                StatusElHeight, hwMain, (HMENU)(UINT_PTR)QUIT_BUTTON_ID, hInst, NULL);
#else
    hwQuitButton = CreateWindowW(L"BUTTON", L"Quit", WS_CHILD | BS_PUSHBUTTON, 0, 0, QuitButtonLength,
                                 StatusElHeight, hwMain, (HMENU)(UINT_PTR)QUIT_BUTTON_ID, hInst, NULL);
    hwStopButton = CreateWindowW(L"BUTTON", L"Stop", WS_CHILD | BS_PUSHBUTTON, 0, 0, QuitButtonLength,
                                 StatusElHeight, hwMain, (HMENU)(UINT_PTR)STOP_BUTTON_ID, hInst, NULL);
#endif

    if (!hwQuitButton)
        goto THE_END;

    SetWindowFont(hwQuitButton, efont, FALSE);
    SetWindowFont(hwStopButton, efont, FALSE);

    /* Define a minimum width */
    int MinWidth = AnalyseLength + SourceLength + QuitButtonLength + QuitButtonLength + 48;
    if (WinLineWidth < MinWidth)
        WinLineWidth = MinWidth;

    /* Make main window and subwindows visible.
       Size of windows allows display of 80 character line.
       Limit window to screen size (if only VGA). */
    if (WinLineWidth > ix)
        WinLineWidth = ix;
#ifndef BIG_WINDOW_FOR_DEBUGGING
    MoveWindow(hwMain, 0, (iyt * 2), WinLineWidth, iyt, FALSE);
#endif
    ShowWindow(hwMain, nShowState);
    ShowWindow(twText, SW_SHOWNORMAL);
    ShowWindow(swString, SW_SHOWNORMAL);
    ShowWindow(hwSource, SW_SHOWNORMAL);
    ShowWindow(hwAnalyse, SW_SHOWNORMAL);
    ShowWindow(hwQuitButton, SW_SHOWNORMAL);
    ShowWindow(hwStopButton, SW_SHOWNORMAL);
    ClearInput();
    DisplayText();
    SetSource("");
    SetAnalyse("Start", 0);
    UpdateWindow(hwMain);
    SetFocus(swString);

    status = MakeArgcArgv(lpszCmdLine, &argc, &argv);

    /* Wait until everything is settled */
    WaitForIdle();

    /* Go to main() */
    nReturnCode = xmain(argc, argv);

THE_END:
    /* terminate */

    /* Free history information if initialized */
    if (p_hi != (struct History_info *) NULL) {
        history_free(p_hi);
    }

    return nReturnCode;
} /* end of function WinMain */



/* This funtion initializes the history buffering with a welcome command */
static struct History_info *init_history(void)
{
    static struct History_info_opt hi_opt = {
        sizeof hi_opt,
        HIST_SIZE, HIST_SIZE, N_BYTE_HIST_BUF,
        4, 20, 10
    };

    struct History_info *p_hi = history_init(&hi_opt);
    if (p_hi == (struct History_info *) NULL) {
        return (struct History_info *) NULL;
    }

    {
        /* Initialize history buffer with empty input line */
        static const char cmd_welcome[] = "";
        (void) history_add(&p_hi, sizeof cmd_welcome - 1, cmd_welcome);
    }

    return p_hi;
} /* end of function init_history */



// -----------------------------------<User-IO>--------------------------------

/* Eigentlich wollte ich die Standard-Streams durch einen Hook in der Library umleiten,
   aber so etwas gibt es anscheinend nicht. Deswegen musz ich praktisch alle
   IO-Funktionen umdefinieren (siehe wstdio.h). Leider geht das nicht bei allen.
   Man schaue also nach, bevor man eine Funktion benutzt!
*/

int
win_x_fflush(FILE *stream)
{
    if (((stream == stdout) && !flogp) || (stream == stderr))
        return 0;
    else
        return fflush(stream);
}


int
win_x_fgetc(FILE *stream)
{
    if (stream == stdin) {
        int c;
        do
            c = w_getch();
        while (c == CR);
        return c;
    } else
        return fgetc(stream);
}


int
win_x_fgetpos(FILE *stream, fpos_t *pos)
{
    int result;
    if ((stream == stdin) || ((stream == stdout) && !flogp) || (stream == stderr)) {
        assert(FALSE);
        return 0;
    } else
        result = fgetpos(stream, pos);
    return result;
}


char *
win_x_fgets(char *s, int n, FILE *stream)
{
    if (stream == stdin) {
        int i = 0;
        int c;
        while (i < (n - 1)) {
            c = w_getch();
            if (c == LF) {
                s[i++] = LF;
                break;
            }
            if (c != CR)
                s[i++] = (char)c;
        }
        s[i] = SE;
        return s;
    } else
        return fgets(s, n, stream);
}


int
win_x_fputc(int c, FILE *stream)
{
    if (!flogp && ((stream == stdout) || (stream == stderr))) {
        if (c == LF)
            w_putch(CR);
        return w_putch(c);
//   Ausgabe in Datei *.log  14.6.2000
    } else if (flogp && ((stream == stdout) || stream == stderr)) {
        return fputc(c, flogp);
    } else
        return fputc(c, stream);
}


int
win_x_fputs(const char *s, FILE *stream)
{
//  if (((stream == stdout) && !flogp) || (stream == stderr)) {    hvogt 14.6.2000
    if ((stream == stdout) || (stream == stderr)) {

        int c = SE;
        if (!s)
            return EOF;
        for (;;) {
            if (*s) {
                c = *s++;
                win_x_fputc(c, stream);
            } else
                return c;
        }
    } else
        return fputs(s, stream);
}


int
win_x_fprintf(FILE *stream, const char *format, ...)
{
    int result;
    char s[IOBufSize];
    va_list args;

    va_start(args, format);

//  if (((stream == stdout) && !flogp) || (stream == stderr)) {
    if ((stream == stdout) || (stream == stderr)) {

        s[0] = SE;
        result = vsprintf(s, format, args);
        win_x_fputs(s, stream);
    } else
        result = vfprintf(stream, format, args);

    va_end(args);
    return result;
}


int
win_x_fclose(FILE *stream)
{
    if ((stream == stdin) || ((stream == stdout) && !flogp) || (stream == stderr)) {
        assert(FALSE);
        return 0;
    }
    return fclose(stream);
}


size_t
win_x_fread(void *ptr, size_t size, size_t n, FILE *stream)
{
//  if ((stream == stdin) || ((stream == stdout) && !flogp) || (stream == stderr)) {
    if (((stream == stdout) && !flogp) || (stream == stderr)) {
        assert(FALSE);
        return 0;
    }

    if (stream == stdin) {
        size_t i = 0;
        int c;
        char *s = (char *) ptr;
        while (i < (size * n - 1)) {
            c = w_getch();
            if (c == LF) {
//              s[i++] = LF;
                break;
            }
            if (c != CR) {
                s[i++] = (char) c;
            }
        }
//      s[i] = SE;
        return (size_t) (i / size);
    } /* end of case of stdin */

    return fread(ptr, size, n, stream);
}


FILE *
win_x_freopen(const char *path, const char *mode, FILE *stream)
{
    if ((stream == stdin)/* || ((stream == stdout) && !flogp) || (stream == stderr)*/) {
        assert(FALSE);
        return 0;
    }
    return freopen(path, mode, stream);
}


int
win_x_fscanf(FILE *stream, const char *format, ...)
{
    int result;
    va_list args;

    va_start(args, format);
    if ((stream == stdin) || ((stream == stdout) && !flogp) || (stream == stderr)) {
        assert(FALSE);
        return 0;
    }
    result = vfscanf(stream, format, args);
    va_end(args);
    return result;
}


int
win_x_fseek(FILE *stream, long offset, int whence)
{
    if ((stream == stdin) || ((stream == stdout) && !flogp) || (stream == stderr)) {
        assert(FALSE);
        return 0;
    }
    return fseek(stream, offset, whence);
}


int
win_x_fsetpos(FILE *stream, const fpos_t *pos)
{
    if ((stream == stdin) || ((stream == stdout) && !flogp) || (stream == stderr)) {
        assert(FALSE);
        return 0;
    }
    return fsetpos(stream, pos);
}


long
win_x_ftell(FILE *stream)
{
    if ((stream == stdin) || ((stream == stdout) && !flogp) || (stream == stderr)) {
        assert(FALSE);
        return 0;
    }
    return ftell(stream);
}


size_t
win_x_fwrite(const void *ptr, size_t size, size_t n, FILE *stream)
{
//  win_x_printf("entered fwrite, size %d, n %d \n", size, n);
    if (stream == stdin) {
        assert(FALSE);
//      win_x_printf("False \n");
        return 0;
    }
    if ((stream == stdout) || (stream == stderr)) {
        const char *s = ptr;
        int c = SE;
        size_t i = 0;
//      char *out;

//      win_x_printf("test1 %s\n", s);

        if (!s)
            return 0 /* EOF */;

        for (i = 0; i < (size * n); i++) {
            if (*s) {
                c = *s++;
                win_x_fputc(c, stream);
            } else
                break;
        }
//      win_x_fread(out, size, n, stream);
//      win_x_printf("test2 %s", out);
        return (int)(i / size);
    }
//  win_x_printf("test3 %s\n", ptr);
    return fwrite(ptr, size, n, stream);
}


char *
win_x_gets(char *s)
{
    return win_x_fgets(s, 10000, stdin);
}


void
win_x_perror(const char *s)
{
    const char *cp;
//  char s[IOBufSize];
    cp = strerror(errno);
    win_x_fprintf(stderr, "%s: %s\n", s, cp);
    /* output to message box
       sprintf(s, "%s: %s\n", s, cp);
       if (!flogp) winmessage(s);*/
}


int
win_x_printf(const char *format, ...)
{
    int result;
    char s[IOBufSize];
    va_list args;

    va_start(args, format);
    s[0] = SE;
    result = vsprintf(s, format, args);
    win_x_fputs(s, stdout);
    va_end(args);

    return result;
}


int
win_x_puts(const char *s)
{
    return win_x_fputs(s, stdout);
}


int
win_x_scanf(const char *format, ...)
{
    NG_IGNORE(format);
    assert(FALSE);
    return FALSE;
}


int
win_x_ungetc(int c, FILE *stream)
{
    NG_IGNORE(c);
    NG_IGNORE(stream);
    assert(FALSE);
    return FALSE;
}


int
win_x_vfprintf(FILE *stream, const char *format, void *arglist)
{
    int result;
    char s[IOBufSize];

    s[0] = SE;
//  if (((stream == stdout) && !flogp) || (stream == stderr)) {
    if ((stream == stdout) || (stream == stderr)) {

        result = vsprintf(s, format, arglist);
        win_x_fputs(s, stdout);
    } else
        result = vfprintf(stream, format, arglist);
    return result;
}


#if 0
int
win_x_vfscanf(FILE *stream, const char *format, void *arglist)
{
    if (stream == stdin) {
        assert(FALSE);
        return 0;
    }
    return vfscanf(stream, format, arglist);
}
#endif


int
win_x_vprintf(const char *format, void *arglist)
{
    int result;
    char s[IOBufSize];

    s[0] = SE;
    result = vsprintf(s, format, arglist);
    win_x_fputs(s, stdout);
    return result;
}


#if 0
int
win_x_vscanf(const char *format, void *arglist)
{
    assert(FALSE);
    return FALSE;
}
#endif


int
win_x_getc(FILE *fp)
{
    return win_x_fgetc(fp);
}


int
win_x_getchar(void)
{
    return win_x_fgetc(stdin);
}


int
win_x_putchar(const int c)
{
    return win_x_fputc(c, stdout);
}


int
win_x_putc(const int c, FILE *fp)
{
    return win_x_fputc(c, fp);
}


int
win_x_feof(FILE *fp)
{
    if ((fp == stdin) || (fp == stdout) || (fp == stderr)) {
        assert(FALSE);
        return 0;
    }
    return feof(fp);
}


int
win_x_ferror(FILE *fp)
{
    if ((fp == stdin) || (fp == stdout) || (fp == stderr)) {
        assert(FALSE);
        return 0;
    }
    return ferror(fp);
}


int
win_x_fputchar(int c)
{
    return win_x_fputc(c, stdout);
}


// --------------------------<Verfuegbarer Speicher>----------------------------

#if 0
size_t
_memavl(void)
{
    MEMORYSTATUS ms;
    DWORD sum;
    ms.dwLength = sizeof(MEMORYSTATUS);
    GlobalMemoryStatus(&ms);
    sum = ms.dwAvailPhys + ms.dwAvailPageFile;
    return (size_t) sum;
}
#endif


// ---------------------<Aufruf eines anderen Programms>-----------------------

#if 0
#ifndef _MSC_VER
int
system(const char *command)
{
    // info-Bloecke
    STARTUPINFO si;
    PROCESS_INFORMATION pi;
    DWORD ExitStatus;

    // Datenstrukturen fuellen
    memset(&si, 0, sizeof(STARTUPINFO));
    si.cb = sizeof(STARTUPINFO);
    memset(&pi, 0, sizeof(PROCESS_INFORMATION));

    // starte den neuen Prozess
    if (!CreateProcess(
            NULL,   // address of module name
            (char *) command,   // address of command line
            NULL,   // address of process security attributes
            NULL,   // address of thread security attributes
            FALSE,  // new process inherits handles
            NORMAL_PRIORITY_CLASS,  // creation flags
            NULL,   // address of new environment block
            NULL,   // address of current directory name
            &si,    // address of STARTUPINFO
            &pi     // address of PROCESS_INFORMATION
            ))
        return -1;

    // dieses Handle musz da sein
    if (!pi.hProcess)
        return -1;

    do {
        // Multitasking ermoeglichen
        WaitForIdle();
        // hole mir den Exit-Code des Prozesses
        if (!GetExitCodeProcess(pi.hProcess, &ExitStatus))
            return -1;
        // solange er existiert
    } while (ExitStatus == STILL_ACTIVE);

    // Handles freigeben
    if (pi.hThread)
        CloseHandle(pi.hThread);
    if (pi.hProcess)
        CloseHandle(pi.hProcess);

    // fertig
    return 0;
} // system Windows95
#endif
#endif


#ifdef __CYGWIN__

/* Strip leading spaces, return a copy of s */
static char *
rlead(char *s)
{
    int i, j = 0;
    static char temp[512];
    bool has_space = TRUE;
    for (i = 0; s[i] != '\0'; i++)
    {
        if (isspace((unsigned char) s[i]) && has_space)
        {
            ; //Do nothing
        }
        else
        {
            temp[j] = s[i];
            j++;
            has_space = FALSE;
        }
    }
    temp[j] = '\0';
    return copy(temp);
}

#endif


void
winmessage(char *new_msg)
{
    /* open a message box only if message is not written into -o xxx.log */
    if (!flogp)
        MessageBox(NULL, new_msg, "Ngspice Info", MB_OK | MB_ICONERROR);
}

void
UpdateMainText(void) {
    DisplayText();
}

#else /* HAS_WINGUI not defined */
/* Prevent warning regarding empty translation unit */
static void dummy(void)
{
    return;
} /* end of function dummy */



#endif /* HAS_WINGUI */
