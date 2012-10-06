/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: Jeffrey M. Hsu
Modified 1999 Emmanuel Rouat
**********/

#include "ngspice/ngspice.h"

#ifndef X_DISPLAY_MISSING

#include "ngspice/cpstd.h"
#include "ngspice/hlpdefs.h"
#include <X11/Xaw/AsciiText.h>
#include <X11/StringDefs.h>
#include <X11/Xaw/Paned.h>
#include <X11/Xaw/Label.h>
#include <X11/Xaw/Viewport.h>
#include <X11/Xaw/Command.h>
#include <X11/Xaw/Box.h>
#include <X11/Shell.h>

static bool started = FALSE;
static topic *topics = NULL;
static void newtopic(Widget w, XtPointer client_data, XtPointer call_data);
static void delete_w(Widget w, XtPointer client_data, XtPointer call_data);
static void quit(Widget w, XtPointer client_data, XtPointer call_data);
static void sputline(char *buf, char *s);
/* atoms for catching window delet by WM x-button */
static Atom atom_wm_delete_window;
static Atom atom_wm_protocols;
static Display *display;


/* callback function for catching window deletion by WM x-button */
static void
handle_wm_messages(Widget w, XtPointer client_data, XEvent *event, Boolean *cont)
{
    topic *top = (topic *) client_data;

    NG_IGNORE(cont);
    NG_IGNORE(w);

    if (event->type == ClientMessage &&
        event->xclient.message_type == atom_wm_protocols &&
        (Atom) event->xclient.data.l[0] == atom_wm_delete_window)
    {
        hlp_killfamily(top);
        hlp_fixchildren(top);
    }
}


/* Create a new window... */
bool
hlp_xdisplay(topic *top)
{

    toplink *tl;
    handle *hand;

    wordlist *wl;
    char *buf;

    static Arg titleargs[] = {
        { XtNtop, (XtArgVal) XtChainTop },
        { XtNbottom, (XtArgVal) XtChainTop },
        { XtNleft, (XtArgVal) XtChainLeft },
        { XtNright, (XtArgVal) XtChainLeft },
        { XtNwidth, (XtArgVal) 650 },
    };

    static Arg formargs[ ] = {
        { XtNtop, (XtArgVal) XtChainTop },
        { XtNtop, (XtArgVal) XtChainTop },
        { XtNtop, (XtArgVal) XtChainTop },
        { XtNbottom, (XtArgVal) XtChainBottom },
    };
    Arg htextargs[7];
/*  Arg vportargs[5]; */
    static Arg bboxargs[ ] = {
        { XtNtop, (XtArgVal) XtChainBottom },
        { XtNtop, (XtArgVal) XtChainBottom },
        { XtNtop, (XtArgVal) XtChainBottom },
        { XtNtop, (XtArgVal) XtChainBottom },
        { XtNbottom, (XtArgVal) XtChainBottom },
        { XtNleft, (XtArgVal) XtChainLeft },
        { XtNright, (XtArgVal) XtChainLeft },
    };
    Arg buttonargs[1];
    Arg labelargs[3];
    Widget buttonwidget;

    if (!started) {     /* have to init everything */

        /* assume X toolkit already initialize */

        started = TRUE;

    }

    top->shellwidget = XtCreateApplicationShell
        ("shell", topLevelShellWidgetClass, NULL, 0);

    if (!top->parent) {
        top->xposition = hlp_initxpos;
        top->yposition = hlp_initypos;
    } else {
        top->xposition = top->parent->xposition + X_INCR;
        top->yposition = top->parent->yposition + Y_INCR;
    }

    XtSetArg(formargs[0], XtNx, top->xposition);
    XtSetArg(formargs[1], XtNy, top->yposition);
    top->formwidget = XtCreateManagedWidget
        ("form", formWidgetClass, top->shellwidget, formargs, XtNumber(formargs));

    /* we really want a title bar widget for this, sigh */
    top->titlewidget = XtCreateManagedWidget
        ("title", boxWidgetClass, top->formwidget, titleargs, XtNumber(titleargs));
    XtSetArg(labelargs[0], XtNlabel, top->title);
    XtCreateManagedWidget
        ("titlelabel", labelWidgetClass, top->titlewidget, labelargs, 1);
    XtSetArg(buttonargs[0], XtNlabel, "quit help");
    buttonwidget = XtCreateManagedWidget
        ("quit", commandWidgetClass, top->titlewidget, buttonargs, 1);
    XtAddCallback(buttonwidget, XtNcallback, quit, top);
    XtSetArg(buttonargs[0], XtNlabel, "delete window");
    buttonwidget = XtCreateManagedWidget
        ("delete", commandWidgetClass, top->titlewidget, buttonargs, XtNumber(buttonargs));
    XtAddCallback(buttonwidget, XtNcallback, delete_w, top);

    buf = TMALLOC(char, 80 * top->numlines + 100);
    buf[0] = '\0';
    for (wl = top->text; wl; wl = wl->wl_next) {
        sputline(buf, wl->wl_word);
    }
    top->chartext = buf;  /* make sure gets deallocated later XXX */
    XtSetArg(htextargs[0], XtNstring, top->chartext);
    XtSetArg(htextargs[1], XtNallowResize, True);
    XtSetArg(htextargs[2], XtNscrollHorizontal, XawtextScrollWhenNeeded);
    XtSetArg(htextargs[3], XtNscrollVertical, XawtextScrollAlways);
    XtSetArg(htextargs[4], XtNfromVert, top->titlewidget);
    XtSetArg(htextargs[5], XtNwidth, 660);
    XtSetArg(htextargs[6], XtNheight, 350);
    top->textwidget = XtCreateManagedWidget
        ("helptext", asciiTextWidgetClass, top->formwidget, htextargs, XtNumber(htextargs));

    if (top->subtopics) {
        XtSetArg(labelargs[0], XtNfromVert, top->textwidget);
        XtSetArg(labelargs[1], XtNvertDistance, 8);
        XtSetArg(labelargs[2], XtNlabel, "Subtopics: ");
        top->sublabelwidget = XtCreateManagedWidget
            ("sublabel", labelWidgetClass, top->formwidget, labelargs, XtNumber(labelargs));

        XtSetArg(bboxargs[0], XtNwidth, 400);
        XtSetArg(bboxargs[1], XtNallowResize, True);
        XtSetArg(bboxargs[2], XtNfromVert, top->sublabelwidget);
        top->subboxwidget = XtCreateManagedWidget
            ("buttonbox", boxWidgetClass, top->formwidget, bboxargs, XtNumber(bboxargs));

        for (tl = top->subtopics; tl; tl = tl->next) {
            tl->button.text = tl->description;
            tl->button.tag = tl->place;
            if (!tl->button.text)
                tl->button.text = "<unknown>";

            XtSetArg(buttonargs[0], XtNlabel, tl->button.text);
            buttonwidget = XtCreateManagedWidget
                (tl->button.text, commandWidgetClass, top->subboxwidget, buttonargs, XtNumber(buttonargs));
            /* core leak XXX */
            hand = TMALLOC(handle, 1);
            hand->result = tl;
            hand->parent = top;
            XtAddCallback(buttonwidget, XtNcallback, newtopic, hand);
        }
    }

    if (top->seealso) {
        if (top->subtopics)
            XtSetArg(labelargs[0], XtNfromVert, top->subboxwidget);
        else
            XtSetArg(labelargs[0], XtNfromVert, top->textwidget);
        XtSetArg(labelargs[1], XtNvertDistance, 8);
        XtSetArg(labelargs[2], XtNlabel, "See also: ");
        top->seelabelwidget = XtCreateManagedWidget
            ("seelabel", labelWidgetClass, top->formwidget, labelargs, XtNumber(labelargs));

        XtSetArg(bboxargs[0], XtNwidth, 400);
        XtSetArg(bboxargs[1], XtNallowResize, True);
        XtSetArg(bboxargs[2], XtNfromVert, top->seelabelwidget);
        top->seeboxwidget = XtCreateManagedWidget
            ("buttonbox", boxWidgetClass, top->formwidget, bboxargs, XtNumber(bboxargs));

        for (tl = top->seealso; tl; tl = tl->next) {
            tl->button.text = tl->description;
            tl->button.tag = tl->place;
            if (!tl->button.text)
                tl->button.text = "<unknown>";

            XtSetArg(buttonargs[0], XtNlabel, tl->button.text);
            buttonwidget = XtCreateManagedWidget
                (tl->button.text, commandWidgetClass, top->seeboxwidget, buttonargs, 1);
            hand = TMALLOC(handle, 1);
            /* core leak XXX */
            hand->result = tl;
            hand->parent = top;
            XtAddCallback(buttonwidget, XtNcallback, newtopic, hand);
        }
    }

    XtRealizeWidget(top->shellwidget);

    top->winlink = topics;
    topics = top;


    /* WM_DELETE_WINDOW protocol */
    display = XtDisplay(top->shellwidget);
    atom_wm_protocols = XInternAtom(display, "WM_PROTOCOLS", False);
    atom_wm_delete_window = XInternAtom(display, "WM_DELETE_WINDOW", False);
    XtAddEventHandler(top->shellwidget, NoEventMask, True, handle_wm_messages, top);
    XSetWMProtocols(display, XtWindow(top->shellwidget), &atom_wm_delete_window, 1);

    return (TRUE);
}


static void
newtopic(Widget w, XtPointer client_data, XtPointer call_data)
{
    topic *parent = ((handle *) client_data)->parent;
    toplink *result = ((handle *) client_data)->result;
    topic *newtop;

    NG_IGNORE(call_data);
    NG_IGNORE(w);

    if (!(newtop = hlp_read(result->place))) {
        fprintf(stderr, "Internal error: bad link\n");
    }

    newtop->next = parent->children;
    parent->children = newtop;
    newtop->parent = parent;
    newtop->xposition = parent->xposition + 50;
    newtop->yposition = parent->yposition + 50;
    if (!hlp_xdisplay(newtop)) {
        fprintf(stderr, "Couldn't open win\n");
        return;
    }
}


static void
delete_w(Widget w, XtPointer client_data, XtPointer call_data)
{
    topic *top = (topic *) client_data;

    NG_IGNORE(call_data);
    NG_IGNORE(w);

    hlp_killfamily(top);
    hlp_fixchildren(top);
}


static void
quit(Widget w, XtPointer client_data, XtPointer call_data)
{
    topic *top = (topic *) client_data, *parent = top->parent;

    NG_IGNORE(call_data);
    NG_IGNORE(w);

    while (parent && parent->parent)
        parent = parent->parent;

    hlp_killfamily(parent ? parent : top);
}


void
hlp_xkillwin(topic *top)
{
    topic *last;

    if (top == topics)
        topics = top->winlink;
    else if (top->winlink) {        /* we need this check for the
                                       pathological case where you have two helps running,
                                       normally hp_killfamily doesn't let this happen */
        for (last = topics; last->winlink; last = last->winlink)
            if (last->winlink == top) {
                last->winlink = top->winlink;
                break;
            }
        if (!last->winlink) {
            fprintf(stderr, "window not in list!!\n");
            return;
        }
    }
    XtDestroyWidget(top->shellwidget);
}


/* rip out font changes and write at end of buffer */
static void
sputline(char *buf, char *s)
{

    char tmp[BSIZE_SP], *tmpp;
    int i = 0;

    while (*s) {
        if (((*s == '\033') && s[1]) ||
            ((*s == '_') && (s[1] == '\b')))
            s += 2;
        else
            tmp[i++] = *s++;
    }

    tmp[i] = '\0';

    /* strcat can't handle long strings */
    tmpp = buf + strlen(buf);
    sprintf(tmpp, "%s\n", tmp);
}

#endif  /* X_DISPLAY_MISSING */
