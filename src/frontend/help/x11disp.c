/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: Jeffrey M. Hsu
Modified 1999 Emmanuel Rouat
**********/

#include <config.h>
#include "ngspice.h"

#ifndef X_DISPLAY_MISSING

#include "cpstd.h"
#include "hlpdefs.h"
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
void newtopic(Widget w, caddr_t client_data, caddr_t call_data), delete(Widget w, caddr_t client_data, caddr_t call_data), quit(Widget w, caddr_t client_data, caddr_t call_data); 
static void sputline(char *buf, char *s);

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

    top->shellwidget = XtCreateApplicationShell("shell",
            topLevelShellWidgetClass, NULL, 0);

    if (!top->parent) {
      top->xposition = hlp_initxpos;
      top->yposition = hlp_initypos;
    } else {
      top->xposition = top->parent->xposition + X_INCR;
      top->yposition = top->parent->yposition + Y_INCR;
    }
    XtSetArg(formargs[0], XtNx, top->xposition);
    XtSetArg(formargs[1], XtNy, top->yposition);
    top->formwidget = XtCreateManagedWidget("form", formWidgetClass,
            top->shellwidget, formargs, XtNumber(formargs));

    /* we really want a title bar widget for this, sigh */
    top->titlewidget = XtCreateManagedWidget("title",
            boxWidgetClass, top->formwidget,
            titleargs, XtNumber(titleargs));
    XtSetArg(labelargs[0], XtNlabel, top->title);
    XtCreateManagedWidget("titlelabel", labelWidgetClass,
            top->titlewidget, labelargs, 1);
    XtSetArg(buttonargs[0], XtNlabel, "quit help");
    buttonwidget = XtCreateManagedWidget("quit", commandWidgetClass,
            top->titlewidget, buttonargs, 1);
    XtAddCallback(buttonwidget, XtNcallback, (XtCallbackProc) quit, top);
    XtSetArg(buttonargs[0], XtNlabel, "delete window");
    buttonwidget = XtCreateManagedWidget("delete", commandWidgetClass,
            top->titlewidget, buttonargs, XtNumber(buttonargs));
    XtAddCallback(buttonwidget, XtNcallback, (XtCallbackProc) delete, top);

    buf = tmalloc(80 * top->numlines + 100);
    buf[0] = '\0';
    for (wl = top->text; wl; wl = wl->wl_next) {
      sputline(buf, wl->wl_word);
    }
    top->chartext = buf;  /* make sure gets deallocated later XXX */
    XtSetArg(htextargs[0], XtNstring, top->chartext);
    XtSetArg(htextargs[1], XtNallowResize, True);
    XtSetArg(htextargs[2], XtNscrollHorizontal, True );
    XtSetArg(htextargs[3], XtNscrollVertical, True );
    XtSetArg(htextargs[4], XtNfromVert, top->titlewidget);
    XtSetArg(htextargs[5], XtNwidth, 660);
    XtSetArg(htextargs[6], XtNheight, 350);
    top->textwidget = XtCreateManagedWidget("helptext",
           asciiTextWidgetClass, top->formwidget, htextargs,
	   XtNumber(htextargs));

    if (top->subtopics) {
      XtSetArg(labelargs[0], XtNfromVert, top->textwidget);
      XtSetArg(labelargs[1], XtNvertDistance, 8);
      XtSetArg(labelargs[2], XtNlabel, "Subtopics: ");
      top->sublabelwidget = XtCreateManagedWidget("sublabel",
          labelWidgetClass, top->formwidget, labelargs, XtNumber(labelargs));

      XtSetArg(bboxargs[0], XtNwidth, 400);
      XtSetArg(bboxargs[1], XtNallowResize, True);
      XtSetArg(bboxargs[2], XtNfromVert, top->sublabelwidget);
      top->subboxwidget = XtCreateManagedWidget("buttonbox",
          boxWidgetClass, top->formwidget, bboxargs, XtNumber(bboxargs));

      for (tl = top->subtopics; tl; tl = tl->next) {
        tl->button.text = tl->description;
        tl->button.tag = tl->place;
        if (!tl->button.text)
          tl->button.text = "<unknown>";

        XtSetArg(buttonargs[0], XtNlabel, tl->button.text);
        buttonwidget = XtCreateManagedWidget(tl->button.text,
		commandWidgetClass, top->subboxwidget, buttonargs,
		XtNumber(buttonargs));
                                /* core leak XXX */
        hand = (handle *) tmalloc(sizeof (struct handle));
        hand->result = tl;
        hand->parent = top;
        XtAddCallback(buttonwidget, XtNcallback, (XtCallbackProc) newtopic, hand);
      }
    }

    if (top->seealso) {
      if (top->subtopics)
        XtSetArg(labelargs[0], XtNfromVert, top->subboxwidget);
      else
        XtSetArg(labelargs[0], XtNfromVert, top->textwidget);
      XtSetArg(labelargs[1], XtNvertDistance, 8);
      XtSetArg(labelargs[2], XtNlabel, "See also: ");
      top->seelabelwidget = XtCreateManagedWidget("seelabel",
          labelWidgetClass, top->formwidget, labelargs, XtNumber(labelargs));

      XtSetArg(bboxargs[0], XtNwidth, 400);
      XtSetArg(bboxargs[1], XtNallowResize, True);
      XtSetArg(bboxargs[2], XtNfromVert, top->seelabelwidget);
      top->seeboxwidget = XtCreateManagedWidget("buttonbox",
          boxWidgetClass, top->formwidget, bboxargs, XtNumber(bboxargs));

      for (tl = top->seealso; tl; tl = tl->next) {
        tl->button.text = tl->description;
        tl->button.tag = tl->place;
        if (!tl->button.text)
          tl->button.text = "<unknown>";

        XtSetArg(buttonargs[0], XtNlabel, tl->button.text);
        buttonwidget = XtCreateManagedWidget(tl->button.text,
		commandWidgetClass, top->seeboxwidget, buttonargs, 1);
		hand = (handle *) tmalloc(sizeof (struct handle));
                                /* core leak XXX */
        hand->result = tl;
        hand->parent = top;
        XtAddCallback(buttonwidget, XtNcallback, (XtCallbackProc) newtopic, hand);
      }
    }

    XtRealizeWidget(top->shellwidget);

    top->winlink = topics;
    topics = top;

    return (TRUE);

}

void
newtopic(Widget w, caddr_t client_data, caddr_t call_data)
{
    topic *parent = ((handle *) client_data)->parent;
    toplink *result = ((handle *) client_data)->result;
    topic *newtop;

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

void
delete(Widget w, caddr_t client_data, caddr_t call_data)
{

    topic *top = (topic *) client_data;

    hlp_killfamily(top);
    hlp_fixchildren(top);
}

void
quit(Widget w, caddr_t client_data, caddr_t call_data)
{

    topic *top = (topic *) client_data, *parent = top->parent;

    while (parent && parent->parent) parent = parent->parent;
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
    return;
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

    return;
}

#endif  /* X_DISPLAY_MISSING */
