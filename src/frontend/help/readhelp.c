/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Author: 1986 Wayne A. Christopher, U. C. Berkeley CAD Group
Modified 1999 Emmanuel Rouat
**********/

/*
 * SJB 20 May 2001
 * Bug fix in help_read()
 * findsubject() now ignores case and does additional searches for partial matches
 * when a complete match is not found - additional code based on code in MacSpice.
 */

#include "ngspice/ngspice.h"
#include "ngspice/cpstd.h"
#include "ngspice/hlpdefs.h"
#include "ngspice/suffix.h"

static char *getsubject(fplace *place);
static toplink *getsubtoplink(char **ss);

static topic *alltopics = NULL;

static fplace *copy_fplace(fplace *place);
static void hlp_topic_free(topic *p_topic);


static int
sortcmp(const void *a, const void *b)
{
    toplink **tlp1 = (toplink **) a;
    toplink **tlp2 = (toplink **) b;

    return (strcmp((*tlp1)->description, (*tlp2)->description));
}


static void
sortlist(toplink **tlp)
{
    toplink *tl;
    size_t num = 0, i;

    for (tl = *tlp; tl; tl = tl->next) {
        num++;
    }
    if (!num) { /* nothing to sort */
        return;
    }

    toplink ** const vec = TMALLOC(toplink *, num);
    for (tl = *tlp, i = 0; tl; tl = tl->next, i++) {
        vec[i] = tl;
    }
    (void) qsort(vec, num, sizeof (toplink *), sortcmp);
    *tlp = vec[0];
    for (i = 0; i < num - 1; i++) {
        vec[i]->next = vec[i + 1];
    }
    vec[i]->next = NULL;
    txfree(vec);
}


topic *
hlp_read(fplace *place)
{
    int xrc = 0;
    char buf[BSIZE_SP];
    topic *top = TMALLOC(topic, 1);
    toplink *topiclink;
    toplink *tl, *tend = NULL;
    wordlist *end = NULL;
    int i, fchanges;
    char *s;
    bool mof = FALSE;

    if (!place) {
        xrc = -1;
        goto EXITPOINT;
    }

    top->place = copy_fplace(place);

    /* get the title */
    if (!place->fp) {
        place->fp = hlp_fopen(place->filename);
    }
    if (!place->fp) {
        xrc = -1;
        goto EXITPOINT;
    }
    fseek(place->fp, place->fpos, SEEK_SET);

    /* skip subject */
    if (fgets(buf, BSIZE_SP, place->fp) == (char *) NULL) {
        fprintf(stderr, "missing subject\n");
        xrc = -1;
        goto EXITPOINT;
    }
    if (fgets(buf, BSIZE_SP, place->fp) == (char *) NULL) {
        fprintf(stderr, "missing title\n");
        xrc = -1;
        goto EXITPOINT;
    }

    for (s = buf; *s && (*s != '\n'); s++) {
        ;
    }
    *s = '\0';
    if ((int) (s - buf) < 7) {
        fprintf(stderr, "invalid title\n");
        xrc = -1;
        goto EXITPOINT;
    }
    top->title = copy(&buf[7]);     /* don't copy "TITLE: " */

    /* get the text */
    /* skip to TEXT: */
    while (fgets(buf, BSIZE_SP, place->fp)) {
        if (!strncmp("TEXT: ", buf, 6))
            break;
        if ((*buf == '\0') ||             /* SJB - bug fix */
            !strncmp("SEEALSO: ", buf, 9) ||
            !strncmp("SUBTOPIC: ", buf, 10)) {
            /* no text */
            top->text = NULL;
            goto endtext;
        }
    }
    mof = TRUE;
    while (mof && !strncmp("TEXT: ", buf, 6)) {
        for (s = &buf[6], fchanges = 0; *s && (*s != '\n'); s++)
            if (((s[0] == '\033') && s[1]) ||
                ((s[0] == '_') && (s[1] == '\b')))
                fchanges++;
        *s = '\0';
        wl_append_word(&(top->text), &end, copy(&buf[6]));
        top->numlines++;
        i = (int) strlen(&buf[6]) - fchanges;
        if (top->maxcols < i)
            top->maxcols = i;
        mof = fgets(buf, BSIZE_SP, place->fp) == NULL ? FALSE : TRUE;
    }
endtext:

    /* get subtopics */
    while(mof && !strncmp("SUBTOPIC: ", buf, 10)) {
        s = &buf[10];
        /* process tokens within line, updating pointer */
        while (*s) {
            if ((topiclink = getsubtoplink(&s)) != NULL) {
                if (tend)
                    tend->next = topiclink;
                else
                    top->subtopics = topiclink;
                tend = topiclink;
            }
        }
        mof = fgets(buf, BSIZE_SP, place->fp) == NULL ? FALSE : TRUE;
    }

    /* get see alsos */
    tend = NULL;
    while(mof && !strncmp("SEEALSO: ", buf, 9)) {
        s = &buf[9];
        /* process tokens within line, updating pointer */
        while (*s) {
            if ((topiclink = getsubtoplink(&s)) != NULL) {
                if (tend)
                    tend->next = topiclink;
                else
                    top->seealso = topiclink;
                tend = topiclink;
            }
        }
        mof = fgets(buf, BSIZE_SP, place->fp) == NULL ? FALSE : TRUE;
    }

    /* Now we have to fill in the subjects
       for the seealsos and subtopics. */
    for (tl = top->seealso; tl; tl = tl->next)
        tl->description = getsubject(tl->place);
    for (tl = top->subtopics; tl; tl = tl->next)
        tl->description = getsubject(tl->place);

    sortlist(&top->seealso);
    /* sortlist(&top->subtopics); It looks nicer if they
       are in the original order */

    top->readlink = alltopics;
    alltopics = top;

EXITPOINT:
    if (xrc != 0) { /* free resources if error */
        hlp_topic_free(top);
        top = (topic *) NULL;
    }

    return top;
} /* end of function hlp_read */



/* *ss is of the form filename:subject */
static toplink *
getsubtoplink(char **ss)
{
    toplink *tl;
    char *tmp, *s, *t;
    char subject[BSIZE_SP];

    if (!**ss)
        return (NULL);

    s = *ss;

    tl = TMALLOC(toplink, 1);
    if ((tmp =strchr(s, ':')) != NULL) {
        tl->place = TMALLOC(fplace, 1);
        tl->place->filename =
            strncpy(TMALLOC(char, tmp - s + 1), s, (size_t) (tmp - s));
        tl->place->filename[tmp - s] = '\0';
        strtolower(tl->place->filename);

        /* see if filename is on approved list */
        if (!hlp_approvedfile(tl->place->filename)) {
            tfree(tl->place);
            tfree(tl);
            /* skip up to next comma or newline */
            while (*s && *s != ',' && *s != '\n')
                s++;
            while (*s && (*s == ',' || *s == ' ' || *s == '\n'))
                s++;
            *ss = s;
            return (NULL);
        }

        tl->place->fp = hlp_fopen(tl->place->filename);
        for (s = tmp + 1, t = subject; *s && *s != ',' && *s != '\n'; s++)
            *t++ = *s;
        *t = '\0';
        tl->place->fpos = findsubject(tl->place->filename, subject);
        if (tl->place->fpos == -1) {
            tfree(tl->place);
            tfree(tl);
            while (*s && (*s == ',' || *s == ' ' || *s == '\n'))
                s++;
            *ss = s;
            return (NULL);
        }
    } else {
        fprintf(stderr, "bad filename:subject pair %s\n", s);
        /* skip up to next free space */
        while (*s && *s != ',' && *s != '\n')
            s++;
        while (*s && (*s == ',' || *s == ' ' || *s == '\n'))
            s++;
        *ss = s;
        tfree(tl->place);
        tfree(tl);
        return (NULL);
    }

    while (*s && (*s == ',' || *s == ' ' || *s == '\n'))
        s++;
    *ss = s;

    return (tl);
}


/* returns a file position, -1 on error */
long
findsubject(char *filename, char *subject)
{

    FILE *fp;
    char buf[BSIZE_SP];
    struct hlp_index indexitem;

    if (!filename) {
        return -1;
    }

    /* open up index for filename */
    sprintf(buf, "%s%s%s.idx", hlp_directory, DIR_PATHSEP, filename);
    hlp_pathfix(buf);
    if ((fp = fopen(buf, "rb")) == NULL) {
        perror(buf);
        return (-1);
    }

    /* try it exactly (but ignore case) */
    while(fread(&indexitem, sizeof (struct hlp_index), 1, fp)) {
        if (!strncasecmp(subject, indexitem.subject, 64)) { /* sjb - ignore case */
            fclose(fp);
            return (indexitem.fpos);
        }
    }

    fclose(fp);

    if ((fp = fopen(buf, "rb")) == NULL) {
        perror(buf);
        return (-1);
    }

    /* try it abbreviated (ignore case)  */
    while(fread(&indexitem, sizeof (struct hlp_index), 1, fp)) {
        if (!strncasecmp(indexitem.subject,subject, strlen(subject))) {
            fclose(fp);
            return (indexitem.fpos);
        }
    }

    fclose(fp);

    if ((fp = fopen(buf, "rb")) == NULL) {
        perror(buf);
        return (-1);
    }

    /* try it within */ /* FIXME: need a case independent version of strstr() */
    while(fread(&indexitem, sizeof (struct hlp_index), 1, fp)) {
        if (strstr(indexitem.subject,subject)) {
            fclose(fp);
            return (indexitem.fpos);
        }
    }

    fclose(fp);
    return (-1);
}


static char *
getsubject(fplace *place)
{
    char buf[BSIZE_SP], *s;

    if (!place->fp)
        place->fp = hlp_fopen(place->filename);
    if (!place->fp)
        return(NULL);

    fseek(place->fp, place->fpos, SEEK_SET);
    if (fgets(buf, BSIZE_SP, place->fp) == (char *) NULL) {
        (void) fprintf(stderr, "Missing subject");
        return (char *) NULL;
    }
    for (s = buf; *s && (*s != '\n'); s++)
        ;
    *s = '\0';
    if ((int) (s - buf) < 9) {
        (void) fprintf(stderr, "Invalid subject");
        return (char *) NULL;
    }
    return copy(buf + 9);     /* don't copy "SUBJECT: " */
}


static
void tlfree(toplink *tl)
{
    toplink *nt = NULL;

    while (tl) {
        txfree(tl->description);
        txfree(tl->place->filename);
        txfree(tl->place);
        /* Don't free the button stuff... */
        nt = tl->next;
        txfree(tl);
        tl = nt;
    }
}



void
hlp_free(void)
{
    topic *top, *nt;

    for (top = alltopics; top; top = nt) {
        nt = top->readlink;
        hlp_topic_free(top);
    }
    alltopics = NULL;
} /* end of function hlp_free */


static void hlp_topic_free(topic *p_topic)
{
    txfree(p_topic->title);
    txfree(p_topic->place);
    wl_free(p_topic->text);
    tlfree(p_topic->subtopics);
    tlfree(p_topic->seealso);
    txfree(p_topic);
} /* end of function hlp_topic_free */



static fplace *
copy_fplace(fplace *place)
{
    fplace *newplace;

    newplace = TMALLOC(fplace, 1);
    newplace->filename = copy(place->filename);
    newplace->fpos = place->fpos;
    newplace->fp = place->fp;

    return (newplace);
}
