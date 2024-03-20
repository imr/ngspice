/* Commands for opening and reading arbitrary files. */

#include <stdio.h>
#include <string.h>
#include <errno.h>

#include "ngspice/ngspice.h"

#include "ngspice/bool.h"
#include "ngspice/wordlist.h"

#include "com_strcmp.h"
#include "variable.h"

/* Track open files with these structures, indexed by the underlying
 * descriptor.  Not many should be needed.
 */

#define MAX_OPEN_FILES 20
#define MAX_TEXT_LINE  8192

static struct {
    FILE       *fp;
    char       *name;
} Open_Files[MAX_OPEN_FILES];

/* Check whether error messages should be suppressed.  That is useful when
 * opening a file to see if it exists.
 */

static int verbose(void)
{
    return !cp_getvar("silent_fileio", CP_BOOL, NULL, 0);
}

/* fopen handle file_name [mode]
 *
 * For example: fopen handle result.txt r
 *
 * The underlying file descriptor (or -1) is returned in the variable "handle".
 */

void com_fopen(wordlist *wl)
{
    char *var, *file_name, *mode;
    FILE *fp;
    int   fd;

    var = wl->wl_word;
    wl = wl->wl_next;
    file_name = cp_unquote(wl->wl_word);
    wl = wl->wl_next;
    mode = wl ? cp_unquote(wl->wl_word) : "r";
    fp = fopen(file_name, mode);
    if (fp) {
        fd = fileno(fp);
        if (fd < MAX_OPEN_FILES) {
            if (Open_Files[fd].fp) // Not expected!
                fclose(Open_Files[fd].fp);
            if (Open_Files[fd].name)
                tfree(Open_Files[fd].name);
            Open_Files[fd].fp = fp;
            Open_Files[fd].name = copy(file_name);
        } else {
            fclose(fp);
            fprintf(stderr,
                    "com_fopen() cannot open %s: too many open files\n",
                    file_name);
            fd = -1;
        }
    } else {
        fd = -1;
        if (verbose()) {
            fprintf(stderr, "com_fopen() cannot open %s: %s\n",
                    file_name, strerror(errno));
        }
    }
    tfree(file_name);
    if (wl)
        tfree(mode);
    cp_vset(var, CP_NUM, &fd);
}

/* Command looks like:
 *   fread result handle [length]
 * where handle is a small positive integer, result names a variable
 * and length is the name of a variable used to return the length of the line.
 * The returned length is -1 at EOF, -2 on failure.
 */

void com_fread(wordlist *wl)
{
    char *handle, *result, *lvar;
    int   fd, length;
    char  buf[MAX_TEXT_LINE];

    result = cp_unquote(wl->wl_word);
    wl = wl->wl_next;
    handle = cp_unquote(wl->wl_word);
    fd = atoi(handle);
    tfree(handle);
    wl = wl->wl_next;
    if (wl)
        lvar = cp_unquote(wl->wl_word);
    else
        lvar = NULL;

    if (fd >= 0 && fd < MAX_OPEN_FILES) {
        if (!Open_Files[fd].fp) {
            /* Allow stdin, for example. */

            Open_Files[fd].fp = fdopen(fd, "r");
            if (!Open_Files[fd].fp && verbose()) {
                fprintf(stderr, "com_fread() cannot open handle %d\n", fd);
                goto err;
            }
        }

        if (fgets(buf, sizeof buf, Open_Files[fd].fp)) {
            length = (int)strlen(buf);
            if (length > 0 && buf[length - 1] == '\n') {
                --length;
                if (length > 0 && buf[length - 1] == '\r') {
                    /* Windows CRLF line termination. */

                    --length;
                }
                buf[length] = '\0';
            } else if (verbose()) {
                fprintf(stderr,
                        "com_fread() found line in %s "
                        "too long for buffer\n",
                        Open_Files[fd].name);
            }
        } else {
            if (feof(Open_Files[fd].fp)) {
                length = -1;
            } else if (verbose()) {
                fprintf(stderr,
                        "com_fread() error reading %s: %s\n",
                        Open_Files[fd].name, strerror(errno));
                length = -2;
            }
            *buf = '\0';
        }
    } else if (verbose()) {
        fprintf(stderr,
                "com_fread(): file handle %d is not in accepted range.\n",
                fd);
    err:
        length = -1;
        *buf = '\0';
    }
    cp_vset(result, CP_STRING, buf);
    tfree(result);
    if (lvar) {
        cp_vset(lvar, CP_NUM, &length);
        tfree(lvar);
    }
}

void com_fclose(wordlist *wl)
{
    char *handle;
    int   fd;

    handle = cp_unquote(wl->wl_word);
    fd = atoi(handle);
    tfree(handle);
    if (fd <= 2 || fd >= MAX_OPEN_FILES)
        return;
    if (Open_Files[fd].fp) {
        fclose(Open_Files[fd].fp);
        Open_Files[fd].fp = NULL;
    }
    if (Open_Files[fd].name)
        tfree(Open_Files[fd].name);
}



