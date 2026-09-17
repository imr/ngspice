/*
 * ngparse glue.  See ngspice/ngparse_glue.h for the design.
 */
#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/ngparse_glue.h"

#ifdef USE_NGPARSE

#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
////new for windows///
#ifdef _WIN32
#include <io.h>       /* _get_osfhandle */
#include <windows.h>  /* GetFinalPathNameByHandleA */
#endif
//// end new for windows///
#include "../misc/mktemp.h"
#include "ngparse.h"  /* the ngparse Rust ABI */

/* Worker cores for ngparse.  One core unless ngparse_cores asks for
 * more -- there is never a reason to set it to 1, that being the default.
 *
 * Single-threaded is the measured right answer, not an omission: expansion is
 * ~10% of deck-load time, the rest being INPpas1/2/3 and model setup here in
 * ngspice, which this does not touch.  The knob exists so that if that ever
 * changes, parallelism lands without a surface change; ngparse accepts a larger
 * value and warns that it is not honored yet. */
static int ngparse_glue_cores(void)
{
    const char *s = getenv("ngparse_cores");
    if (!s || !*s)
        return 1;
    long n = strtol(s, NULL, 10);
    if (n < 1) {
        fprintf(stderr,
                "ngparse: ngparse_cores=%s is not a positive integer; using 1\n",
                s);
        return 1;
    }
    return (int) n;
}

/* Compatibility dialect for ngparse, read from ngspice's `ngbehavior` variable
 * (set in spinit / .spiceinit).  ngparse has taken over expansion, so ngspice's
 * own pspice_compat pass -- which is what rewrites if()/VSWITCH/TABLE and injects
 * the PSpice funcs -- never runs on our already-flat deck.  We therefore tell
 * ngparse the mode so it can apply those conversions itself and a `ngbehavior=ps`
 * run matches the reference.  1 = PSpice; 0 = default/HSPICE. */
static int ngparse_glue_compat(void)
{
    char behaviour[128];
    if (cp_getvar("ngbehavior", CP_STRING, behaviour, sizeof(behaviour))) {
        /* leading "ps" selects PSpice, matching how ngspice keys newcompat.ps */
        if (behaviour[0] == 'p' && behaviour[1] == 's')
            return 1;
    }
    return 0;
}

/* ON by default in a build configured --enable-ngparse: enabling it at build
 * time is already the deliberate choice, so there is nothing to opt into again
 * on every run.  `--no-ngparse` turns it off for a run -- an escape hatch if
 * some deck ever trips ngparse up, without needing a rebuild. */
static bool ngparse_requested = TRUE;

void ngparse_glue_request(bool on)
{
    ngparse_requested = on;
}

/* Does `filename` start with *ng_script -- i.e. is it a command file rather than
 * a netlist?
 *
 * inp_spsource's `comfile` argument cannot answer this for us: it detects
 * *ng_script by probing the deck AFTER inp_readall() has read it, which is well
 * past the point where we must decide.  So run the same test directly on the
 * file, with the same "blank card" rule as that probe.
 *
 * Getting this wrong is not subtle: ngparse expands the .control script as if it
 * were a netlist, producing an empty circuit ("Circuit: *" / "incomplete or
 * empty netlist") before the script's own `source` line ever runs. */
static bool file_is_ng_script(const char *filename)
{
    char buf[BSIZE_SP + 1];
    bool is_script = FALSE;
    FILE *f = fopen(filename, "r");

    if (!f)
        return FALSE;  /* let the normal path report the open failure */
    while (fgets(buf, sizeof buf, f)) {
        if (buf[0] == '\0' || buf[0] == '\n' ||
            (buf[0] == '\r' && buf[1] == '\n'))
            continue;  /* leading blank card */
        is_script = ciprefix("*ng_script", buf) ? TRUE : FALSE;
        break;
    }
    fclose(f);
    return is_script;
}

bool ngparse_glue_enabled(bool comfile, bool intfile, const char *filename)
{
    if (!ngparse_requested)
        return FALSE;
    /* Command files (*ng_script) are .control scripts, not netlists, and an
     * internal/array deck has no file for ngparse to read. */
    if (comfile || intfile || !filename)
        return FALSE;
    if (file_is_ng_script(filename))
        return FALSE;
    return TRUE;
}

/* Resolve the real filesystem path of the file ngspice already opened: `fp` was
 * located via `sourcepath`/inputdir, whereas `filename` may be a bare relative
 * name that does NOT exist in the process cwd -- which is exactly the case when a
 * *ng_script control deck does `source foo.net` with `set sourcepath = (dir)`.
 * ngparse must read the file ngspice found, not re-open the relative name, so we
 * read the open descriptor's path from /proc. Falls back to a copy of `filename`
 * if that is unavailable (an internal deck with no fp, or a non-Linux host); the
 * caller then behaves exactly as before. Returned string is owned by the caller. */
char *ngparse_glue_realpath(FILE *fp, const char *filename)
{
    /* A NULL filename is the caller opting out (internal/array deck, or a
     * multi-file `source` concatenated into a temp): keep it NULL so the glue
     * skips, exactly as before. Only resolve a real, named file. */
    if (!filename)
        return NULL;
/////////new for windows//////////
#ifdef _WIN32
    /* Windows has no /proc: recover the path ngspice actually opened from the underlying OS handle via GetFinalPathNameByHandle -- the Win32 equivalent
     * of reading /proc/self/fd.  Without this, "source foo.net" located through "set sourcepath = (dir)" would hand ngparse the bare relative name, which
     * does not exist in the process cwd. */
    if (fp) {
        HANDLE h = (HANDLE) _get_osfhandle(fileno(fp));
        if (h != INVALID_HANDLE_VALUE) {
            char buf[PATH_MAX];
            DWORD n = GetFinalPathNameByHandleA(h, buf, sizeof buf - 1,
                                                FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
            if (n > 0 && n < sizeof buf) {
                char *p = buf;
                /* Strip the extended-length prefix:
                 * "\\?\UNC\srv\shr\..." -> "\\srv\shr\...",
                 * "\\?\C:\..."          -> "C:\...". */
                if (strncmp(p, "\\\\?\\UNC\\", 8) == 0) {
                    p += 6;
                    p[0] = '\\';
                    p[1] = '\\';
                } else if (strncmp(p, "\\\\?\\", 4) == 0) {
                    p += 4;
                }
                if (access(p, R_OK) == 0)
                    return copy(p);
            }
        }
    }
#else
/////////end new for windows//////////
    if (fp) {
        char proc[64];
        char buf[PATH_MAX];
        snprintf(proc, sizeof proc, "/proc/self/fd/%d", fileno(fp));
        ssize_t n = readlink(proc, buf, sizeof buf - 1);
        if (n > 0) {
            buf[n] = '\0';
            /* Trust it ONLY if it names a real, readable file. In batch mode
             * ngspice slurps the top-level deck into an UNLINKED temp, whose
             * descriptor reads back as "/tmp/#NNN (deleted)" -- not openable, and
             * handing it to ngparse would break the *ng_script check and the
             * expand. Fall back to `filename` (the pre-resolution behavior) then;
             * the real win is the `source foo.net` case, where fp IS the file
             * ngspice found via sourcepath and access() succeeds. */
            if (access(buf, R_OK) == 0)
                return copy(buf);
        }
    }
#endif
    return copy(filename ? filename : "");
}

char *ngparse_glue_expand(const char *filename)
{
    NgpDeck *deck;
    size_t i, n, drops;
    char *tmp_path;
    FILE *out;

    deck = ngparse_expand_file(filename, ngparse_glue_cores(), ngparse_glue_compat());
    if (!deck) {
        const char *err = ngparse_last_error();
        fprintf(stderr, "ngparse: %s\n", err ? err : "expansion failed");
        return NULL;
    }

    n = ngparse_deck_len(deck);
    if (n == 0) {
        fprintf(stderr, "ngparse: %s expanded to an empty deck\n", filename);
        ngparse_deck_free(deck);
        return NULL;
    }

    /* A dropped parameter is never harmless: the device or model silently falls
     * back to its DEFAULT, which converges to a wrong answer rather than
     * erroring.  Report every one -- do not let the user find out from a bad
     * waveform. */
    drops = ngparse_deck_drop_count(deck);
    if (drops > 0) {
        fprintf(stderr,
                "ngparse: WARNING: %zu parameter(s) could not be resolved and were dropped;\n"
                "  the affected device/model falls back to its DEFAULT value:\n",
                drops);
        for (i = 0; i < drops && i < 10; i++)
            fprintf(stderr, "    %s\n", ngparse_deck_drop(deck, i));
        if (drops > 10)
            fprintf(stderr, "    ... and %zu more\n", drops - 10);
    }

    /* smktemp() is ngspice's portable temp-name helper (see com_xgraph.c). */
    tmp_path = smktemp("ngp");
    out = fopen(tmp_path, "w");
    if (!out) {
        fprintf(stderr, "ngparse: cannot write the temporary deck %s: %s\n",
                tmp_path, strerror(errno));
        tfree(tmp_path);
        ngparse_deck_free(deck);
        return NULL;
    }

    /* Card 0 is the TITLE, and is written first: whatever reads this deck back
     * consumes line 1 as the title and starts the netlist at line 2.  Dropping
     * it would silently eat the first real card. */
    for (i = 0; i < n; i++) {
        const char *card = ngparse_deck_card(deck, i);
        if (fprintf(out, "%s\n", card ? card : "") < 0) {
            fprintf(stderr, "ngparse: writing the temporary deck failed: %s\n",
                    strerror(errno));
            fclose(out);
            remove(tmp_path);
            tfree(tmp_path);
            ngparse_deck_free(deck);
            return NULL;
        }
    }
    if (fclose(out) != 0) {
        fprintf(stderr, "ngparse: closing the temporary deck failed: %s\n",
                strerror(errno));
        remove(tmp_path);
        tfree(tmp_path);
        ngparse_deck_free(deck);
        return NULL;
    }

    if (ft_ngdebug)
        fprintf(stdout, "ngparse: %s -> %s (%zu cards, %zu drops)\n",
                filename, tmp_path, n, drops);

    ngparse_deck_free(deck);
    return tmp_path;
}

#else /* !USE_NGPARSE */

void ngparse_glue_request(bool on)
{
    /* --no-ngparse (on == FALSE) asks for the old parser, which is all this
     * build has, so say nothing.  Only an explicit --ngparse is worth a word. */
    if (on)
        fprintf(stderr,
                "ngspice: --ngparse: this build has no ngparse support "
                "(configure with --enable-ngparse); parsing normally.\n");
}

bool ngparse_glue_enabled(bool comfile, bool intfile, const char *filename)
{
    NG_IGNORE(comfile);
    NG_IGNORE(intfile);
    NG_IGNORE(filename);
    return FALSE;
}

char *ngparse_glue_expand(const char *filename)
{
    NG_IGNORE(filename);
    return NULL;
}

char *ngparse_glue_realpath(FILE *fp, const char *filename)
{
    NG_IGNORE(fp);
    NG_IGNORE(filename);
    return NULL;
}

#endif /* USE_NGPARSE */
