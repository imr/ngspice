/* Enhancement-200: the `pre_snp` control command.
 *
 * `pre_snp <file.sNp> [module]` converts a Touchstone S-parameter file to a
 * Verilog-A n-port model (via snp2va_convert, the C port of snp2va.py) and then
 * invokes openvaf-r to compile it, producing <file>.osdi next to the source --
 * so `pre_osdi <file>.osdi` then loads it, symmetric with the existing flow.
 * The `.va`/`.osdi` are written beside the `.sNp` with the same base name.
 *
 * openvaf-r is located via, in order: the `openvaf` ngspice variable, the
 * OPENVAF environment variable, $SPICE_LIB_DIR/openvaf-r (the prebuilt bin
 * bundle keeps it there), then PATH.
 */
#include "ngspice/ngspice.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/cpextern.h"

#include "snp2va.h"

#include <sys/stat.h>

static int file_exists(const char *p)
{
    struct stat st;
    return stat(p, &st) == 0;
}

/* Locate the openvaf-r compiler. Returns a malloc'd string (copy()). */
static char *find_openvaf(void)
{
    char var[1024];
    char *e;
    if (cp_getvar("openvaf", CP_STRING, var, sizeof var) && var[0])
        return copy(var);
    e = getenv("OPENVAF");
    if (e && e[0])
        return copy(e);
    e = getenv("SPICE_LIB_DIR");
    if (e && e[0]) {
        char buf[1200];
        (void) snprintf(buf, sizeof buf, "%s/openvaf-r", e);
        if (file_exists(buf))
            return copy(buf);
    }
    return copy("openvaf-r");                       /* rely on PATH */
}

/* base = basename(snp) with the extension dropped; sanitized to a Verilog id. */
static void derive_module(const char *snp, char *mod, size_t modlen)
{
    const char *base = strrchr(snp, '/');
#ifdef _WIN32
    const char *bs = strrchr(snp, '\\');
    if (bs && (!base || bs > base)) base = bs;
#endif
    base = base ? base + 1 : snp;
    size_t i = 0;
    for (; base[i] && base[i] != '.' && i + 6 < modlen; i++) {
        char c = base[i];
        mod[i] = (isalnum((unsigned char) c) || c == '_') ? c : '_';
    }
    mod[i] = '\0';
    if (i == 0 || isdigit((unsigned char) mod[0])) {   /* must start with a letter */
        memmove(mod + 1, mod, i + 1);
        mod[0] = 'm';
    }
}

/* Replace the extension of `src` with `ext` into `dst`. */
static void with_ext(const char *src, const char *ext, char *dst, size_t dstlen)
{
    (void) snprintf(dst, dstlen, "%s", src);
    char *dot = strrchr(dst, '.');
    char *slash = strrchr(dst, '/');
    if (dot && (!slash || dot > slash))
        *dot = '\0';
    size_t n = strlen(dst);
    (void) snprintf(dst + n, dstlen - n, "%s", ext);
}

void com_pre_snp(wordlist *wl)
{
    char module[256], va[1200], osdi[1200], nport[1200], msg[256], *snp, *ovf;
    char *cmd;
    size_t cmdlen;
    int rc, native = 0;

    /* optional leading backend flag: -osdi (default) or -native */
    while (wl && wl->wl_word && wl->wl_word[0] == '-') {
        if (eq(wl->wl_word, "-native"))    native = 1;
        else if (eq(wl->wl_word, "-osdi")) native = 0;
        else { fprintf(cp_err, "pre_snp: unknown option '%s'\n", wl->wl_word); return; }
        wl = wl->wl_next;
    }

    if (!wl || !wl->wl_word) {
        fprintf(cp_err, "usage: pre_snp [-osdi|-native] <file.sNp> [module]\n"
                        "  -osdi   (default) Touchstone -> Verilog-A -> openvaf-r -> <file>.osdi,\n"
                        "                    then load with `pre_osdi <file>.osdi`.\n"
                        "  -native           Touchstone -> <file>.nport for the built-in n-port\n"
                        "                    device (no compiler); use it in the deck with\n"
                        "                    `N1 <ports..> <ref> m` / `.model m nport(file=\"<file>.nport\")`.\n");
        return;
    }
    snp = wl->wl_word;
    if (wl->wl_next && wl->wl_next->wl_word) {
        (void) snprintf(module, sizeof module, "%s", wl->wl_next->wl_word);
    } else {
        derive_module(snp, module, sizeof module);
    }

    /* -native: emit the compact .nport fit file; no Verilog-A / openvaf-r step. */
    if (native) {
        with_ext(snp, ".nport", nport, sizeof nport);
        if (snp2nport_convert(snp, nport, msg, sizeof msg)) {
            fprintf(cp_err, "pre_snp: %s\n", msg);
            return;
        }
        fprintf(cp_out, "pre_snp: %s -> %s  (%s)\n", snp, nport, msg);
        fprintf(cp_out, "pre_snp: use it with  `N1 <ports..> <ref> m`  and\n"
                        "                       `.model m nport(file=\"%s\")`\n", nport);
        return;
    }

    with_ext(snp, ".va", va, sizeof va);
    with_ext(snp, ".osdi", osdi, sizeof osdi);

    /* 1. Touchstone -> Verilog-A (the C converter) */
    if (snp2va_convert(snp, va, module, msg, sizeof msg)) {
        fprintf(cp_err, "pre_snp: %s\n", msg);
        return;
    }
    fprintf(cp_out, "pre_snp: %s -> %s  (%s, module '%s')\n", snp, va, msg, module);

    /* 2. compile with openvaf-r -> .osdi */
    ovf = find_openvaf();
    cmdlen = strlen(ovf) + strlen(va) + strlen(osdi) + 32;
    cmd = TMALLOC(char, cmdlen);
    (void) snprintf(cmd, cmdlen, "\"%s\" \"%s\" -o \"%s\"", ovf, va, osdi);
    rc = system(cmd);
    tfree(cmd);
    if (rc != 0) {
        fprintf(cp_err, "pre_snp: openvaf-r failed (exit %d) compiling %s.\n"
                        "  Set the compiler with `set openvaf=/path/to/openvaf-r`, the OPENVAF\n"
                        "  environment variable, or put openvaf-r in $SPICE_LIB_DIR or PATH.\n",
                rc, va);
        tfree(ovf);
        return;
    }
    tfree(ovf);
    fprintf(cp_out, "pre_snp: compiled -> %s   (load it with `pre_osdi %s`)\n", osdi, osdi);
}
