/*
 * OSDI deferred-evaluation side table — implementation.
 * See osdi_defer.h for the design rationale.
 */

#include "ngspice/osdi_defer.h"
#include <string.h>
#include <stdlib.h>

/* The numparam directory's public header for the eval-with-scope
 * helper and the dico_t accessor.  Path is relative to src/include's
 * search root + the per-component layout. */
#include "../frontend/numparam/numparam.h"

/* Single global list head.  ngspice runs a single simulator process per
 * netlist, so a global is sufficient and matches existing patterns
 * (e.g. modtabhash). */
static OsdiDeferEntry *g_defer_head = NULL;

static char *xstrdup(const char *s) {
    size_t n = strlen(s) + 1;
    char *r = (char *)malloc(n);
    if (!r) return NULL;
    memcpy(r, s, n);
    return r;
}

/* Forward decl: scope-snapshot helper, defined further down (needs the
 * identifier-scanning helpers from the line-preprocessor section). */
static void capture_scope_snapshot(OsdiDeferEntry *e);

int osdi_defer_register(const char *model_name,
                        const char *param_name,
                        const char *expr_text)
{
    if (!model_name || !param_name || !expr_text) return 1;

    OsdiDeferEntry *e = (OsdiDeferEntry *)malloc(sizeof *e);
    if (!e) return 1;

    e->model_name = xstrdup(model_name);
    e->param_name = xstrdup(param_name);
    e->expr_text  = xstrdup(expr_text);
    e->n_snap = 0;
    e->snap_names = NULL;
    e->snap_values = NULL;
    if (!e->model_name || !e->param_name || !e->expr_text) {
        free(e->model_name);
        free(e->param_name);
        free(e->expr_text);
        free(e);
        return 1;
    }
    /* Capture the subckt-instance scope NOW.  By register time the
     * surrounding subckt has been expanded per-instance, so the dico
     * resolves identifiers like `xl_nfet` to the value bound by THIS
     * subckt instance's params list. */
    capture_scope_snapshot(e);

    e->next = g_defer_head;
    g_defer_head = e;
    return 0;
}

/* Match a runtime model name (possibly subckt-prefixed and bin-suffixed)
 * against a deferred-eval entry's registered name.  Registration captures
 * the ORIGINAL model name as it appeared on the .model card (e.g.
 * `pfet` or `egnfet_s2`).  After subckt expansion + W/L binning, the
 * runtime sees names like `x2.xmp1:pfet.0` for the SAME logical model.
 * Strip those decorations before comparison.
 *
 * Algorithm:
 *   1. Drop any trailing `.[0-9]+` (bin suffix).
 *   2. Drop any leading `<subckt-path>:` (subckt name prefix), defined
 *      as everything up to and including the LAST `:` in the name.
 *   3. Case-insensitive strcmp against the registered name. */
static bool name_matches(const char *registered, const char *runtime) {
    if (!registered || !runtime) return false;
    /* Strip everything up to and including the LAST `:` from the
     * runtime name.  Registration captures the model name as it
     * appeared on the .model card (e.g. `pfet.0`).  Subckt expansion
     * prefixes the model name with the subckt path: `x2.xmp1:pfet.0`.
     * The bin suffix `.N` is part of BOTH names (the .model card had
     * `.0` already), so we don't touch it. */
    const char *base = runtime;
    const char *last_colon = strrchr(runtime, ':');
    if (last_colon)
        base = last_colon + 1;
    return strcasecmp(registered, base) == 0;
}

bool osdi_defer_has(const char *model_name) {
    if (!model_name) return false;
    for (OsdiDeferEntry *e = g_defer_head; e; e = e->next)
        if (name_matches(e->model_name, model_name))
            return true;
    return false;
}

int osdi_defer_for_model(const char *model_name,
                         void (*cb)(const char *, const char *, void *),
                         void *cb_arg)
{
    if (!model_name || !cb) return 0;
    int n = 0;
    for (OsdiDeferEntry *e = g_defer_head; e; e = e->next) {
        if (name_matches(e->model_name, model_name)) {
            cb(e->param_name, e->expr_text, cb_arg);
            n++;
        }
    }
    return n;
}

/* Bin-range side table: separate list from the deferred-expression
 * entries because populated by a different caller (INPgetModBin in
 * the parser) at a different time (after subckt expansion + binning,
 * but before OSDIsetup).  Keyed by the runtime model name (e.g.
 * `pfet.0`); OSDIsetup strips the subckt prefix before looking up. */
typedef struct OsdiBinRange {
    char *model_name;
    double lmin, lmax;
    struct OsdiBinRange *next;
} OsdiBinRange;

static OsdiBinRange *g_bin_head = NULL;

void osdi_defer_record_bin_range(const char *model_name,
                                 double lmin, double lmax) {
    if (!model_name) return;
    /* Replace if already present (idempotent for repeated calls). */
    for (OsdiBinRange *b = g_bin_head; b; b = b->next) {
        if (strcasecmp(b->model_name, model_name) == 0) {
            b->lmin = lmin;
            b->lmax = lmax;
            return;
        }
    }
    OsdiBinRange *b = (OsdiBinRange *)malloc(sizeof *b);
    if (!b) return;
    b->model_name = xstrdup(model_name);
    b->lmin = lmin;
    b->lmax = lmax;
    b->next = g_bin_head;
    g_bin_head = b;
}

bool osdi_defer_get_bin_range(const char *model_name,
                              double *lmin, double *lmax) {
    if (!model_name) return false;
    /* Same subckt-prefix stripping as name_matches — applied to BOTH
     * sides because INPgetModBin records `INPmodName` which carries
     * the subckt path prefix (e.g. `x3.xmn1:nfet.0`), and the runtime
     * lookup also passes a prefixed name.  Symmetric stripping ensures
     * lookups find the bin range regardless of which subckt-instance
     * recorded it. */
    const char *q_base = model_name;
    const char *q_colon = strrchr(model_name, ':');
    if (q_colon) q_base = q_colon + 1;
    for (OsdiBinRange *b = g_bin_head; b; b = b->next) {
        const char *s_base = b->model_name;
        const char *s_colon = strrchr(b->model_name, ':');
        if (s_colon) s_base = s_colon + 1;
        if (strcasecmp(s_base, q_base) == 0) {
            if (lmin) *lmin = b->lmin;
            if (lmax) *lmax = b->lmax;
            return true;
        }
    }
    return false;
}

void osdi_defer_clear(void) {
    OsdiDeferEntry *e = g_defer_head;
    while (e) {
        OsdiDeferEntry *next = e->next;
        free(e->model_name);
        free(e->param_name);
        free(e->expr_text);
        for (int i = 0; i < e->n_snap; i++) free(e->snap_names[i]);
        free(e->snap_names);
        free(e->snap_values);
        free(e);
        e = next;
    }
    g_defer_head = NULL;
}


/* ============================================================
 *  Line preprocessor: detect + rewrite deferred-eval expressions
 *  on `.model` lines targeting OSDI device types.
 * ============================================================ */

#include <ctype.h>

/* Skip leading whitespace. */
static const char *skip_ws(const char *p) {
    while (*p && isspace((unsigned char)*p)) p++;
    return p;
}

/* Identifier characters per HSPICE/Verilog-A rules. */
static int is_ident_start(int c) {
    return isalpha(c) || c == '_';
}
static int is_ident_cont(int c) {
    return isalnum(c) || c == '_';
}

/* Returns true if `expr` (a brace-or-quote body, no enclosing chars)
 * mentions any of the per-instance reserved identifiers as a bare
 * word — case-insensitively, not as a substring of a larger
 * identifier and not after a `.` (subckt-path qualifier).
 *
 * Reserved set:
 *   l, w, nf, m, xnf — standard HSPICE instance params
 *   l_calc           — Samsung/foundry "computed length" (= l + p_la
 *                       in the foundry subckt; defaults to l alone) */
static bool expr_refs_instance_geom(const char *expr) {
    static const char *const RES[] = {
        "l", "w", "nf", "m", "xnf", "l_calc", NULL
    };
    for (const char *p = expr; *p; ) {
        if (is_ident_start((unsigned char)*p)) {
            const char *start = p;
            while (is_ident_cont((unsigned char)*p)) p++;
            size_t n = (size_t)(p - start);
            /* skip if preceded by '.' (subckt-path) */
            if (start > expr && start[-1] == '.') continue;
            for (int i = 0; RES[i]; i++) {
                size_t rn = strlen(RES[i]);
                if (n == rn && strncasecmp(start, RES[i], rn) == 0)
                    return true;
            }
        } else {
            p++;
        }
    }
    return false;
}

/* Look for `level=<digits>` in the first part of a `.model` line and
 * return the integer.  Returns -1 if absent or unparseable. */
static int parse_level(const char *line) {
    /* find "level" then '=' then digits */
    for (const char *p = line; *p; p++) {
        if ((p == line || !is_ident_cont((unsigned char)p[-1])) &&
            (p[0] == 'l' || p[0] == 'L') &&
            (p[1] == 'e' || p[1] == 'E') &&
            (p[2] == 'v' || p[2] == 'V') &&
            (p[3] == 'e' || p[3] == 'E') &&
            (p[4] == 'l' || p[4] == 'L') &&
            !is_ident_cont((unsigned char)p[5])) {
            p += 5;
            p = skip_ws(p);
            if (*p != '=') continue;
            p++;
            p = skip_ws(p);
            /* possible single-quote wrap */
            if (*p == '\'' || *p == '{') p++;
            if (isdigit((unsigned char)*p))
                return atoi(p);
        }
    }
    return -1;
}

/* Parse the model name from a `.model` line: the second whitespace-
 * separated token.  Returns a freshly-allocated string the caller must
 * free, or NULL on parse failure. */
static char *parse_model_name(const char *line) {
    const char *p = skip_ws(line);
    if (strncasecmp(p, ".model", 6) != 0) return NULL;
    p += 6;
    p = skip_ws(p);
    const char *start = p;
    while (*p && !isspace((unsigned char)*p) && *p != '(' && *p != '=')
        p++;
    if (p == start) return NULL;
    size_t n = (size_t)(p - start);
    char *r = (char *)malloc(n + 1);
    if (!r) return NULL;
    memcpy(r, start, n);
    r[n] = '\0';
    return r;
}

/* Given that we've just seen `param_name` followed by `=` on a model
 * line, scan back from `eq_pos` to find the parameter name's start.
 * Returns the index in `line` of the first char of the name, or
 * (size_t)-1 if not found. */
static size_t find_param_name_start(const char *line, size_t eq_pos) {
    if (eq_pos == 0) return (size_t)-1;
    size_t i = eq_pos;
    /* skip whitespace before '=' */
    while (i > 0 && isspace((unsigned char)line[i - 1])) i--;
    size_t end = i;
    /* walk back over identifier chars */
    while (i > 0 && is_ident_cont((unsigned char)line[i - 1])) i--;
    if (i == end) return (size_t)-1;
    return i;
}

int osdi_defer_preprocess_line(char **line_p) {
    if (!line_p || !*line_p) return 0;
    const char *line = *line_p;

    /* Quick reject: not a .model line. */
    const char *p = skip_ws(line);
    if (strncasecmp(p, ".model", 6) != 0) return 0;

    /* Only OSDI-targeted levels (72 BSIM-CMG, 77 BSIM-BULK) participate.
     * Adjust here if new OSDI MOS levels are added to inpdomod.c. */
    int level = parse_level(line);
    if (level != 72 && level != 77) return 0;

    char *model_name = parse_model_name(line);
    if (!model_name) return 0;

    /* Walk the line copying chars; when we hit `<param>={...}` or
     * `<param>='...'` whose body references instance geom, register and
     * rewrite to `<param>=0`.  Build the output in a new buffer. */
    size_t cap = strlen(line) + 64;
    char *out = (char *)malloc(cap);
    if (!out) { free(model_name); return 1; }
    size_t olen = 0;

    #define EMIT_CH(ch) do { \
        if (olen + 1 >= cap) { \
            cap *= 2; \
            char *_t = (char *)realloc(out, cap); \
            if (!_t) { free(out); free(model_name); return 1; } \
            out = _t; \
        } \
        out[olen++] = (ch); \
    } while (0)

    const char *cur = line;
    while (*cur) {
        if (*cur != '{' && *cur != '\'') {
            EMIT_CH(*cur);
            cur++;
            continue;
        }
        char open = *cur;
        char close = (open == '{') ? '}' : '\'';

        /* Locate matching close (brace-balanced for '{', literal for '\''). */
        const char *body_start = cur + 1;
        const char *q = body_start;
        int nest = 1;
        while (*q) {
            if (open == '{') {
                if (*q == '{') nest++;
                else if (*q == '}') { nest--; if (nest == 0) break; }
            } else {
                if (*q == '\'') { nest = 0; break; }
            }
            q++;
        }
        if (*q == '\0') {
            /* unterminated — give up, emit verbatim and continue */
            EMIT_CH(*cur);
            cur++;
            continue;
        }

        /* `body` = [body_start, q); doesn't include the closing delim. */
        size_t body_len = (size_t)(q - body_start);
        char *body = (char *)malloc(body_len + 1);
        if (!body) { free(out); free(model_name); return 1; }
        memcpy(body, body_start, body_len);
        body[body_len] = '\0';

        bool defer = expr_refs_instance_geom(body);
        if (defer) {
            /* Find the param name immediately preceding the `=` that
             * precedes this expression.  olen is the current write
             * position in out; the just-emitted run ends with the `=`. */
            size_t eq_pos = olen;
            while (eq_pos > 0 && out[eq_pos - 1] != '=') eq_pos--;
            if (eq_pos == 0) {
                /* no `=` found — not an `<param>=<expr>` shape; bail */
                defer = false;
            } else {
                size_t name_start = find_param_name_start(out, eq_pos - 1);
                if (name_start == (size_t)-1) {
                    defer = false;
                } else {
                    /* Slice param name out of `out` for registration. */
                    size_t name_len = (eq_pos - 1) - name_start;
                    /* trim trailing ws in name */
                    while (name_len > 0 && isspace((unsigned char)out[name_start + name_len - 1]))
                        name_len--;
                    if (name_len == 0) {
                        defer = false;
                    } else {
                        char *pname = (char *)malloc(name_len + 1);
                        if (!pname) { free(body); free(out); free(model_name); return 1; }
                        memcpy(pname, out + name_start, name_len);
                        pname[name_len] = '\0';
                        osdi_defer_register(model_name, pname, body);
                        free(pname);
                    }
                }
            }
        }

        if (defer) {
            /* Replace `{ body }` / `' body '` with `{0}` / `'0'` in the
             * output.  Keep the delimiters so numparam still recognises
             * the slot as a brace expression and substitutes the
             * computed value (0) into the corresponding MARKER
             * placeholder in the rewritten card->line.  The model
             * parameter ends up = 0 as a placeholder; per-instance
             * deferred-eval in OSDItemp installs the real value. */
            EMIT_CH(open);
            EMIT_CH('0');
            EMIT_CH(close);
        } else {
            /* Not deferred; emit the original expression verbatim so
             * numparam can run it. */
            EMIT_CH(open);
            for (size_t k = 0; k < body_len; k++) EMIT_CH(body[k]);
            EMIT_CH(close);
        }
        free(body);
        cur = q + 1;
    }
    EMIT_CH('\0');
    /* roll back the implicit terminator so olen excludes it */
    olen--;

    /* Replace the input buffer. */
    free(*line_p);
    *line_p = out;
    free(model_name);
    return 0;

    #undef EMIT_CH
}


/* ============================================================
 *  Scope-snapshot capture (called from osdi_defer_register)
 * ============================================================ */

/* Names the eval path always pushes explicitly; never snapshot them
 * from the dico (would either duplicate or shadow the per-instance
 * value with a parse-time value). */
static bool is_eval_reserved(const char *name, size_t n) {
    static const char *const RES[] = {
        "l", "w", "nf", "m", "xnf", "l_calc", NULL
    };
    for (int i = 0; RES[i]; i++) {
        size_t rn = strlen(RES[i]);
        if (n == rn && strncasecmp(name, RES[i], rn) == 0) return true;
    }
    return false;
}

/* Well-known math/keyword identifiers — saves a dico lookup for the
 * hottest non-symbols.  Lookup failure is harmless, so this list need
 * not be exhaustive. */
static bool is_known_keyword(const char *name, size_t n) {
    static const char *const KW[] = {
        "exp", "log", "ln",  "log10", "sqrt", "sin", "cos", "tan",
        "asin","acos","atan","atan2", "sinh", "cosh","tanh",
        "abs", "min", "max", "pow",   "int",  "floor","ceil",
        "if",  "else","then","sgn",   "fabs", "hypot",
        NULL
    };
    for (int i = 0; KW[i]; i++) {
        size_t kn = strlen(KW[i]);
        if (n == kn && strncasecmp(name, KW[i], kn) == 0) return true;
    }
    return false;
}

static void capture_scope_snapshot(OsdiDeferEntry *e) {
    if (!e || !e->expr_text) return;

    dico_t *dico = nupa_get_dico();
    if (!dico) return;

    int cap = 0;
    /* Walk identifiers in expr_text, applying the same lexical rules
     * as expr_refs_instance_geom: idents preceded by `.` are
     * subckt-path qualifiers, not free variables.  For each unique
     * non-reserved ident, look up the current dico (which walks the
     * full scope stack) and snapshot its scalar value if it's
     * NUPA_REAL.  Anything that doesn't resolve is left for the
     * eval-time path to flag — its absence is either a true missing
     * param OR a math keyword we didn't filter. */
    for (const char *p = e->expr_text; *p; ) {
        if (!is_ident_start((unsigned char)*p)) { p++; continue; }

        const char *start = p;
        while (is_ident_cont((unsigned char)*p)) p++;
        size_t n = (size_t)(p - start);

        /* Skip subckt-path-qualified identifiers (`foo.bar` — the
         * `bar` half is qualified). */
        if (start > e->expr_text && start[-1] == '.') continue;

        if (is_eval_reserved(start, n)) continue;
        if (is_known_keyword(start, n)) continue;
        if (n >= 128) continue;   /* implausibly long, skip */

        /* Dedupe against snapshot built so far. */
        bool dup = false;
        for (int i = 0; i < e->n_snap; i++) {
            if (strncasecmp(e->snap_names[i], start, n) == 0 &&
                e->snap_names[i][n] == '\0') { dup = true; break; }
        }
        if (dup) continue;

        char buf[128];
        memcpy(buf, start, n);
        buf[n] = '\0';

        entry_t *de = entrynb(dico, buf);
        if (!de || de->tp != NUPA_REAL) continue;

        /* Grow arrays as needed. */
        if (e->n_snap == cap) {
            int ncap = cap ? cap * 2 : 8;
            char  **nn = (char **) realloc(e->snap_names,  ncap * sizeof *nn);
            double *nv = (double *)realloc(e->snap_values, ncap * sizeof *nv);
            if (!nn || !nv) {
                /* allocation failed — keep what we have, abandon the rest */
                if (nn) e->snap_names  = nn;
                if (nv) e->snap_values = nv;
                return;
            }
            e->snap_names  = nn;
            e->snap_values = nv;
            cap = ncap;
        }
        e->snap_names[e->n_snap]  = xstrdup(buf);
        e->snap_values[e->n_snap] = de->vl;
        if (!e->snap_names[e->n_snap]) return;
        e->n_snap++;
    }
}


/* ============================================================
 *  Per-instance evaluation: thin wrapper over nupa_eval_with_scope
 * ============================================================ */

int osdi_defer_eval(const char *expr_text,
                    double l, double w, double nf, double m,
                    double *out)
{
    if (!expr_text || !out) return 1;
    dico_t *dico = nupa_get_dico();
    if (!dico) return 1;

    /* Per-instance symbols in scope:
     *   l, w, nf, m — standard HSPICE instance params
     *   xnf         — alias of nf (HSPICE convention)
     *   l_calc      — foundry "effective" length, defaults to l (the
     *                 foundry subckt would compute `l + p_la` but
     *                 p_la is 0 unless set per-instance). */
    static const char *base_names[] = { "l", "w", "nf", "m", "xnf", "l_calc" };
    double base_values[6] = { l, w, nf, m, nf, l };
    const int n_base = 6;

    /* Locate the entry that owns this expr_text — we passed the
     * pointer to the caller via osdi_defer_for_model's cb, so
     * pointer-equality is the canonical lookup.  Falls back to no-snap
     * if not found (defensive — shouldn't happen in practice). */
    const OsdiDeferEntry *entry = NULL;
    for (OsdiDeferEntry *it = g_defer_head; it; it = it->next) {
        if (it->expr_text == expr_text) { entry = it; break; }
    }

    if (!entry || entry->n_snap == 0) {
        return nupa_eval_with_scope(dico, expr_text,
                                    base_names, base_values, n_base, out);
    }

    /* Merge snap + base.  Insertion order matters: nupa_eval_with_scope
     * inserts into one hash with op='N', so the LAST insert for a
     * given name wins.  Put snap FIRST (lower priority), base LAST
     * (higher priority — instance geom overrides any snap that
     * happened to capture the same name from a parent scope). */
    int n_total = entry->n_snap + n_base;
    const char **names  = (const char **)malloc(n_total * sizeof *names);
    double      *values = (double *)     malloc(n_total * sizeof *values);
    if (!names || !values) { free(names); free(values); return 1; }

    for (int i = 0; i < entry->n_snap; i++) {
        names[i]  = entry->snap_names[i];
        values[i] = entry->snap_values[i];
    }
    for (int i = 0; i < n_base; i++) {
        names[entry->n_snap + i]  = base_names[i];
        values[entry->n_snap + i] = base_values[i];
    }

    int rc = nupa_eval_with_scope(dico, expr_text, names, values, n_total, out);
    free(names);
    free(values);
    return rc;
}
