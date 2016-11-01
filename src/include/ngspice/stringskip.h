#ifndef ngspice_STRINGSKIP_H
#define ngspice_STRINGSKIP_H

static inline char *skip_non_ws(const char *s)  { while (*s && !isspace_c(*s)) s++; return (char *) s; }
static inline char *skip_ws(const char *s)      { while (       isspace_c(*s)) s++; return (char *) s; }

static inline char *depreciated_skip_back_non_ws(char *s) { while (s[-1] && !isspace_c(s[-1])) s--; return s; }
static inline char *depreciated_skip_back_ws(char *s)     { while (isspace_c(s[-1]))           s--; return s; }

static inline char *skip_back_non_ws(const char *s, const char *start) { while (s > start && !isspace_c(s[-1])) s--; return (char *) s; }
static inline char *skip_back_ws(const char *s, const char *start)     { while (s > start && isspace_c(s[-1])) s--; return (char *) s; }

#endif
