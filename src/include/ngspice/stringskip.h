#ifndef ngspice_STRINGSKIP_H
#define ngspice_STRINGSKIP_H

static inline char *TEMPORARY_SKIP_NON_WS_X0(char *s)  { while (*s && !isspace_c(*s)) s++; return s; }
static inline char *TEMPORARY_SKIP_WS_X1(char *s)      { while (       isspace_c(*s)) s++; return s; }

#endif
