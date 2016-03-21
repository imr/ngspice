#ifndef ngspice_STRINGSKIP_H
#define ngspice_STRINGSKIP_H

static inline char *skip_non_ws(char *s)  { while (*s && !isspace_c(*s)) s++; return s; }
static inline char *skip_ws(char *s)      { while (       isspace_c(*s)) s++; return s; }

#endif
