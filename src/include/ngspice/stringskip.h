#ifndef ngspice_STRINGSKIP_H
#define ngspice_STRINGSKIP_H

#define TEMPORARY_SKIP_NON_WS_X0(s)  do { while (*(s) && !isspace_c(*(s))) (s)++; } while(0)
#define TEMPORARY_SKIP_WS_X1(s)      do { while (         isspace_c(*(s))) (s)++; } while(0)

#endif
