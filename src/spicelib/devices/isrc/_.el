(diff "isrcacct.c" "../vsrc/vsrcacct.c" "-U2")
(diff "isrcacld.c" "../vsrc/vsrcacld.c" "-U2")
(diff "isrcask.c"  "../vsrc/vsrcask.c"  "-U2")
(diff "isrc.c"     "../vsrc/vsrc.c"     "-U2")
(diff "isrcdefs.h" "../vsrc/vsrcdefs.h" "-U2")
(diff "isrcdel.c"  "../vsrc/vsrcdel.c"  "-U2")
(diff "isrcdest.c" "../vsrc/vsrcdest.c" "-U2")
(diff "isrcext.h"  "../vsrc/vsrcext.h"  "-U2")
(diff "isrcinit.c" "../vsrc/vsrcinit.c" "-U2")
(diff "isrcinit.h" "../vsrc/vsrcinit.h" "-U2")
(diff "isrcitf.h"  "../vsrc/vsrcitf.h"  "-U2")
(diff "isrcload.c" "../vsrc/vsrcload.c" "-U2")
(diff "isrcmdel.c" "../vsrc/vsrcmdel.c" "-U2")
(diff "isrcpar.c"  "../vsrc/vsrcpar.c"  "-U2")
(diff "isrctemp.c" "../vsrc/vsrctemp.c" "-U2")


(defun here-c1(file)
  (compile (concat "cd ../../../../tests/regression/pwl-src && ../../../../w32/src/ngspice -b " file) t))

(defun here-c2(file)
  (compile (concat "cd ../../../../tests/regression/pwl-src && valgrind --track-origins=yes --leak-check=full --show-reachable=yes ../../../../w32/src/ngspice -b " file) t))


(here-c1 "isrc-pwl-1.cir")
(here-c1 "vsrc-pwl-1.cir")
(here-c1 "vsrc-pwl-2.cir")

(here-c2 "isrc-pwl-1.cir")
(here-c2 "vsrc-pwl-1.cir")
(here-c2 "vsrc-pwl-2.cir")
