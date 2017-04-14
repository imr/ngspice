(compile "wget http://bsim.berkeley.edu/BSIM4/BSIM440.zip")

(compile "unzip -j BSIM440.zip 'BSIM440/src/*'")
(compile "git add b4*.c bsim4*.h inp2m.c inpdomod.c inpfindl.c noisean.c B4TERMS_OF_USE")

;;; whitespace cleanup
(compile "perl -i -pe 's/[ \\t\\r]*\\n/\\n/g;' *.c *.h")

;;; code transformation
(compile "perl -i -pe 's/BSIM4(?!\\.4\\.0)/BSIM4v4/g;' *.c *.h")
(compile "perl -i -pe 's/bsim4/bsim4v4/g;' *.c *.h")

(compile "perl -i -pe 's|#include \"spice.h\"|#include \"ngspice/ngspice.h\"|g;' *.c *.h")
(compile "perl -i -pe 's/#include \"(devdefs|suffix|const|ifsim|cktdefs|sperror|complex|noisedef|smpdefs|iferrmsg|noisedef|gendefs|trandefs).h\"/#include \"ngspice\\/$1.h\"/g;' *.c *.h")

(compile "perl -i -pe 's/^#include <stdio.h>[ \\t]*\\n//g;' *.c *.h")
(compile "perl -i -pe 's/FABS/fabs/g;' *.c *.h")

(compile "perl -i -pe 's/^#include <math.h>[ \\t]*\\n//g;' *.c *.h")
(compile "perl -i -pe 's/^#include \"util.h\"[ \\t]*\\n//g;' *.c *.h")

;;---

(compile "for i in init.c init.h ; do cp ../bsim4v5/bsim4v5$i bsim4v4$i ; git add bsim4v4$i ; done")
(compile "for i in soachk.c ; do cp ../bsim4v5/b4v5$i b4v4$i ; git add b4v4$i ; done")
(compile "for i in Makefile.am ; do cp ../bsim4v5/$i $i ; git add $i ; done")

(compile "perl -i -pe 's/(bsim4)v5/${1}v4/ig;' *.c *.h")
(compile "perl -i -pe 's/(b(?:sim)?4)v5/${1}v4/ig;' Makefile.am")
