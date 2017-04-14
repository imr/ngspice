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
