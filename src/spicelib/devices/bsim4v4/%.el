(compile "wget http://bsim.berkeley.edu/BSIM4/BSIM440.zip")

(compile "unzip -j BSIM440.zip 'BSIM440/src/*'")
(compile "git add b4*.c bsim4*.h inp2m.c inpdomod.c inpfindl.c noisean.c B4TERMS_OF_USE")

;;; whitespace cleanup
(compile "perl -i -pe 's/[ \\t\\r]*\\n/\\n/g;' *.c *.h")
