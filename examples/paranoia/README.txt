
To run the paranoia test suite in parallel on Linux with valgrind:

1. Download the paranoia tests (paranoia.7z) from the ngspice Quality web page.

2. p7zip -d paranoia.7z
Rename the unzipped directory to a name without spaces which would
otherwise confuse valgrind.

3. cd into the renamed unzipped directory.

4. copy runtests.sh and textract.py from the examples/paranoia directory in
your git repository to the current directory.

5. If your computer has several cores, you can modify the -j4 in the line
  time parallel -j4 bash ::: $2/*
in runtests.sh and increase the number of parallel jobs.

6. ./runtests.sh <paranoia_shell_script> <test_area_directory>
For example:
  ./runtests.sh paranoia_test.sh testdir
Note that the test area directory must not exist before you invoke runtests.sh.

Now relax and drink a cup of coffee. If you don't want to run the tests in
parallel, it will take several cups.

