#this file defines some common routines used by the OSDI test cases
import subprocess
import os
import shutil
import glob
from pathlib import Path

# specify location of Ngspice executable to be tested
directory_testing = os.path.dirname(__file__)
ngspice_path = os.path.join(directory_testing, "../debug/src/ngspice")
ngspice_path = os.path.abspath(ngspice_path)

def create_shared_objects(directory):
    c_files = []
    for c_file in glob.glob(directory + "/*.c"):
        basename = Path(c_file).stem
        c_files.append(basename)

    for c_file in c_files:
        subprocess.run(
            [
                "gcc",
                "-c",
                "-Wall",
                "-I",
                "../../src/osdi/",
                "-fpic",
                c_file + ".c",
                "-ggdb",
            ],
            cwd=directory,
        )
        subprocess.run(
            ["gcc", "-shared", "-o", c_file + ".osdi", c_file + ".o", "-ggdb"],
            cwd=directory,
        )
        subprocess.run(
            ["mv", c_file + ".osdi", "test_osdi/" + c_file + ".osdi"], cwd=directory
        )
        subprocess.run(["rm", c_file + ".o"], cwd=directory)

def prepare_dirs(directory):
    # directories for test cases
    dir_osdi = os.path.join(directory, "test_osdi")
    dir_built_in = os.path.join(directory, "test_built_in")

    for directory_i in [dir_osdi, dir_built_in]:
        # remove old results
        shutil.rmtree(directory_i, ignore_errors=True)
        # make new directories
        os.makedirs(directory_i, exist_ok=True)


    return dir_osdi, dir_built_in

def prepare_netlists(directory):
    path_netlist = os.path.join(directory, "netlist.sp")

    # directories for test cases
    dir_osdi = os.path.join(directory, "test_osdi")
    dir_built_in = os.path.join(directory, "test_built_in")

    # open netlist and activate Ngspice devices
    with open(path_netlist) as netlist_handle:
        netlist_raw = netlist_handle.read()

    netlist_osdi = netlist_raw.replace("*OSDI_ACTIVATE*", "")
    netlist_built_in = netlist_raw.replace("*BUILT_IN_ACTIVATE*", "")

    # write netlists
    with open(os.path.join(dir_osdi, "netlist.sp"), "w") as netlist_handle:
        netlist_handle.write(netlist_osdi)

    with open(os.path.join(dir_built_in, "netlist.sp"), "w") as netlist_handle:
        netlist_handle.write(netlist_built_in)
        
def run_simulations(dirs):
    for dir_i in dirs:
        subprocess.run(
            [
                ngspice_path,
                "netlist.sp",
                "-b",
            ],
            cwd=dir_i,
        )
    
def prepare_test(directory):
    dir_osdi, dir_built_in = prepare_dirs(directory)
    create_shared_objects(directory)
    prepare_netlists(directory)
    run_simulations([dir_osdi, dir_built_in])

    return dir_osdi, dir_built_in
