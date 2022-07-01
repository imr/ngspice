#this file defines some common routines used by the OSDI test cases
import os
import shutil
import glob
from pathlib import Path
from typing import Optional
import regex as re
from subprocess import run, PIPE
import pandas as pd
import numpy as np
from math import atan2
import sys

# specify location of Ngspice executable to be tested
directory_testing = os.path.dirname(__file__)
ngspice_path = os.path.join(directory_testing, "../release/src/ngspice")
ngspice_path = os.path.abspath(ngspice_path)

rtol = 0.032
atol_dc = 1e-14
atol_ac = 4e-19

twoPi = 8.0*atan2(1.0,1.0)

def create_shared_objects(directory):
    c_files = []
    for c_file in glob.glob(directory + "/*.c"):
        basename = Path(c_file).stem
        c_files.append(basename)

    for c_file in c_files:
        run(
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
        run(
            ["gcc", "-shared", "-o", c_file + ".osdi", c_file + ".o", "-ggdb"],
            cwd=directory,
        )
        run(
            ["mv", c_file + ".osdi", "test_osdi/" + c_file + ".osdi"], cwd=directory
        )
        run(["rm", c_file + ".o"], cwd=directory)

    # for va_file in glob.glob(directory + "/*.va"):
    #     result = run(
    #         [
    #             "openvaf","-b", va_file
    #         ],
    #         # capture_output=True,
    #         cwd=directory,
    #     )

    #     run(
    #         ["cp", result.stdout[:-1], "test_osdi/" + Path(va_file).stem + ".osdi"], cwd=directory
    #     )





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
        run(
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



def parse_list(line):
    return (val for val in re.split(r"\s+", line) if val != '')

def parse_temps(line):
    return [temp for temp in parse_list(line)]


class TestInfo:
    biases: Optional[dict[str, str]] = None
    bias_list: Optional[tuple[str, list[str]]]  = None
    bias_sweep = None
    temps: Optional[list[str]] = None
    freqs: Optional[str] = None
    dc_outputs: Optional[list[tuple[str, str]]] = None
    ac_outputs:  Optional[dict[str,list[tuple[str, str, bool, str, str]]]] = None
    instanceParameters: str= ""
    modelParameters: str = ""
    line: str = ""

    def __init__(self, name, lines, parent):
        self.name = name
        self.lines= lines
        self.parse()
        if self.temps is None:
            self.temps = parent.temps
        self.pins = parent.pins
        self.floating = parent.floating



    def parse_temps(self):
        temps = parse_temps(self.line)
        if self.temps is None:
            self.temps = temps
        else:
            self.temps += temps

    def parse_model_params(self):
        for param in parse_list(self.line):
            path = Path(param)
            if path.exists():
                self.modelParameters = path.read_text()
            else:
                self.modelParameters += f"+ {param}\n"

    def parse_instance_params(self):
        for param in parse_list(self.line):
            self.instanceParameters += f" {param}"


    def parse_bias_list(self):
        if self.bias_list:
            raise ValueError(f"ERROR second bias_list spec {self.line}")
        res = re.match(r"V\s*\(\s*(\w+)\s*\)\s*=", self.line)
        pin = res[1]
        vals = self.line[res.end():].strip()
        vals = [val for val in re.split(r"\s*,\s*", vals)]
        self.bias_list = (pin, vals)


    def parse_biases(self):
        if self.biases:
            raise ValueError(f"ERROR second biases spec {self.line}")
        self.biases = {}
        for bias in parse_list(self.line):
            res = re.match(r"V\s*\(\s*(\w+)\s*\)\s*=", bias)
            pin = res[1]
            val = bias[res.end():].strip()
            self.biases[pin] = val

    def parse_outputs(self):
        for output in parse_list(self.line):
            res = re.match(r"([IV])\s*\(\s*(\w+)\s*\)", output)
            if res:
                pin = res[2]
                if res[1] == "I":
                    output = f"i(v{pin})", f"I({pin})"
                else:
                    output = f"v({pin})", f"V({pin})"
                if self.dc_outputs:
                    self.dc_outputs.append(output)
                else:
                    self.dc_outputs = [output]
                continue


            res = re.match(r"([CG])\s*\(\s*(\w+)\s*,\s*(\w+)\s*\)", output)
            if res:
                kind = res[1]
                pin1 = res[2]
                pin2 = res[3]

                if kind == "G":
                    output = f"real(i(v{pin1}))", f"g({pin1},{pin2})", False, pin1, pin2
                elif kind == "C":
                    output = f"imag(i(v{pin1}))", f"c({pin1},{pin2})", True, pin1, pin2

                if self.ac_outputs:
                    if pin2 in self.ac_outputs:
                        self.ac_outputs[pin2].append(output)
                    else:
                        self.ac_outputs[pin2] = [output]
                else:
                    self.ac_outputs = {pin2: [output]}
                continue

    def parse_frequency(self):
        res = re.match(r"(lin|oct|dec)\s+(\S+)\s+(\S+)\s+(\S+)\s*", self.line)
        kind = res[1]
        num_steps = int(res[2])
        start = res[3]
        end = res[4]
        if start != end:

            if kind == "lin":
                num_points = num_steps + 1
            else:
                num_points = num_steps
        else:
            assert num_steps == 1
            num_points = 1
        self.freqs = f"{kind} {num_points} {start} {end}"


    def parse_bias_sweep(self):
        res = re.match(r"V\s*\(\s*(\w+)\s*\)\s*=", self.line)
        pin = res[1]
        args = self.line[res.end():]
        args = [float(arg) for arg in re.split(r"\s*,\s*", args)]
        if len(args) != 3:
            raise ValueError(f"bias sweep must have 3 arguments found {args} in {self.line}")
        self.bias_sweep = (pin, args)


    def try_parse(self, prefix: str, f):
        if self.line.startswith(prefix):
            self.line = self.line[len(prefix):].strip()
            f()

    def parse_line(self):
        if self.try_parse("temperature", self.parse_temps):
            return
        if self.try_parse("modelParameters", self.parse_model_params):
            return
        if self.try_parse("instanceParameters", self.parse_instance_params):
            return
        if self.try_parse("biasList", self.parse_bias_list):
            return
        if self.try_parse("listBias", self.parse_bias_list):
            return
        if self.try_parse("biases", self.parse_biases):
            return
        if self.try_parse("output", self.parse_outputs):
            return
        if self.try_parse("outputs", self.parse_outputs):
            return
        if self.try_parse("biasSweep", self.parse_bias_sweep):
            return
        if self.try_parse("freq", self.parse_frequency):
            return
        if self.try_parse("frequency", self.parse_frequency):
            return

    def parse(self):
        for line in self.lines:
            self.line = line
            self.parse_line()
    
    def gen_netlist(self, osdi_file, va_module, type_arg):
        if self.bias_list:
            bias_start = f"foreach bias {' '.join(self.bias_list[1])}\nalter v{self.bias_list[0]}=$bias"
            bias_end = "end"
        else:
            bias_start = bias_end = ""
        
        if self.dc_outputs:
            if not self.bias_sweep:
                raise ValueError("dc bias sweep msising")
            outputs = " ".join(output for output, _ in self.dc_outputs)
            sweep = f"dc v{self.bias_sweep[0]} {self.bias_sweep[1][0]} {self.bias_sweep[1][1]} {self.bias_sweep[1][2]}\n wrdata {self.dc_results_path()} {outputs}"
        elif self.ac_outputs:
            freqs = self.freqs
            if not self.freqs:
                freqs = f"lin 1 {1/twoPi} {1/twoPi}"
            if self.bias_sweep:
                if self.bias_list:
                    bias_start += "\n"
                    bias_end += "\n"
                vals = np.arange(self.bias_sweep[1][0], self.bias_sweep[1][1] + self.bias_sweep[1][2]*0.1, self.bias_sweep[1][2])
                vals = [str(val) for val in vals]
                bias_start += f"foreach bias {' '.join(vals)}\nalter v{self.bias_sweep[0]}=$bias"
                bias_end += "end"

            sweep = ""
            for pin, outputs in self.ac_outputs.items():
                sweep += f"alter v{pin} ac = 1\nac {freqs}\n"
                outputs = " ".join(output[0] for output in outputs)
                sweep += f"wrdata {self.ac_results_path(pin)} {outputs}\n"
                sweep += f"alter v{pin} ac = 0\n"
        else:
            return ""

        biases = self.biases
        if not biases:
            biases = dict()

        source = "\n".join(f"v{pin} {pin} {0} dc={biases.get(pin, 0)}" for pin in self.pins if not pin in self.floating)
        source += "".join(f"\nr{i} {pin} {0} r=1G" for i,pin in enumerate(self.floating))

        return f"""CMC testsuite {self.name}
.options abstol=1e-15

{source}

.model test_model {va_module}
{self.modelParameters} {type_arg}

A1 {' '.join(self.pins)} test_model {self.instanceParameters}

.control
pre_osdi {osdi_file}

set filetype=ascii
set wr_vecnames
set wr_singlescale
set appendwrite

foreach tamb {' '.join(self.temps)}
  set temp=$tamb
  {bias_start}
    {sweep}
   {bias_end}
end
quit 0
.endc
.end
"""

    def dc_results_path(self) -> Path:
        return Path("results")/f"{self.name}.ngspice"

    def ac_results_path(self, pin: str) -> Path:
        return Path("results")/f"{self.name}_{pin}.ngspice"

    def run(self, osdi_file, va_module, type_arg):
        if not (self.dc_outputs or self.ac_outputs):
            return

        print(f"running {self.name}...")

        netlist_path = Path("netlists")/f"{self.name}.sp"
        netlist = self.gen_netlist(osdi_file, va_module, type_arg)
        Path(netlist_path).write_text(netlist)

        res = run([ngspice_path, netlist_path, "-b"], capture_output=True)
        res.check_returncode()
        # res.check_returncode()

        reference_path = Path("reference")/f"{self.name}.standard"
        references = pd.read_csv(reference_path, sep="\\s+")

        if self.dc_outputs:
            results_path = self.dc_results_path()

            if not results_path.exists():
                print(f"ERROR check failed for {self.name}\nsimulation file is missing - likely convergence issues!")
                return

            results = pd.read_csv(results_path, sep="\\s+")
            results = results.apply(pd.to_numeric, errors='coerce')
            firstcol = results.iloc[:,1].to_numpy()
            results = results[np.bitwise_not(np.isnan(firstcol))]

            for result_col, ref_col in self.dc_outputs:
                reference = references[ref_col].to_numpy()
                result = results[result_col].to_numpy()
                if "I(" in ref_col:
                    result = -result

                adiff = np.abs(result-reference)
                rdiff = adiff/reference
                err = np.bitwise_not(np.bitwise_or(rdiff < rtol, adiff < atol_dc))
                if not np.any(err):
                    continue
                maxatol = np.max(adiff[err])
                maxrtol = np.max(rdiff[err])
                print(f"ERROR check failed for {ref_col}\nrtol={maxrtol} atol={maxatol}\nresult:\n{result[err]}\nreference:\n{reference[err]}\nrtol:\n{rdiff[err]}")

        elif self.ac_outputs:
            for pin, outputs in self.ac_outputs.items():
                results_path = self.ac_results_path(pin)
                if not results_path.exists():
                    print(f"ERROR check failed for {self.name} (ac {pin})\nsimulation file is missing - likely convergence issues!")
                    continue

                results = pd.read_csv(results_path, sep="\\s+")
                results = results.apply(pd.to_numeric, errors='coerce')
                firstcol = results.iloc[:,1].to_numpy()
                results = results[np.bitwise_not(np.isnan(firstcol))]

                for result_col, ref_col, is_cap, pin1, pin2  in outputs:
                    result = results[result_col].to_numpy()
                    reference = references[ref_col].to_numpy()
                    if is_cap:
                        if"Freq" in references:
                            result = result /(twoPi*results["frequency"])
                        if pin1 == pin2:
                            result = -result
                    else:
                        result = -result

                    adiff = np.abs(result-reference)
                    rdiff = adiff/reference
                    err = np.bitwise_not(np.bitwise_or(rdiff < rtol, adiff < atol_ac))
                    if not np.any(err):
                        continue
                    maxatol = np.max(adiff[err])
                    maxrtol = np.max(rdiff[err])
                    print(f"ERROR check failed for {ref_col}\nrtol={maxrtol} atol={maxatol}\nresult:\n{result[err]}\nreference:\n{reference[err]}\nrtol:\n{rdiff[err]}")





        

    
def removeComments(string):
    string = re.sub(re.compile(r"/\*.*?\*/",re.DOTALL ) ,"" ,string) # remove all occurrences streamed comments (/*COMMENT */) from string
    string = re.sub(re.compile(r"//.*?\n" ) ,"" ,string) # remove all occurrence single-line comments (//COMMENT\n ) from string
    return string

class QaSpec:
    temps: list[str]
    pins: list[str]
    floating: list[str]
    tests: list[TestInfo]
    dir: Path

    def __init__(self, dir: Path):
        self.dir = dir
        self.temps = []
        self.pins = []
        self.tests = []
        self.floating = []
        self.parse()

    def parse(self):
        old_dir = os.getcwd()
        os.chdir(self.dir)
        qa_spec = Path("qaSpec").read_text()
        qa_spec = removeComments(qa_spec)
        lines = [line.strip() for line in qa_spec.split('\n')]

        i = 0
        while i < len(lines):
            line = lines[i]
            i+= 1
            if line.startswith("temperature"):
                line = line[len("temperature"):]
                self.temps = parse_temps(line)
            elif line.startswith("pins"):
                line = line[len("pins"):]
                self.pins = [pin for pin in re.findall(r"\w+", line) if pin != "pins"]

            elif line.startswith("float") or line.startswith("floating"):
                self.floating = [pin for pin in re.findall(r"\w+", line) if pin != "floating" and pin != "float"]
            elif line.startswith("test"):
                test_name = line[4:].strip()
                start = i
                while i < len(lines) and lines[i] != "":
                    i += 1
                end = i

                test = TestInfo(test_name, lines[start:end], self)
                self.tests.append(test)

        os.chdir(old_dir)


    def run(self, va_file, va_module, type_arg, filter=None):
        result = run(
            ["openvaf","-b", va_file],
            stdout=PIPE,
        )
        result.check_returncode()
        osdi_file = result.stdout[:-1].decode("utf-8") 
        
        old_dir = os.getcwd()
        os.chdir(self.dir)

        dirpath = Path('netlists')
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)
        os.mkdir("netlists")

        dirpath = Path('results')
        if dirpath.exists() and dirpath.is_dir():
            shutil.rmtree(dirpath)
        os.mkdir("results")
        for test in self.tests:
            if filter and not test.name in filter:
                continue
            test.run(osdi_file, va_module, type_arg)
        os.chdir(old_dir)
