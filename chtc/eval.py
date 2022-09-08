import pandas as pd
import subprocess
import os
import numpy as np
import multiprocessing as mp
import sys

file = "predict-128-128-10ep-ep10"

with open("tok-test-src.tsv") as f: 
    before = f.read().replace("__gcd", "gcd")
before = before.split("\n")[:-1]

with open(file) as f: 
    after = f.read().replace("__gcd", "gcd")
after = after.split("\n")

df = pd.read_csv('train/split/spoc-train-test.tsv', sep='\t')

not_null = np.where(df['text'].notnull())[0]

packages = ["<iostream>", "<vector>", "<map>", "<set>", "<queue>", "<math.h>", "<iomanip>", "<tuple>", "<stack>", "<numeric>", "<sstream>"]

def reconstruct_code(code_df): 
    global packages 
    code = ""
    for p in packages: 
        code += "#include " + p + "\n"
    code += "using namespace std;\n"
    for t in code_df.itertuples(): 
        code += "\t" * getattr(t, "indent") + getattr(t, "code") + "\n"
    return code

for name, group in df.groupby(["subid"]): 
    code = reconstruct_code(group)
    with open("code/" + str(name) + ".cpp", "w") as f: 
        f.write(code)
        
def check_compilable(subid): 
    try: 
        subprocess.check_output("g++ -std=c++17 -o code_exe/" + subid + " code_rep/" + subid + ".cpp", shell = True, stderr=subprocess.PIPE)
        return 1
    except subprocess.CalledProcessError as e:
        return 0
    
def check_compilable2(subid): 
    try: 
        subprocess.check_output("g++ -std=c++17 -o code_exe/" + subid + "_orginial code/" + subid + ".cpp", shell = True, stderr=subprocess.PIPE)
        return ""
    except subprocess.CalledProcessError as e:
        return subid
    
def parse_test_cases(probid, pp):
    tc = ""
    if pp == "all": 
        with open("testcases/" + probid + "/" + probid + "_testcases.txt") as f: 
            tc = f.read()
    else: 
        with open("testcases/" + probid + "/" + probid + "_testcases_public.txt") as f: 
            tc = f.read()
    tcs_in = []
    tcs_out = []
    tcs = tc.split("###ENDOUTPUT###")
    for i in tcs: 
        tt = i.split("###ENDINPUT###")
        if len(tt) == 2: 
            tcs_in.append(tt[0])
            tcs_out.append(tt[1])
    return (tcs_in, tcs_out)

def run_test_cases(tcs_in, tcs_out, exe, wid): 
    in_file_name = "code_exe/in_file" + str(wid) + ".txt"
    for i in range(len(tcs_in)): 
        with open(in_file_name, "w") as f_in: 
            f_in.write(tcs_in[i])

        with open(in_file_name, "r") as f_in: 
            try: 
                out = subprocess.run("./" + exe, shell = True, timeout = 2, stdin=f_in, stdout=subprocess.PIPE)
            except subprocess.TimeoutExpired:
                return 0 
            if out.stdout.decode("utf-8").strip() != tcs_out[i].strip():
                return 0
    return 1  

def work(start, end, wid, beam):
    global not_null
    
    total_compilable = 0 
    total_passed = 0 

    with open("output" + str(wid) + ".txt", "w") as output_file: 
        for i in range(start, end): 
            if i % 10 == 0: 
                print("wid: {}, i: {}, total_compilable: {}, total_passed: {}".format(wid, i, total_compilable, total_passed))

            output_str = ""

            idx = not_null[i]
            subid = str(df.at[idx, "subid"])
            output_str += subid + ","

            probid = str(df.at[idx, "probid"])
            output_str += probid + ","

            # original compilable 
            if check_compilable2(subid) == "": 
                output_str += "True,"
            else: 
                output_str += "False,,,,,\n"
                output_file.write(output_str)
                continue

            tcs_in, tcs_out = parse_test_cases(probid, "all")

            # original passed
            if run_test_cases(tcs_in, tcs_out, "code_exe/" + subid + "_orginial", wid) == 1: 
                output_str += "True,"
            else: 
                output_str += "False,,,,\n"
                output_file.write(output_str)
                continue

            rep_line_idx = df.at[idx, "line"] + len(packages) + 1
            rep_line_indent = df.at[idx, "indent"]

            with open("code/" + subid + ".cpp") as f:
                original_code = f.read()
            rep_code = original_code.split("\n")

            compilable = 0 
            passed = 0 
            for j in range(beam):
                rep_line = "\t" * rep_line_indent + after[i * 5 + j]
                rep_code[rep_line_idx] = rep_line
                rep_code_join = "\n".join(rep_code)

                with open("code_rep/" + subid + ".cpp", "w") as f: 
                    f.write(rep_code_join)

                res = check_compilable(subid)

                if res == 1: 
                    compilable += 1
                    total_compilable += 1
                    pass_res = run_test_cases(tcs_in, tcs_out, "code_exe/" + subid, wid)
                    if pass_res == 1:
                        passed += 1 
                        total_passed += 1  

                if j == 0: 
                    if compilable == 1: 
                        output_str += "True,"
                    else: 
                        output_str += "False,"
                    if passed == 1: 
                        output_str += "True,"
                    else: 
                        output_str += "False,"

            if compilable > 0: 
                output_str += "True,"
            else: 
                output_str += "False,\n"
                output_file.write(output_str)
                continue 

            if passed > 0:
                output_str += "True\n"
            else: 
                output_str += "False\n"
            output_file.write(output_str)
    return (total_compilable, total_passed)

def main(): 
    global file
    # start scale file
    start = sys.argv[0]
    scale = sys.argv[1]
    if sys.argv[2] != None: 
        file = sys.argv[2]
    if sys.argv[3] != None: 
        beam = sys.argv[3]
    print(work(start * scale, (start + 1) * scale, start), beam=5)
    
if __name__=="__main__":
#     num_workers = 2
#     args = []
#     for i in range(num_workers): 
#         args.append((i * scale, (i + 1) * scale, i))
#     with mp.Pool(processes=num_workers) as pool:
#         pool.starmap(work, args)
#     print(work(i * scale, (i + 1) * scale, i))
    main()