import os
import difflib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# ANSI escape codes for beautiful terminal colors
class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

def process_chunk(chunk_exact, chunk_approx):
    """Worker function to process a chunk of lines in parallel."""
    local_total_error = 0
    local_line_count = 0
    local_mismatches = 0
    local_max_error = 0  

    for line_e, line_a in zip(chunk_exact, chunk_approx):
        parts_e = line_e.strip().split()
        parts_a = line_a.strip().split()
        
        # Make sure it's a valid coordinate line with at least 4 values
        if len(parts_e) < 4 or len(parts_a) < 4:
            continue
            
        try:
            intensity_exact = int(parts_e[3])
            intensity_approx = int(parts_a[3])
            
            error = abs(intensity_exact - intensity_approx)
            local_total_error += error
            local_line_count += 1
            
            if error > 0:
                local_mismatches += 1
                
            
            if error > local_max_error:
                local_max_error = error 

        except ValueError:
            pass 

    return local_total_error, local_line_count, local_mismatches, local_max_error


def evaluate_algorithms(exact_file: str, approx_file: str) -> None:
    if not os.path.exists(exact_file) or not os.path.exists(approx_file):
        print(f"{Color.RED}Error: Make sure both '{exact_file}' and '{approx_file}' exist.{Color.RESET}")
        return

    # 1. Read files into memory
    with open(exact_file, 'r') as f1, open(approx_file, 'r') as f2:
        lines_exact = f1.readlines()
        lines_approx = f2.readlines()

    # ==========================================
    # PART 1: VISUAL DIFFERENCES (Sequential)
    # ==========================================
    print(f"\n{Color.CYAN}=== PART 1: VISUAL DIFFERENCES ==={Color.RESET}")
    if len(lines_exact) > 50000:
        print(f"{Color.YELLOW}Files are very large. Skipping visual diff to save time.{Color.RESET}")
    else:
        diff = difflib.Differ()
        diff_result = list(diff.compare(lines_exact, lines_approx))
        
        differences_found = False
        for line in diff_result:
            prefix = line[:2]
            content = line[2:].strip()
            
            if prefix == '- ':
                print(f"{Color.RED}[Exact ] {content}{Color.RESET}")
                differences_found = True
            elif prefix == '+ ':
                print(f"{Color.GREEN}[Approx] {content}{Color.RESET}")
                
        if not differences_found:
            print(f"{Color.GREEN}Files are completely identical line-by-line!{Color.RESET}")

    # ==========================================
    # PART 2: MAE CALCULATION (Parallelized)
    # ==========================================
    print(f"\n{Color.CYAN}=== PART 2: MEAN ABSOLUTE ERROR (MAE) ==={Color.RESET}")
    
    total_error = 0
    line_count = 0
    mismatches = 0
    global_max_error = 0 

    # Determine chunk size based on CPU cores
    num_cores = multiprocessing.cpu_count()
    total_lines = min(len(lines_exact), len(lines_approx))
    
    if total_lines == 0:
        print(f"{Color.RED}Files are empty or invalid.{Color.RESET}")
        return

    # Create chunks for parallel processing
    chunk_size = max(1, total_lines // num_cores)
    chunks_exact = [lines_exact[i:i + chunk_size] for i in range(0, total_lines, chunk_size)]
    chunks_approx = [lines_approx[i:i + chunk_size] for i in range(0, total_lines, chunk_size)]

    # Spin up worker processes
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        results = executor.map(process_chunk, chunks_exact, chunks_approx)

    # THIS WAS THE FIX: Unpacking 4 variables here
    for local_error, local_count, local_mismatches, local_max in results:
        total_error += local_error
        line_count += local_count
        mismatches += local_mismatches
        
        if local_max > global_max_error:
            global_max_error = local_max

    if line_count == 0:
        print(f"{Color.RED}No valid data found to calculate MAE.{Color.RESET}")
        return

    # Calculate final score
    mae = total_error / line_count
    
    print(f"Total Points Checked : {line_count}")
    print(f"Points with Errors   : {mismatches}")
    print(f"Max Single Error     : {Color.RED}{global_max_error}{Color.RESET}") 
    print(f"Final MAE Score      : {Color.YELLOW}{mae:.4f}{Color.RESET}")
    
    # Grading Verdict
    print("\n[Verdict]: ", end="")
    if mae == 0.0:
        print(f"{Color.GREEN}Perfect Match! (100% Correct){Color.RESET}\n")
    elif mae <= 5.0:
        print(f"{Color.GREEN}Excellent! Deviation is extremely low and well within approximation limits.{Color.RESET}\n")
    else:
        print(f"{Color.RED}Warning: High MAE. Your approximate algorithm might be too inaccurate.{Color.RESET}\n")

if __name__ == "__main__":
    evaluate_algorithms('approx.knn.txt', '../CUDA/approx_knn.txt')