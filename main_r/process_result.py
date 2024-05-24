import os
import glob
import argparse


def make_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default="D:\Code\sphere2vec_exp")
    parser.add_argument('--filter_crit', default="model_birdsnap_ebird_meta")
    return parser


def extract_last_top1_acc_from_log(log_lines):
    top1_acc_lines = [line.strip() for line in log_lines if "Top 1\tacc (%)" in line]
    return top1_acc_lines[-1] if top1_acc_lines else None


def process_log_files(log_dir):
    log_files = glob.glob(os.path.join(log_dir, "*.log"))
    results = []

    for log_file in log_files:
        with open(log_file, 'r') as f:
            log_lines = f.readlines()
            last_top1_acc_line = extract_last_top1_acc_from_log(log_lines)
            if last_top1_acc_line:
                results.append((log_file, last_top1_acc_line))

    return results


def parse_top1_acc_line(line):
    parts = line.split()
    return float(parts[-1])


def main(args):
    log_dir = args.log_dir
    results = process_log_files(log_dir)
    filter_criterion = args.filter_crit
    # do filtration
    filtered_results = [(log_file, line) for log_file, line in results if
                        filter_criterion in log_file]

    # find max Top1 acc
    max_acc = -1
    max_acc_result = None

    for log_file, line in filtered_results:
        top1_acc = parse_top1_acc_line(line)
        if top1_acc is not None and top1_acc > max_acc:
            max_acc = top1_acc
            max_acc_result = (log_file, line)

    if max_acc_result:
        log_file, line = max_acc_result
        print(f"File: {log_file}")
        print(f"Top 1 acc line: {line}")


if __name__ == "__main__":
    parser = make_args_parser()
    args = parser.parse_args()
    main(args)
