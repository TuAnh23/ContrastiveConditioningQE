from utils import load_text_file, write_text_file
import argparse
from sacremoses import MosesTokenizer, MosesDetokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split', type=str, default="dev")
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()
    print(args)

    if args.data_split == "dev":
        tok_src_sents = load_text_file(f"en-de-dev/dev.src")
        tok_tgt_sents = load_text_file(f"en-de-dev/dev.mt")
        labels_src = load_text_file(f"en-de-dev/dev.source_tags")
    else:
        raise RuntimeError(f"Split {args.data_split} not available.")

    log2_prob_sent = load_text_file(f"{args.output_dir}/sent_log_prob.out")
    prob_sent = [2**float(x) for x in log2_prob_sent]

    # For each word in the source sentence, record the difference of the translation's score given the original source
    # and given the source where the word is replaced with 'unk'
    prob_sent_counter = 0
    diff_prob = []

    # Separate the prob diff of OK words and BAD words
    ok_prob_diff = []
    bad_prob_diff = []

    for sent_i in range(len(tok_src_sents)):
        diff_prob_per_sent = []
        original_score = prob_sent[prob_sent_counter]
        prob_sent_counter = prob_sent_counter + 1
        ok_bad_labels = labels_src[sent_i].split()
        for tok_i in range(len(tok_src_sents[sent_i].split())):
            prob_diff = str(original_score - prob_sent[prob_sent_counter])
            prob_sent_counter = prob_sent_counter + 1
            diff_prob_per_sent.append(prob_diff)
            if ok_bad_labels[tok_i] == "BAD":
                bad_prob_diff.append(prob_diff)
            elif ok_bad_labels[tok_i] == "OK":
                ok_prob_diff.append(prob_diff)
            else:
                raise RuntimeError(f"Labels {ok_bad_labels[tok_i]} unknown.")
        diff_prob.append(diff_prob_per_sent)

    write_text_file(ok_prob_diff, f"{args.output_dir}/ok_prob_diff.txt")
    write_text_file(bad_prob_diff, f"{args.output_dir}/bad_prob_diff.txt")

    write_text_file([' '.join(x) for x in diff_prob], f"{args.output_dir}/prob_diff.txt")


if __name__ == "__main__":
    main()
