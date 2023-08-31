from utils import load_text_file, write_text_file
import argparse
from sacremoses import MosesTokenizer, MosesDetokenizer
import gensim.downloader as api


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_split', type=str, default="dev")
    args = parser.parse_args()
    print(args)

    if args.data_split == "dev":
        tok_src_sents = load_text_file(f"en-de-dev/dev.src")
        tok_tgt_sents = load_text_file(f"en-de-dev/dev.mt")
    else:
        raise RuntimeError(f"Split {args.data_split} not available.")

    output_src_sents = []
    output_tgt_sents = []
    detokenizer_src = MosesDetokenizer(lang='en')
    detokenizer_tgt = MosesDetokenizer(lang='de')

    word_replacement_model = api.load('glove-wiki-gigaword-100')

    for sent_i in range(len(tok_src_sents)):
        splitted_tok_src_sent = tok_src_sents[sent_i].split()
        # Add the original src to the list
        output_src_sents.append(detokenizer_src.detokenize(splitted_tok_src_sent))
        # Add the corresponding translation
        output_tgt_sents.append(detokenizer_tgt.detokenize(tok_tgt_sents[sent_i].split()))

        # Add the perturbed src sentences to the list
        for tok_i in range(0, len(splitted_tok_src_sent)):
            tok_new_sent = []
            for j in range(0, len(splitted_tok_src_sent)):
                if j == tok_i:
                    # Replace the token
                    # tok_new_sent.append('')
                    original_tok = splitted_tok_src_sent[tok_i]
                    if original_tok in word_replacement_model.index_to_key:
                        new_tok = word_replacement_model.most_similar(positive=[original_tok, 'not'], topn=1)[0][0]
                    else:
                        new_tok = ''
                    tok_new_sent.append(new_tok)
                else:
                    # Keep the same token
                    tok_new_sent.append(splitted_tok_src_sent[j])
            output_src_sents.append(detokenizer_src.detokenize(tok_new_sent))
            output_tgt_sents.append(detokenizer_tgt.detokenize(tok_tgt_sents[sent_i].split()))

    if args.data_split == "dev":
        write_text_file(output_src_sents, "en-de-dev/input.en")
        write_text_file(output_tgt_sents, "en-de-dev/input.de")
    else:
        raise RuntimeError(f"Split {args.data_split} not available.")


if __name__ == "__main__":
    main()
