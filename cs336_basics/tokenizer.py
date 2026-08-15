import os
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path: str | os.PathLike, 
              vocab_size: int, 
              special_tokens: list[str]
              )-> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    with open(input_path, encoding="utf-8") as f:
        text = f.read()

        pattern = "|".join(re.escape(t) for t in special_tokens)

        pieces = re.split(pattern, text)

        corpus = {}
        for piece in pieces:
            for m in re.finditer(PAT, piece):
                token = m.group()
                enc = token.encode("utf-8")
                key = tuple(bytes([b]) for b in enc)
                corpus[key] = corpus.get(key, 0) + 1

    merges = []

    while 256 + len(merges) + len(special_tokens) < vocab_size:

        pair_counts = {}

        for tup, freq in corpus.items():

            for i in range(len(tup) - 1):
                pair = (tup[i], tup[i+1])
                pair_counts[pair] = pair_counts.get(pair, 0) + freq


        if pair_counts == {}:
            break


        best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

        new_corpus = {}
        for tup, freq in corpus.items():
            i = 0
            result = []
            while i < len(tup):
                if i + 1 < len(tup) and (tup[i], tup[i+1]) == best:
                    result.append(tup[i] + tup[i + 1])
                    i += 2
                else:
                    result.append(tup[i])
                    i += 1
            
            key = tuple(result)

            new_corpus[key] = new_corpus.get(key, 0) + freq


        corpus = new_corpus
        merges.append(best)

    vocab = {}

    token_id = 0
    for i in range(256):
        vocab[token_id] = bytes([i])
        token_id += 1

    for merge in merges:
        vocab[token_id] = merge[0] + merge[1]
        token_id += 1

    for special_token in special_tokens:
        vocab[token_id] = special_token.encode("utf-8")
        token_id += 1

    return vocab, merges

        
        

if __name__ == "__main__":
    train_bpe('tests/fixtures/tinystories_sample_5M.txt', 500, ['<|endoftext|>'])