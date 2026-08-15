import os
import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
import multiprocessing as mp

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_SINGLE_BYTES = tuple(bytes([i]) for i in range(256))
PARALLEL_FILE_SIZE = 50 * 1024 * 1024 # 50MB

## 并行优化 worker 函数
def _pre_tokenize_chunk(args):
    input_path, start, end, special_tokens = args
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end-start).decode("utf-8", errors="ignore")

        pattern = "|".join(re.escape(t) for t in special_tokens)

        pieces = re.split(pattern, text)

        corpus = {}
        for piece in pieces:
            for m in re.finditer(PAT, piece):
                token = m.group()
                enc = token.encode("utf-8")
                key = tuple(_SINGLE_BYTES[b] for b in enc)
                corpus[key] = corpus.get(key, 0) + 1
    
    return corpus

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

    file_size = os.path.getsize(input_path)
    offset_list = find_chunk_boundaries(open(input_path, "rb"), os.cpu_count(), b"<|endoftext|>")
    corpus = {}
    tasks = []
    for i in range(len(offset_list) - 1):
        tasks.append((input_path, offset_list[i], offset_list[i+1], special_tokens))

    if file_size < PARALLEL_FILE_SIZE:
        ## 文件大小小于阈值，不进行并行
        corpus = _pre_tokenize_chunk((input_path, 0, file_size, special_tokens))
    else:
        with mp.Pool(processes=os.cpu_count()) as pool:
            partials = pool.map(_pre_tokenize_chunk, tasks)
        for partial in partials:
            for key, freq in partial.items():
                corpus[key] = corpus.get(key, 0) + freq

    pair_counts = {}
    ## 建立一个倒排索引
    pair_to_keys = {}

    for tup, freq in corpus.items():

        for i in range(len(tup) - 1):
            pair = (tup[i], tup[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + freq
            pair_to_keys.setdefault(pair, set()).add(tup)

    merges = []

    while 256 + len(merges) + len(special_tokens) < vocab_size:

        if pair_counts == {}:
            break

        best = max(pair_counts.items(), key=lambda kv: (kv[1], kv[0]))[0]

        ## 利用倒排索引直接找需要发生merge 的tuple
        affected = list(pair_to_keys.get(best, ()))

        for key in affected:
            freq = corpus.pop(key)
            i = 0
            result = []
            changed = False
            while i < len(key):
                if i + 1 < len(key) and (key[i], key[i+1]) == best:
                    result.append(key[i] + key[i + 1])
                    i += 2
                    changed = True

                else:
                    result.append(key[i])
                    i += 1
            
            new_key = tuple(result)
            corpus[new_key] = freq

            if changed:
                for i in range(len(key) - 1):
                    pair = (key[i], key[i+1])
                    pair_counts[pair] = pair_counts.get(pair, 0) - freq
                    if pair_counts[pair] <= 0:
                        del pair_counts[pair]
                    pair_to_keys[pair].discard(key)

                for i in range(len(new_key) - 1):
                    pair = (new_key[i], new_key[i+1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + freq
                    pair_to_keys.setdefault(pair, set()).add(new_key)


        merges.append(best)

    vocab = {}

    token_id = 0
    for i in range(256):
        vocab[token_id] = _SINGLE_BYTES[i]
        token_id += 1

    for merge in merges:
        vocab[token_id] = merge[0] + merge[1]
        token_id += 1

    for special_token in special_tokens:
        vocab[token_id] = special_token.encode("utf-8")
        token_id += 1

    return vocab, merges

        

    

if __name__ == "__main__":
    import time
    start = time.perf_counter()
    train_bpe('data/TinyStoriesV2-GPT4-train.txt', 10000, ['<|endoftext|>'])
    end = time.perf_counter()
    print(end-start)