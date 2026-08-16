import os
import regex as re
from cs336_basics.pretokenization_example import find_chunk_boundaries
import multiprocessing as mp
import json
from cs336_basics.utils import gpt2_bytes_to_unicode
import time
from collections.abc import Iterator

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

def save_vocab_merges(vocab, merges, vocab_path, merges_path):

    char_map = gpt2_bytes_to_unicode()

    vocab_saved = {}
    for token_id, token_bytes in vocab.items():
        token = "".join(char_map[b] for b in token_bytes)
        vocab_saved[token] = token_id

    with open (vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_saved, f, indent=2, ensure_ascii=False)

    with open(merges_path, "w", encoding="utf-8") as f:
        for merge in merges:
            f.write("".join(char_map[b] for b in merge[0]) + " " + "".join(char_map[b] for b in merge[1]) + "\n")

def load_vocab_merges(vocab_path, merges_path):
    rev_map = {v: k for k, v in gpt2_bytes_to_unicode().items()}

    vocab_saved = json.load(open(vocab_path, encoding="utf-8"))
    vocab = {}

    for token, token_id in vocab_saved.items():
        token_bytes = bytes([rev_map[c] for c in token])
        vocab[token_id] = token_bytes
    
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            cleaned = line.rstrip()
            if not cleaned:
                continue
            tok1_str, tok2_str = cleaned.split()
            merge = (
                bytes([rev_map[c] for c in tok1_str]),
                bytes([rev_map[c] for c in tok2_str])
            )
            merges.append(merge)

    return vocab, merges

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self._rev_vocab = {}
        for token_id, token_bytes in self.vocab.items():
            self._rev_vocab[token_bytes] = token_id

        self._special_tokens_set = set(self.special_tokens)

    @classmethod
    def from_files(cls, vocab_path, merges_path, special_tokens=None):
        vocab, merges = load_vocab_merges(vocab_path, merges_path)
        return cls(vocab, merges, special_tokens)

    def encode(self, text) -> list[int]:
        result = []

        if self.special_tokens:
            # 对special tokens 排序，字符串长的排在前面，才能正确把 "<|endoftext|><|endoftext|>" 识别出来
            # 因为正则表达式是从左到右匹配的，如果 <|endoftext|> 在前，就会匹配为两个token
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)

            pattern = "|".join(re.escape(t) for t in sorted_special_tokens)
            pieces = re.split(f"({pattern})", text)  # 带捕获组的 re.split 把分隔符(special_token)也放进结果
        else:
            pieces = [text]
        
        for piece in pieces:
            if piece in self._special_tokens_set:
                result.append(self._rev_vocab[piece.encode("utf-8")])
                continue
            
            for m in re.finditer(PAT, piece):
                token = m.group()
                enc = token.encode("utf-8")
                key = tuple(_SINGLE_BYTES[b] for b in enc)
    
                for merge in self.merges:
                        i = 0
                        new_key = []
                        while i < len(key):
                            if i + 1 < len(key) and (key[i], key[i+1]) == merge:
                                new_key.append(key[i] + key[i + 1])
                                i += 2
            
                            else:
                                new_key.append(key[i])
                                i += 1
                        
                        key = tuple(new_key)

                for token in key:
                    token_id = self._rev_vocab[token]
                    result.append(token_id)

        return result

    def encode_iterable(self, iterable) -> Iterator[int]:
        pendding = ""
        for line in iterable:
            text = pendding + line
            stripped = text.rstrip()    # 去掉尾部空白后的部分
            pendding = text[len(stripped):] # 被去掉的空白，留着下一行一起处理

            if stripped:
                yield from self.encode(stripped)

        if pendding:
            yield from self.encode(pendding)

    def decode(self, ids) -> str:
        result = b"".join(self.vocab[token_id] for token_id in ids).decode("utf-8", errors="replace")
        return result


if __name__ == "__main__":
    start = time.perf_counter()
    # vocab, merges = train_bpe('data/TinyStoriesV2-GPT4-train.txt', 10000, ['<|endoftext|>'])
    vocab, merges = train_bpe('tests/fixtures/tinystories_sample_5M.txt', 1000, ['<|endoftext|>'])
    # vocab, merges = train_bpe('data/owt_train.txt', 32000, ['<|endoftext|>']) ## OOM
    end = time.perf_counter()

    # save_vocab_merges(vocab, merges, "data/TinyStoriesV2-GPT4-train_vocab.json", "data/TinyStoriesV2-GPT4-train_merges.txt")
    # loaded_vocab, loaded_merges = load_vocab_merges("data/TinyStoriesV2-GPT4-train_vocab.json", "data/TinyStoriesV2-GPT4-train_merges.txt")

    save_vocab_merges(vocab, merges, "data/tinystories_sample_5M_vocab.json", "data/tinystories_sample_5M_merges.txt")
    loaded_vocab, loaded_merges = load_vocab_merges("data/tinystories_sample_5M_vocab.json", "data/tinystories_sample_5M_merges.txt")

    # save_vocab_merges(vocab, merges, "data/owt_train_vocab.json", "data/owt_train_merges.txt")
    # loaded_vocab, loaded_merges = load_vocab_merges("data/owt_train_vocab.json", "data/owt_train_merges.txt")

    tokenizer = Tokenizer(vocab, merges, ['<|endoftext|>'])
    print(tokenizer.decode(tokenizer.encode("Hello, how are you?")))

    print(end-start)