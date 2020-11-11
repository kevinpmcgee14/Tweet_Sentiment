import torchtext
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import Vocab 
from torchtext.data.functional import numericalize_tokens_from_iterator


class Tokenizer(object):

    def __init__(self, type_str, text, pre_rules=[], post_rules=[]):
        self.tokenizer = get_tokenizer(type_str)
        self.pre_rules = pre_rules
        self.post_rules = post_rules
        self.vocab = self.make_vocab(text)

    def tokenize(self, texts):
      tokens = []
      if isinstance(texts, str):
        texts = [texts]
      for text in texts:
        token = text
        for rule in self.pre_rules:
            token = rule(token)
        token = self.tokenizer(token)
        for rule in self.post_rules:
            token = [rule(t) for t in token]
        tokens.append(token)
      return tokens

    def make_vocab(self, texts, min_freq=3, max_vocab=60000):
      objs = [] 
      cnt = Counter()
      for sentence in tqdm(texts, total=len(texts)):
        tok_sentence = self.tokenize(sentence)
        objs.append(tok_sentence)
        sentence_count = Counter(tok_sentence)
        for word in sentence_count.keys():
            if word in cnt.keys():
                cnt[word] += sentence_count[word]
            else:
                cnt[word] = 1
      vocab = Vocab(cnt, max_size=max_vocab, min_freq=min_freq)
      return vocab

    def numericalize(self, tokens):
      if isinstance(tokens, str):
        tokens = [tokens]
      ids = []
      itter_nums = numericalize_tokens_from_iterator(self.vocab, tokens)
      for nums in itter_nums:
          ids.append([id for id in nums])
      return ids

    def __call__(self, texts):
        return self.numericalize(self.tokenize(texts))


class TokenizedObjects(object):

    def __init__(self, text, targs, tok):
        self.text = text
        self.targs = pd.get_dummies(targs).values.tolist()
        self.tokenizer = tok
        self.objs = tok.tokenize(self.text)
        self.max_seq = self.get_max_length(self.objs)
        self.ints = tok.numericalize(self.objs)

        ints_objs = np.array([self.ints, self.objs])
        X_train, X_valid, y_train, y_valid = train_test_split(ints_objs.T, np.array(targs), test_size=0.1)
        self.train = {'ints': X_train[:, 0].tolist(), 'objs': X_train[:, 1].tolist(), 'targets': y_train.tolist()}
        self.valid = {'ints': X_valid[:, 0].tolist(), 'objs': X_valid[:, 1].tolist(), 'targets': y_valid.tolist()}
    
    def get_max_length(self, objs):
      sorted_i = sorted(objs, key=len, reverse=True)
      return len(sorted_i[0])