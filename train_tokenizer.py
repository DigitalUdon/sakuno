import os
import sentencepiece as spm
from transformers import T5Tokenizer


def train_tokenizer(corpus_file, user_defined_symbols):
    os.makedirs("models", exist_ok=True)
    os.makedirs("models/custom_t5_tokenizer", exist_ok=True)
    
    spm.SentencePieceTrainer.train(
        input=corpus_file,
        model_type="unigram",
        model_prefix="models/sentencepiece_model",
        vocab_size=16000,
        accept_language=["ja", "en"],
        character_coverage=0.9995,
        user_defined_symbols=user_defined_symbols,
        byte_fallback=True,
        add_dummy_prefix=False,
        allow_whitespace_only_pieces=True,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_id=0,
        bos_piece="<BOS>",
        eos_piece="<EOS>",
        pad_piece="<PAD>",
        unk_piece="<UNK>",
        num_threads=os.cpu_count(),
        input_sentence_size=1000000,
        shuffle_input_sentence=True,
    )

    tokenizer = T5Tokenizer(
        vocab_file="models/sentencepiece_model.model",
        unk_token="<UNK>",
        eos_token="<EOS>",
        pad_token="<PAD>",
        extra_ids=0
    )
    
    tokenizer.save_pretrained("models/custom_t5_tokenizer")


if __name__ == "__main__":
    user_defined_symbols = [
        "<CONTEXT>",
        "</CONTEXT>",
        "<ROLE>",
        "</ROLE>",
        "User",
        "Assistant",
    ]
    
    train_tokenizer("corpus.txt", user_defined_symbols)
