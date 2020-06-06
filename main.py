from bendeep import sentiment


if __name__ == "__main__":
    model_path = "senti_trained.pt"
    vocab_path = "vocab.txt"
    text = "রোহিঙ্গা মুসলমানদের দুর্ভোগের অন্ত নেই।জলে কুমির ডাংগায় বাঘ।আজকে দুটি ঘটনা আমাকে ভীষণ ব্যতিত করেছে।নিরবে কিছুক্ষন অশ্রু বিসর্জন দিয়ে মনটাকে হাল্কা করার ব্যর্থ প্রয়াস চালিয়েছি।"

    sentiment.analyze(model_path, vocab_path, text)

