import numpy as np
from fontTools.ttLib.tables.S__i_l_f import Pass


class NaiveBayesForSpam:
    def train(self, ham_messages, spam_messages):
        self.words = list(set(" ".join(ham_messages + spam_messages).lower().split()))
        self.priors = np.zeros(2)
        self.priors[0] = float(len(ham_messages)) / (len(ham_messages) + len(spam_messages))
        self.priors[1] = 1.0 - self.priors[0]

        self.likelihoods = []
        for w in self.words:
            prob_ham = (1.0 + len([m for m in ham_messages if w in m.lower()])) / len(ham_messages)
            prob_spam = (1.0 + len([m for m in spam_messages if w in m.lower()])) / len(spam_messages)
            self.likelihoods.append([min(prob_ham, 0.95), min(prob_spam, 0.95)])

        # shape becomes (2, num_words):
        # row 0 = P(word|ham), row 1 = P(word|spam)
        self.likelihoods = np.array(self.likelihoods).T
        Pass

    def predict(self, message):
        log_posteriors = np.log(self.priors)
        sample_words = message.lower()

        for i, w in enumerate(self.words):
            if w in sample_words:
                log_posteriors += np.log(self.likelihoods[:, i])
            else:
                log_posteriors += np.log(1 - self.likelihoods[:, i])

        max_class_index = np.argmax(log_posteriors)
        return "ham" if max_class_index == 0 else "spam"


def main():
    ham_messages = [
        "see you at lunch",
        "project meeting tomorrow",
        "can we reschedule the call",
        "thanks for your help",
    ]
    spam_messages = [
        "win cash now",
        "claim your free prize",
        "limited offer buy now",
        "earn money fast",
    ]

    model = NaiveBayesForSpam()
    model.train(ham_messages, spam_messages)

    print("Words:", model.words)
    print("Likelihoods shape:", model.likelihoods.shape)
    print("\nPer-word [P(word|ham), P(word|spam)]:")
    for i, w in enumerate(model.words):
        pair = model.likelihoods[:, i]
        print(f"  {w:12s} -> {pair.tolist()}")

    sample = "free cash offer"
    print(f"\nPrediction for '{sample}':", model.predict(sample))


if __name__ == "__main__":
    main()
