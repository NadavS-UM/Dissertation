import pandas as pd
import transformers
import torch 
import numpy as np
import matplotlib.pyplot as plt


#for Hebrew speakers

df = pd.read_csv('data from Hebrew NSs.csv')


correct = df['AUTOMATIC'] == df['MANUAL']

tokeniser = transformers.GPT2Tokenizer.from_pretrained('gpt2')
gpt = transformers.GPT2LMHeadModel.from_pretrained('gpt2')

sent_log_probs = []
for text in df['AUTOMATIC']:

    token_indexes = tokeniser(text, return_tensors="pt")

    logits = gpt(token_indexes['input_ids']).logits
    logprobs = torch.log_softmax(logits, dim=2)
    sent_logprob = 0.0
    for (i, token_index) in enumerate(token_indexes['input_ids'][0]):
        sent_logprob += logprobs[0, i, token_index].tolist()
    sent_log_probs.append(sent_logprob)


points = np.array(sent_log_probs)

points1 = points[correct == 1]
points2 = points[correct == 0]

plt.plot(points1, color='black', linestyle='', marker='o')
plt.plot(points2, color='red', linestyle='', marker='o')
plt.savefig('plot_Hebrew_NS.pdf')


#for English speakers

df2 = pd.read_csv('data from English NSs.csv')


correct = df2['AUTOMATIC'] == df2['MANUAL']

tokeniser = transformers.GPT2Tokenizer.from_pretrained('gpt2')
gpt = transformers.GPT2LMHeadModel.from_pretrained('gpt2')

sent_log_probs = []
for text in df2['AUTOMATIC']:

    token_indexes = tokeniser(text, return_tensors="pt")

    logits = gpt(token_indexes['input_ids']).logits
    logprobs = torch.log_softmax(logits, dim=2)
    sent_logprob = 0.0
    for (i, token_index) in enumerate(token_indexes['input_ids'][0]):
        sent_logprob += logprobs[0, i, token_index].tolist()
    sent_log_probs.append(sent_logprob)


points = np.array(sent_log_probs)

points3 = points[correct == 1]
points4 = points[correct == 0]

plt.plot(points3, color='black', linestyle='', marker='o')
plt.plot(points4, color='red', linestyle='', marker='o')
plt.savefig('plot_English_NS.pdf')




#log prob for Hebrew speakers

df2 = pd.read_csv('data from Hebrew NSs.csv')
del df2["MANUAL"]
df2['LOG PROB'] = points
df2['AUTOMATIC'] = df2['AUTOMATIC'].str.rstrip('\xa0')
df2['IS_CORRECT'] = correct
df2.to_csv('log_prob_Hebrew.csv', encoding='utf-8')


#log prob for English speakers

df2 = pd.read_csv('data from English NSs.csv')
del df2["MANUAL"]
df2['LOG PROB'] = points
df2['AUTOMATIC'] = df2['AUTOMATIC'].str.rstrip('\xa0')
df2['IS_CORRECT'] = correct
df2.to_csv('log_prob_English.csv', encoding='utf-8')


#violin figure for Hebrew speakers

(fig, axs) = plt.subplots(1, 2)
axs[0].violinplot(points1, showmedians=True)
axs[0].set_ylim(-180, 0)
axs[1].violinplot(points2, showmedians=True)
axs[1].set_ylim(-180, 0)
axs[0].set_title("Correct Sentences")
axs[1].set_title("Incorrect Sentences")
axs[0].set_xticklabels([])
axs[1].set_xticklabels([])
axs[0].set_xticks([])
axs[1].set_xticks([])
plt.subplots_adjust(wspace=0.5)
axs[0].grid()
axs[1].grid()

plt.show()
plt.savefig('violin_plot_Hebrew.pdf')