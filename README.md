# My Solution to AGRO CODE Competition

In November 2021 there was a **Machine Learning competition** in Russia.
You can read about it here (in russian, of course):

[https://contest.ds.agro-code.ru/competition](https://contest.ds.agro-code.ru/competition)

The task is to predict the disease(s) of cows by a given textual
description of it in natural language (in russian language).
It is a **Multi-Label Classification** problem, since
there could be several diseases for some texts in the training set.

I registered to participate, but somehow forgot about the _deadline_,
which was 17 of November :)

That is why the solutions I posted were marked as: **"out of competition"**.

There were 47 participants, only 36 of which were *"in competition"*,
others, like me, have missed the deadline.

## Baseline

Organizers provided some baseline, as a python notebook: [Baseline.ipynb](Baseline.ipynb).
The baseline solution is based on the
[**CatBoostClassifier**](https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier).

It is capable to work with texts, treating words as **tokens**.
Though, without analysis of word forms or word meanings.

Baseline also specified some scoring function, based on **log_loss**.
The score of baseline model is: **0.38** (the lower - the better). 

## My Approach

In my opinion, treating words as tokens is very **bad idea**.

Because russian language has a lot of [**grammatical cases**](https://en.wikipedia.org/wiki/Grammatical_case).
It is when the form of the word depends on a context:

> "корова", "коровы", "корове", "коровье", ...

Also because in russian people often use
[**diminutives**](https://en.wikipedia.org/wiki/Diminutive)
and
[**augmentatives**](https://en.wikipedia.org/wiki/Augmentative).
These add some *tiny* meaning to the base meaning of a word,
but *significantly* increase the complexity for Machine Learning.

> "коровушка", "коровёнка", "коровка" <br/>
> "придоил", "додоил", "передоил"

Instead of using tokens I decided to utilize [**word embeddings**](https://en.wikipedia.org/wiki/Word_embedding).
I.e. when each word is represented as a point in N-dimensional space.
Typically, N is: 100...300.

When you have a proper word embeddings, built on a large language corpus,
you can do amazing things with them.

For example, you can use **vector operations** in this space to search for appropriate words.

``v <- (Woman - Man);  King + v ~> Queen``

``v <- (Truck - Car);  Puppy + v ~> Dog``

Another important thing is that words which are often used together,
will have their embeddings located **nearby** in this multidimensional space. 

## Dictionaries of Embeddings for Russian Language

We need to get some pre-trained dictionaries of embeddings for russian words.
And we need it to have free license for commercial usage.

There are some available dictionaries at
[**RusVectores**](https://rusvectores.org/ru/).
But dictionaries there contain 150...300 thousands of words,
which is rather small.
Besides, I could not find any license conditions on their site.

There is [**"Natasha"**](https://github.com/natasha/navec) project.
It contains 250...500k words. License is **MIT**, free for commercial usage.

There is also another project:
[**DeepPavlov**](https://docs.deeppavlov.ai/en/0.0.7/intro/pretrained_vectors.html)
, which contains about 1500k words.
License is **Apache 2.0**, free for commercial usage.

I decided to use the last variant - **DeepPavlov** dictionary.
We need to download the dictionary file, 4.14Gb, and load it into memory.

I wrote a special class for this task: `GloveModel`.
You may find it in [agrocode_embeddings_catboost.ipynb](agrocode_embeddings_catboost.ipynb).

## First Attempt - Transformers

The task of disease labelling is very similar to
[**Sentiment Analysis**](https://en.wikipedia.org/wiki/Sentiment_analysis).
I.e. when you recognize and assign some labels to input text, like:
`Positive`, `Negative` or `Neutral`.

In this case there could be multiple labels for a text,
which is a Multi-Label classification.

### Description

I decided that [**Transformer**](https://en.wikipedia.org/wiki/Transformer_(machine_learning_model))
architecture is suited for this task.
Particularly, its first half - the **TransformerEncoder** part.

The basic idea is the following:

1. The model takes as the input the sequence of words embeddings.
   
2. For each embedding we add information about its position in text,
   using `PositionalEncoding` class.

3. Next comes `TransformerEncoder` with several built-in layers of `MultiHeadAttention`
   followed by `Linear`, `LayerNorm` and `Dropout` layers.
   It processes all the sequence, highlighting some important and shadowing some useless parts of it.

4. Then comes the `MultiHeadAttention` layer, which compares each embedding in the sequence with
   some **Target Embeddings**, it effectively sums up the whole sequence into **several final embeddings**,
   one for each target.

5. Finally, there comes the `Linear` layer, which receives these final embeddings
   and outputs **probabilities**, one for each target. 

The source code for this class is in [agrocode_embeddings_transformer.py](agrocode_embeddings_transformer.py).
Here is the source code of the main `forward` method of the neural net:

```python
    def forward(self, texts: List[str]):
        # Convert batch of texts into tensor of embeddings
        x, padding_mask, batch_offsets = self.texts2batch(texts)
        # x has shape: (sequence_length, batch_size, d_model)
        # padding_mask has shape: (batch_size, sequence_length)
        # batch_offsets is the list of length of batch_size, which contains a list of offsets for each tag
        
        # Add positional information into x
        x = self.position_encoder.forward(x, mask=padding_mask)
        
        # Initialize self-attention mask, so that words could attend only prior words.
        attn_mask = None
        if self.causal_mask:
            attn_mask = th.full((len(x), len(x)), -math.inf, device=x.device, dtype=x.dtype)
            attn_mask = th.triu(attn_mask, diagonal=1)

        x = self.transformer_encoder.forward(x, mask=attn_mask, src_key_padding_mask=padding_mask)
        # x still has shape (sequence_length, batch_size, d_model)
        
        # Combine source embeddings into one embedding, one for each target
        attn_output, attn_weights = self.collect.forward(
            query=self.targets.expand((self.num_targets, x.size(1), self.d_model)),
            key=x,
            value=x,
            key_padding_mask=padding_mask,
            need_weights=True
        )
        # attn_output has the shape: (num_targets, batch_size, d_model)
        # attn_weights has the shape: (batch_size, num_targets, sequence_length)
        
        attn_output = attn_output.permute((1, 0, 2)).reshape(x.size(1), -1)
        # attn_output now has the shape: (batch_size, num_targets * d_model)

        output = th.sigmoid(self.output.forward(attn_output))
        # output has the shape: (batch_size, num_targets)
        
```

The training of this model is typical for all PyTorch project - the simple loop over training epochs.
Training takes from several minutes to one hour at maximum.

### Score

Unfortunately, the result was unsatisfying, score on validation set was about **0.41** (the lower - the better),
which is even worse than in the baseline (0.38)!

### Conclusion

I believe, the poor performance of Transformer architecture
is the **small training dataset**, which has only **294 records**!
Note, that you need to split this dataset into tran and validation parts,
thus you've got small number of examples to learn on.

## Second Attempt - CatBoostClassifier

I decided to try
[**CatBoostClassifier**](https://catboost.ai/en/docs/concepts/python-reference_catboostclassifier),
which is based on
[**Random Forest Algorithm**](https://en.wikipedia.org/wiki/Random_forest).

Still, I wanted to avoid word tokens, and use word embeddings instead.
However, there was some big problem, as CatBoostClassifier was not designed
to get a sequence of embeddings of **arbitrary length** as the input.

After a while, I came up with an idea of how to convert text
of arbitrary length to a set of numbers of a fixed length.
(Disclaimer: I'm sure, this idea was described somewhere long before it came to my mind.
I just never heard about it.)

### Description

The idea is to use some **keyword embeddings** as an anchors,
and measure distance of **each word** of text to all keyword embeddings,
keeping only the **minimal distances** found.

Next the idea is described in details.

Take some keywords or phrases.
In this particular task they could be the **symptoms of different diseases**, like:
> `watery eyes`, `bloody urine`, `fever`, `convulsions`, etc. 

The more various keywords you get - the better.
But also it will require more time to converge.

We need to compute their respective **embeddings**.
When a keyword is a single word - it is merely its embedding.
But when you have a phrase - you need to combine several embeddings
of each word of the phrase into one embedding.

There are different ways to do it.
I used the **mean** operation in this project.
Some papers say that using **addition** also works fine.

These **keyword embeddings** now become like **beacons** or **anchors** in the multidimensional embeddings space.
They will help us to convert a text of *arbitrary length* to a *fixed set of distances* to these anchors.

1. Each text we divide into **tokens** - words and punctuation symbols.

2. For each token we take its **embedding**.
   For missing words - a special embedding 'unk'.
   For numbers - a special embedding 'num'.

3. For each token's embedding we compute the **distances** to all anchor embeddings.
   Euclidean distance may not be the perfect solution for multidimensional space.
   That is why we utilize 4 different distance functions: ``(cosine, cityblock, euclidean, braycurtis)``

4. While processing all tokens, we will keep only the **minimal distance** to each anchor embedding.

5. This way, after processing we get a set of minimal distances to each anchor embedding,
   in particular: 4 minimal distances due to using of four different distance functions.
   After all, we have multiple new features: ``num_features = num_anchors * num_distance_functions``.

For this project I used **45** keyword phrases, which produced **180** numerical features
for each text in training dataset.
Note, that random forest algorithm does NOT need normalized features.

> This approach has its limitations. It can not correctly understand negation. Example:
> 
> The owner says: **"My cow has no bleeding, only diarrhea"**
>
> The model sees the word: **bleeding** - it may consider a cow has an injury!
> Because model can not understand the phrase 'has no bleeding'.

To build the classifier I used the same approach as in the Baseline.

The model is based on `sklearn.multiclass.OneVsRestClassifier` from scikit package,
which uses multiple `catboost.CatBoostClassifier` instances to predict each disease probability independently.

```python
from catboost import CatBoostClassifier
from sklearn.multiclass import OneVsRestClassifier

RANDOM_SEED = 2021

estimator = CatBoostClassifier(
    max_depth=12,
    iterations=1000,
    verbose=False,
    allow_writing_files=False,
    random_seed=RANDOM_SEED,
)

model = OneVsRestClassifier(estimator=estimator)
log.info(f'Initialized model: {model}')

log.info(f'Starting training...')
model.fit(X_train, y_train)
log.info('Done')
```

### Score

The score on validation set was about **0.03** (the lower - the better),
which is far better than in the baseline (0.38)!

### Conclusion

This approach seem to work fine.
It does not treat words as a set of unique tokens.
Instead, it relies on the semantic meaning of a word, which is encoded in its embedding.

Thus, model can (hopefully) manage to work with texts and words, **which it has never seen before**.
Simply because the embeddings of the words in new texts will, somehow,
be related to the anchor embeddings, which model uses.

## Final Thoughts

I can hardly ever beat *huge and heavily trained models*, which gain the highest scores in the leaderboard.
I believe, though, my idea is right and my model is robust to the new unexpected texts.

My thoughts about huge, highly overfitted models, which gain top scores in competitions,
are best said by Michał Marcinkiewicz in his article:
["The Real World is not a Kaggle Competition"](https://www.netguru.com/codestories/real-world-is-not-a-kaggle-competition).

The approach I used has its limitations.
Like said before, it can not correctly understand negation or any complex relations between words.
A Transformer architecture is capable of doing so,
but it requires much more training data than was given in this competition.

Working on natural language processing is fun!

:)
