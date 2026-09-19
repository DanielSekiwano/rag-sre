# Retrieval-Augmented Generation Search and Reasoning Engine
> [!NOTE]
> Source code is withheld to comply with academic project guidelines at the [**KU Leuven**](https://www.kuleuven.be/english/kuleuven). This repository functions solely as a technical showcase detailing architecture, implementation strategy and results.
> 
> This is a written report on my RAG project completed at [KU Leuven](https://www.kuleuven.be/english/kuleuven). We were supplied the datasets for both parts, as well as some boilerplate code to e.g. load in these datasets etc. The methodology (processing pipeline, architecture etc) was decided upon by myself.

# Part 1: Recipe Reasoning
The following synopsis was provided by KU Leuven:

*You are given the cooking recipes dataset introduced by Majumder et al. (2019). This dataset contains over 180 000 recipes and over 700 000 recipe reviews covering 18 years of user interactions and uploads on food.com. We are only interested in the recipes subset for this project. We have filtered the dataset such that it contains only the relevant metadata for your implementation, and processed it for easy integration with the datasets package. For each recipe in the dataset, the following information is provided:*
| Field | Description |
| ---: | :--- |
| **`name`** | the name of the recipes as originally posted (e.g. “baked winter squash mexican style”) |
| **`tags`** | the user-provided tags associated with the recipe (e.g. “60-minutes-or-less”) |
| **`description`** | a story the user has attached to the recipe |
| **`ingredients`** | the list of ingredients required for making the recipe |
| **`steps`** | the step-by-step instructions for making the recipe, in order |

*You are tasked with constructing a RAG pipeline with word-level IR that answers queries about this dataset.*

## Architecture
We can break this pipeline down into a set of high-level steps, visualised below as a flowchart:

<img width="1473" height="446" alt="image" src="https://github.com/user-attachments/assets/df5b1194-e27a-4549-bcff-953f4a6fe1f5" />

The general flow is:
1. Recipes are preprocessed, vectorised, and stored by the system
2. User sends a query to the system, which is:
   
   * Passed to the prompt template to form the final prompt sent to the language model (LM)
   
   * Preprocessed, vectorised, and used for Nearest Neighbour search with the vectorised recipes, thresholding to decide which recipes to pass as context to the prompt template and eventually the language model
   
3. Prompt template formats the query and the context according to the prompt syntax of the LM, which returns an informed response

We can now look at each of these components individually.

## Preprocessing
Firstly, I had to decide which fields carried a high signal-to-noise ratio (SNR), thus making them relevant for the vectorisation. For the purposes of illustration, below I have pasted an example recipe:
```
'name' = penne gorgonzola chicken roma tomatoes

'ingredients' = boneless skinless chicken breasts, penne pasta, butter, heavy cream, gorgonzola, parmesan cheese, nutmeg, roma tomatoes

'steps' = grill or broil chicken breasts, slice into thin strips, cook pasta according to package directions, toss with olive oil , return to pot, while pasta and chicken are cooking , melt butter in medium saucepan , heat to boiling , stirring constantly , boil 1 minute, slowly add heavy cream , heat to a boil to reduce , stirring constantly, boil for 2 minutes, reduce heat , add gorgonzola , parmesan and nutmeg, stir until melted, serve sauce over noodles , top with chicken and chopped tomatoes

'tags' = 30-minutes-or-less, time-to-make, course, main-ingredient, cuisine, preparation, occasion, main-dish, pasta, poultry, european, dinner-party, holiday-event, kid-friendly, romantic, italian, chicken, dietary, copycat, comfort-food, valentines-day, meat, pasta-rice-and-grains, penne, novelty, taste-mood

'description' = although i usually post healthy recipes, this one does not fall into that category! it is rich, creamy and decadent. if you like cream sauces, watch out!! this is a recipe my husband developed based on a dish at an italian restaurant. i think his is just as good.

'official_id' = 157316
```

At first glance, we can see that the `official_id` and `description` fields carry a low SNR, as the former is an arbitrary ID, and the latter contains a large amount of fluff. It is particularly important to exclude `description` here, as this fluff is harmful in a vector model, where every word contributes to the recipe's final vector. Increasing the number of irrelevant words, such as `husband` in a recipe decreases the importance of words such as `grill` or `chicken`, which we are more likely to be interested in.

I also removed the `tags` field. This field was a bit trickier to decide upon, but upon reflection on the metrics calculated later on, the `tags` field was doing more harm than good. [See Table 2 in the retrieval phase for more information](#retrieval). This left me with 3 distinct fields to work with: `name`, `ingredients`, and `steps`.

To preprocess this large corporus of 231,637 recipes, I chose to use [spaCy](https://spacy.io/)'s library to perform fast lemmatisation, processing 1000-batch documents. The actual lemmatisation was simple - lowercasing, removing punctuation, whitespace, and non-alphanumeric characters. I used the `en_core_web_sm` pipeline for the list of generic stopwords in this initial preprocessing step.

Especially when dealing with recipes, unigrams are not enough. Words that can lack meaning by themselves, such as `slow` and `cooker`, carry more meaning when working as a multi-word expression, such as the bigram `slow_cooker`. To train this bigram model, I used [Gensim](https://radimrehurek.com/gensim/)'s [Phrases](https://radimrehurek.com/gensim/models/phrases.html), which calculates a normalised pointwise mutual information (PMI) score, essentially checking how independent tokens are. In simple terms, if a token occurs with another token frequently enough, it is considered to be a bigram.

Finally, I included a title boost of 3, by repeating the title thrice before concatenating the description and steps to the recipe. This is to emphasise the short titles, whose meaning can be diluted in recipes with longer `ingredients` or `steps`.

## Document Embeddings
I embedded each recipe $d_i$ as a TF-IDF vector $\vec{d_i}$ using the [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) from [scikit-learn](https://scikit-learn.org/stable/). This allows you to add a list of your own stopwords, and I decided to create an empirical list of stopwords, based on the low SNR words that made it through to this stage from the previous preprocessing - words such as `depend_size`, `sheet`, and `easily_pierce`. These are words which appear enough to have meaning, but from a human standpoint, only add noise to our recipe vectors. I discovered these words by looking at many random recipes, and outputting highly weighted words, picking out ones that don't carry the information a user of the system is likely to search for. Given more time, perhaps this process could also be automated.

Within the Vectorizer itself, I used smoothing to prevent division by zero or common words having zero weight, as well as L2-normalisation to eliminate document length bias, and sublinear term frequency scaling to prevent highly frequent words from dominating the recipe's vector representation. Finally, a minimum document frequency of 0.01% was used alongside a maximum document frequency of 40%. This means that words must appear in at least ~23 documents to be considered an actual token, and if they appear in more than ~92655 documents, they are considered as noise. This is all to aid in having the most distinguishing words for each recipe.

Here is an example output for of the top 20 words for a recipe, with their TFIDF values:
```
Recipe: kitchen chili:
kitchen        0.40
chili          0.30
kidney         0.30
lettuce        0.25
bean           0.24
beef           0.23
rotel          0.21
tomato         0.21
sautee         0.21
ground         0.19
diced          0.18
wilt           0.17
slow_cooker    0.16
shred          0.15
clean          0.15
yellow         0.13
paste          0.13
like           0.12
cumin          0.12
cold           0.12
```
