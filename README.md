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

I also removed the `tags` field. This field was a bit trickier to decide upon, but upon reflection on the metrics calculated later on, the `tags` field was doing more harm than good. [See Table 2 in the retrieval phase for more information](#metrics). This left me with 3 distinct fields to work with: `name`, `ingredients`, and `steps`.

To preprocess this large corporus of 231,637 recipes, I chose to use [spaCy](https://spacy.io/)'s library to perform fast lemmatisation, processing 1000-batch documents. The actual lemmatisation was simple - lowercasing, removing punctuation, whitespace, and non-alphanumeric characters. I used the `en_core_web_sm` pipeline for the list of generic stopwords in this initial preprocessing step.

Especially when dealing with recipes, unigrams are not enough. Words that can lack meaning by themselves, such as `slow` and `cooker`, carry more meaning when working as a multi-word expression, such as the bigram `slow_cooker`. To train this bigram model, I used [Gensim](https://radimrehurek.com/gensim/)'s [Phrases](https://radimrehurek.com/gensim/models/phrases.html), which calculates a normalised pointwise mutual information (PMI) score, essentially checking how independent tokens are. In simple terms, if a token occurs with another token frequently enough, it is considered to be a bigram.

Finally, I included a title boost of 3, by repeating the title thrice before concatenating the description and steps to the recipe. This is to emphasise the short titles, whose meaning can be diluted in recipes with longer `ingredients` or `steps`.

## Document Embeddings
I embedded each recipe $d_i$ as a TF-IDF vector $\vec{d_i}$ using the [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) from [scikit-learn](https://scikit-learn.org/stable/). This allows you to add a list of your own stopwords, and I decided to create an empirical list of stopwords, based on the low SNR words that made it through to this stage from the previous preprocessing - words such as `depend_size`, `sheet`, and `easily_pierce`. These are words which appear enough to have meaning, but from a human standpoint, only add noise to our recipe vectors. I discovered these words by looking at many random recipes, and outputting highly weighted words, picking out ones that don't carry the information a user of the system is likely to search for. Given more time, perhaps this process could also be automated.

Within the Vectorizer itself, I used smoothing to prevent division by zero or common words having zero weight, as well as L2-normalisation to eliminate document length bias, and sublinear term frequency scaling to prevent highly frequent words from dominating the recipe's vector representation. Finally, a minimum document frequency of 0.01% was used alongside a maximum document frequency of 40%. This means that words must appear in at least ~23 documents to be considered an actual token, and if they appear in more than ~92655 documents, they are considered as noise. This is all to aid in having the most distinguishing words for each recipe.

Here is an example output for of the top 20 words for a recipe, with their TF-IDF values:
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

## Retrieval
To retrieve relevant recipes given a query, we perform a nearest neighbours search between the query and our set of recipes, both vectorised. The similarity operator used is cosine similarity

$$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\lVert u \rVert \lVert v \rVert }$$

which measures the angle between the query vector and the recipe vector, irrespective of their lengths. This operator provides magnitude invariance, making it a suitable operator for comparing short queries to long recipes.
N.B. queries are preprocessed the same as recipes.

### Thresholding
To determine which recipes are relevant, and how many to return, I chose to use a relative score threshold in combination with an absolute similarity floor.

For example, given a relative score threshold of 80% and an absolute similarity floor of 0.1, recipes will be accepted if they have a similarity score that is at least 80% of the most similar recipe, as long as this score is also greater than 0.1. This outperforms initial ideas I had, which were a simple similarity floor, $\text{top-}k$ recipes, and $\text{top-}p$% of recipes.

A similarity floor fails if all recipes retrieved are below the arbitrarily set floor despite being relevant.

$\text{Top-}k$ recipes is an improvement upon this, but requires selecting a fixed number of recipes to always output, which can confuse the LM as wildly irrelevant recipes could be retrieved.

$\text{Top-}p$% of recipes is also another potential improvement, but this time the problem lies in low relevance recipes. If all recipes retrieved have a low similarity, e.g. 0.05, we will regardless retrieve a substantial portion of low similarity recipes.

My relative score threshold in combination with the similarity floor combines the above similarity floor with the $\text{top-}p$% method, mitigating the downsides of each, as we retrieve documents that are relevant based on how relevant the most relevant document is, similar to rescaling our domain, but we do not venture into similarities that are too low to be considered.

The relative score threshold, $\alpha$, and the similarity floor, cannot be set arbitrarily, as stated above. Instead, I tuned these hyperparamters, using the a grid search with the Macro-F1 score of retrieved recipes as a criterion to maximise. The results of this can be seen in Table 1, alongside a heatmap visualising the optimisation surface:

<img width="562" height="393" alt="image" src="https://github.com/user-attachments/assets/45ef2d59-54d5-4438-9379-e7e3583a4824" />

<img width="741" height="556" alt="image" src="https://github.com/user-attachments/assets/9a2a0093-7393-4f4d-96ab-2e87e2c1646b" />


The Macro-F1 score peaked with a floor of 0.25 and an $\alpha$ of 0.59, and so I justifiably set the hyperparamters.

### Metrics
I then calculated the macro- and micro-average precision, recall, F1 for the entire set of queries (provided as tests), as well as the Mean Average Precision (MAP), both with and without tags included in the preprocessing step of the recipes. These metrics can be seen below in Table 2:

<img width="1068" height="170" alt="image" src="https://github.com/user-attachments/assets/a1242c1c-4932-42d1-b612-5c81fbcaa5f0" />

We notice here that excluding `tags` from the recipes outperforms their inclusion, most clearly demonstrated in the micro-average metrics as well as the MAP, providing sufficient justification for excluding the `tags` field from the recipes.

### Retrieved Recipes
Here are a couple of examples of recipes that are retrieved given the query:
```
Found 46 recipes for 'cajun style gumbo with an easy roux':
--------------------------------------------------
1.  [ID: 186726] simple chicken gumbo
    Ingredients: chicken, onions, vegetable oil, flour, tap water, salt, pepper, white rice
    Steps: heat oil in large pot and add flour to make a roux, make the roux as dark as you can without burning it , or yah yah in cajun country, add the chopped onions when roux is ready , stir for about a minute , then add the water, add chicken , salt and pepper , bring to a boil , and simmer until chicken is very tender, serve in a bowl over cooked white rice
    Similarity Score: 0.6124
    (Ratio to best: 1.0000)
------------------------------
2.  [ID: 76648] easy gumbo
    Ingredients: flour, vegetable oil, chicken broth, rotisserie-cooked chicken, celery, carrots, white onion, tabasco sauce, cayenne pepper, dried basil leaves
    Steps: to make roux: combine flour and vegetable oil in a medium to large saucepan on low heat, stirring frequently , allow flour / oil combination cook until medium brown, this could take up to 2 hours, while waiting for the roux to cook , pour chicken broth into a large pot and heat to a boil, cut or tear all of the meat from the rotisserie chicken into bite size pieces and place it into the boiling chicken broth, add celery , carrots , onion , tabasco sauce , cayenne pepper , add basil leaves to chicken and broth, allow soup mixture to boil for about 1 / 2 an hour stirring frequently, lower heat of soup, once roux has reached a brown color , combine the roux with soup to make gumbo !, bring gumbo to a boil for 10 minutes, lower heat, serve over rice
    Similarity Score: 0.5183
    (Ratio to best: 0.8464)
------------------------------
3.  [ID: 33841] cajun style chicken gumbo
    Ingredients: boneless skinless chicken breast, cajun seasoning, dried thyme leaves, canola oil, onion, green bell pepper, carrot, celery, garlic cloves, all-purpose flour, reduced-sodium chicken broth, no-salt-added stewed tomatoes, hot pepper sauce, cooked rice, fresh parsley
    Steps: cut chicken into 1 inch pieces, place in medium bowl, sprinkle with seasoning and thyme, toss well, set aside, heat oil in large sauce pan over medium-high heat, add onion , bell pepper , carrots , celery and garlic to saucepan, cover and cook 10 minutes or until vegetable are crisp-tender , stirring once, add chicken, cook 3 minutes , stirring occasionally, sprinkle mixture with flour , cook 1 minute , stirring frequently, add chicken broth , tomatoes and pepper sauce, bring to a boil over high heat, reduce heat to medium, simmer , uncovered , 10 minutes or until chicken is no longer pink in center , vegetables are tender and sauces is slightly thickened, ladle gumbo into 4 shallow bowls , top each with a scoop of rice sprinkle with parsley and serve with additional pepper sauce , if desired
    Similarity Score: 0.5114
    (Ratio to best: 0.8352)
------------------------------
```
```
Found 125 recipes for 'vegetarian lasagna but easy':
--------------------------------------------------
1.  [ID: 168141] quick easy vegetarian lasagna
    Ingredients: soft tofu, parmesan cheese, eggs, garlic cloves, fresh basil, salt, fresh ground pepper, tomato and basil pasta sauce, no-boil lasagna noodles, baby spinach, low fat mozzarella
    Steps: heat oven to 375f, combine tofu , parmesan , eggs , garlic , herbs , salt and pepper in a medium bowl, lightly coat a 13x9-inch baking dish with vegetable cooking spray, spread about 1 cup of tomato sauce at the bottom of the pan, arrange one layer of noodles on top, spread with half the tofu mixture, top with half of the spinach , a third of the remaining sauce , and a third of the mozzarella, repeat layers , ending with noodles , sauce , and mozzarella, cover with aluminum foil and bake 30-35 minutes, let stand 5 minutes before cutting
    Similarity Score: 0.5577
    (Ratio to best: 1.0000)
------------------------------
2.  [ID: 144569] boil cheesy lasagna vegetarian optional meat sauce
    Ingredients: lasagna noodles, ricotta cheese, parmesan cheese, eggs, pasta sauce, mozzarella cheese
    Steps: preheat oven to 350f, combine ricotta , parmesan , and eggs and mix well, in a 9x13 dish , spread about 1 / 3 of the sauce, layer with half each of the uncooked lasagna noodles , ricotta mixture , remaining pasta sauce , and mozarella, repeat layering, cover tightly with alumninum foil and bake covered for 45 minutes, uncover and bake an additional 15 minutes, let stand 10 minutes before serving
    Similarity Score: 0.5050
    (Ratio to best: 0.9055)
------------------------------
3.  [ID: 222156] vegetarian lasagna
    Ingredients: spaghetti sauce, carrot, oregano, cooked lasagna noodles, ricotta cheese, frozen chopped spinach, eggs, zucchini, fresh mushrooms, part-skim mozzarella cheese, parmesan cheese
    Steps: mix carrots , oregano , and spaghetti sauce together, mix ricotta , spinach , and eggs together in separate bowl, spread cup spaghetti sauce in bottom of 9 x 13 inch baking dish, layer 3 lasagna noodles , remaining sauce , ricotta mixture , sliced zucchini , sliced mushrooms , mozzarella , and parmesan, repeat layers with remaining ingredients, bake in 350 degrees oven for about 45 minutes
    Similarity Score: 0.4934
    (Ratio to best: 0.8848)
------------------------------
```
## TF-IDF Drawbacks
Embedding documents as TF-IDF vectors does have its drawbacks.

For example, TF-IDF embeddings have no semantics knowledge e.g. not understanding what a stone fruit is:

<img width="1461" height="259" alt="image" src="https://github.com/user-attachments/assets/4ad17fbc-a475-4ad7-81f0-dd942ce0569f" />

<img width="1130" height="282" alt="image" src="https://github.com/user-attachments/assets/826c8125-74a7-4752-91ed-59e310f84c9f" />

There is also no concept of negation when using TF-IDF embeddings, requiring roundabout queries to achieve search goals:

<img width="1363" height="287" alt="image" src="https://github.com/user-attachments/assets/5b0bce60-848e-49f2-823c-d3df2c059923" />

<img width="1488" height="245" alt="image" src="https://github.com/user-attachments/assets/1c1a0b6d-c468-46da-a9b4-2b5e8061f30e" />

We later pivot to [neural embeddings](#neural-embeddings) in an attempt to counteract these drawbacks.

## Prompt Engineering
For this project, I chose [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) as my LM, due to its great performance despite being lightweight and easy to run on Colab GPU. Since LMs have a limited context window, I made the decision to remove `ingredients`, and only retain `name` and `steps`. This did not significantly impact performance, likely because ingredients are usually all mentioned in `steps`.

Using the following prompt template:
```python
# Context construction using recipe fields: 'name' and 'steps'
context = "\n\n".join([
    f"Name: {r['name']}\nSteps: {r['steps']}"
    for r in retrieved_recipes
])

messages = [
    {
        "role": "user",
        "content": f"""You are a helpful sous-chef. Based on these recipes, help the user with their request, adhering to the following rules:
1. Rely STRICTLY on the supplied recipes. If the context is insufficient or irrelevant, state this clearly and DO NOT attempt to fulfill the request using outside knowledge.
2. Always provide a clear, step-by-step recipe, clearly reasoning across supplied recipes when necessary.

Recipes:
{context}

User Request: {query}"""
    }
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```
the LM was able to reason across recipes, and determine when it could not answer a request, evidenced by answers such as:
```
"an easier roux making process compared to the traditional method that involves waiting up to 2 hours … rather than waiting for the flour to brown on its own … making the process more efficient."
"there isn't a specific meatball recipe … that matches any of the provided recipes … since Bertha's Meatballs can be made and simmered in spaghetti sauce, this recipe will produce a Spaghetti and Meatballs dish.”
```
Full results can be found [here]().

# Part 2: ACL Papers
In this section, we deal with a subset of all the Natural Language Processing (NLP) papers from the ACL Anthology, which was scraped and parsed by [Rohatgi et al. (2023)](https://aclanthology.org/2023.emnlp-main.640/).

These technical papers have a much richer word vocabulary, partly due to being much longer than the recipes from Part 1, and also due to jargon and proper names. For this section, we will focus on using neural document embeddings as opposed to a vector space model.

## Architecture
Once again, we can break this system down into its building blocks, visualising with a flowchart:

<img width="1399" height="682" alt="image" src="https://github.com/user-attachments/assets/062b53b4-d5bc-4550-ab9a-32d227947472" />

The general flow is:
1. Papers are chunked, vectorised, and have the same metrics calculated on them as before (done once for comparison with neural)
2. User sends a query to the system, which is:
   
   * Passed to the various prompt template to form the final prompts sent to the LM
   
   * Embedded by the neural model, and used for Nearest Neighbour search with the embedded paper chunks, still thresholding as before
   
3. Prompt template formats the query and the context (relevant paper chunks) according to the prompt syntax of the LM, which returns an informed response

We can now look at each of these components individually.

## Vector vs Neural Embeddings
