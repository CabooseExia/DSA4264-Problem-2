# Technical Report

**Project: DSA4264 Problem 2**  
**Members: Bae Soo Youn, Ethan Loh Zhi Kai, Muhammad Irfan Bin Salleh, Wee Wei Kit Glenn, Wong Si Yuan**  
Last updated on 13/11/2024

## Section 1: Context
This project originated in response to the increasing prevalence of harmful content on social media, as highlighted by the Ministry of Digital Development and Information (MDDI)’s 2024 Online Safety Poll. The survey, conducted with 2,098 Singapore respondents aged 15 and above, found that 74% of respondents encountered harmful online content in 2024, marking a significant rise from 65% in 2023. Specifically, two-thirds (66%) encountered harmful content on designated social media services, up from 57% in the previous year.

This harmful content frequently incited racial or religious tension, with a substantial portion of it appearing on widely used platforms like Facebook (60%) and Instagram (45%). Despite these encounters, 61% of users took no action, indicating a gap in proactive responses against such content. Given this growing trend and the potential impact on social harmony, MDDI saw the need for a rigorous study to analyze the rise in toxicity and hatefulness in online discourse, particularly on Singapore-related social media channels, to inform potential policy interventions and strengthen online safety initiatives.

Thus, this project was established to quantify the rise in toxic content, identify its primary drivers, and provide actionable recommendations for mitigating harm on social media.

## Section 2: Scope

### 2.1 Problem

**Problem Statement**

The Ministry of Digital Development and Information’s (MDDI) Online Trust and Safety department is increasingly concerned with the rise in toxic and hateful content on social media. A recent poll highlights this trend, showing a jump from 57% to 66% of respondents encountering harmful content within a year. This toxicity problem has significant implications for Singapore, particularly as it affects children and increases social polarization. The challenge is understanding how hatefulness and toxicity have evolved on these platforms and identifying key drivers, which is crucial for formulating effective policy interventions.

**Affected Stakeholders**

MDDI’s Online Trust and Safety department is directly impacted, as they need comprehensive insights to implement policy changes aimed at reducing online harm. Other stakeholders include major social media platforms (e.g., Meta, Google, TikTok), responsible for implementing content moderation, and other government agencies, such as the Ministry of Community, Culture, and Youth, which share a vested interest in mitigating the societal effects of online hate speech and toxicity.

**Significance of the Problem**

Unchecked proliferation of harmful content risks deepening societal divides in Singapore’s multicultural society and poses threats to vulnerable populations, particularly children. The rise in toxic discourse complicates efforts to maintain a cohesive society and could lead to increased social tensions and disengagement.

**Why Data Science is the Solution**

Analyzing all online comments manually is infeasible given the volume of data. Data science, specifically Natural Language Processing (NLP), offers computational solutions for tracking toxicity and hatefulness across extensive datasets. By leveraging machine learning and NLP, MDDI can identify the level, drivers, and patterns of toxicity over time. This provides a scalable, objective foundation for understanding the trends and for recommending targeted interventions and collaborations with social media platforms.

### 2.2 Success Criteria

**Business Goals**

***Actionable Insights for Policy Recommendations***

Success will be measured by the ability to generate clear, data-driven insights that highlight the level and sources of hatefulness and toxicity on Reddit over time. Specifically, this project should identify the key drivers behind rising toxicity, allowing MDDI to craft targeted policy recommendations for social media platforms, aimed at reducing harmful content.

***Recommendations for Social Media Moderation***

The project should deliver specific, implementable recommendations for social media companies to curb toxic content. These recommendations will be evaluated based on their relevance to the trends and drivers identified, and the extent to which they offer actionable guidance for moderation policies on social media.

**Operational Goals**

***Detailed Analysis of Toxic Content Drivers***

Success will also be assessed on the project’s effectiveness in breaking down factors contributing to toxicity. This includes identifying topics, themes, or events correlated with spikes in toxic content, offering a granular understanding of what triggers hatefulness and toxicity.

***Efficient and Scalable Analysis Model***

Develop a scalable model for analyzing large volumes of Reddit data on an ongoing basis. The success of this model will be evaluated by its reliability and ability to process data efficiently, enabling regular updates to inform real-time policy interventions if required.

### 2.3 Assumptions

**Expertise in Recognizing Hateful and Toxic Language** 

We assume that our team has a solid grasp of toxicity patterns in Singaporean online discourse, enabling accurate interpretation of culturally nuanced content. To mitigate subjectivity, we selected validated models with strong benchmarks in toxicity detection, ensuring objectivity in labeling.

**Adequate Model Generalization** 

The chosen model, `sileod/deberta-v3-base-tasksource-toxicity`, is assumed to generalize well to our dataset, despite a slightly lower F1-score compared to `gpt-4o-mini`. We expect its accuracy to be sufficient for detecting toxic and hateful content in Singapore's context, with ongoing monitoring for any potential labeling drift.

**Interpretation of Moderation Indicators:** 

Without full access to Reddit’s moderation data, we assume that `[deleted]` or `[removed]` tags do not always indicate toxic content. These removals may relate to subreddit rules or other non-harmful reasons, allowing us to focus toxicity analysis on available, interpretable data.

**Alignment with MDDI Definitions:** 

We assume that our models' interpretations of hatefulness and toxicity align with MDDI’s standards. This requires our models to accurately label culturally specific language within the scope of Singaporean discourse, supporting consistent and relevant insights for policy recommendations.

## Section 3: Methodology
Our project methodology has 4 phases as shown in the table below:
| Phase                           | Description                                                                                                     | Purpose                                                                                                         |
|---------------------------------|-----------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|
| **Phase 1: Identifying Hate and Toxicity**         | Label 5 million rows of Reddit comments as either hateful/toxic or non-hateful/non-toxic.                      | Determine the baseline of hate and toxicity in the dataset.                                                   |
| **Phase 2: Community & User Behaviour Analysis**   | Analyze community behaviors and identify patterns in hate and toxicity across subreddits.                     | 1) Identify the most hateful and toxic subreddit. <br> 2) Analyze popular posts in each subreddit. <br> 3) Identify posts that generate the most hate and toxicity. <br> 4) Detect potential super-spreaders of hate and toxicity. |
| **Phase 3: Trend & Driver Analysis**               | Analyze trends and drivers of hate and toxicity over time.                                                     | 1) Determine if there is an increasing trend in hate and toxicity. <br> 2) Identify topics with rising trends in hate and toxicity. <br> 3) Pinpoint drivers for the increase in hate and toxicity per topic. |
| **Phase 4: Non-hateful/toxic Trigger Analysis**    | Investigate triggers in non-hateful/non-toxic comments that lead to hate and toxicity responses.              | 1) Detect non-hateful & non-toxic comments that may trigger hate and toxicity. <br> 2) Identify words/phrases in non-hateful & non-toxic comments that incite or trigger toxic responses. |

In the subsections below, if "`Phase Specific`" is indicated beside the subsection title, it means that the methods for that subsection vary by phase.

### 3.1 Technical Assumptions
This section outlines the foundational assumptions guiding model development, resource allocation, and hypotheses for our data science project.

#### 3.1.1 Definitions of Hate and Toxicity

**Toxicity** includes offensive or aggressive language that may not target anyone specifically but still disrupts positive interaction.

**Hate** refers to language that targets specific groups with hostility or prejudice.

**Unified "Harmful Content" Label**: Both toxic and hateful content are grouped under "Harmful Content" for simplicity, aligning with MDDI's goal of reducing harmful online discourse. This unified approach supports consistent detection, insights, and policy recommendations.

#### 3.1.2 Data Features
Available dataset features include:
| Feature         | Description                                                                                                     |
|-----------------|-----------------------------------------------------------------------------------------------------------------|
| **text**        | The primary Reddit comment content for toxicity analysis.                                                       |
| **timestamp**   | Date and time of each comment, supporting trend analysis.                                                       |
| **username**    | The identifier for each commenter, allowing user-level analysis to identify recurrent toxic behavior.           |
| **link**        | The URL of the Reddit thread containing the comment.                                                            |
| **link_id**     | Metadata that allows for hierarchical analysis of comments, such as identifying parent-child comment relationships within threads. |
| **parent_id**   | Metadata to understand the parent-child relationships between comments within a thread.                         |
| **id**          | Unique identifier for each comment.                                                                            |
| **subreddit_id**| Identifier for the subreddit, helpful for filtering or comparing across subreddits.                             |
| **moderation**  | Information about whether the comment was flagged as controversial or removed, useful for studying moderation effects. |

#### 3.1.3 Computational Resources
Our resources for this project are:
| Resource                               | Description                                                                                                                       |
|----------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------|
| **Five Personal Laptops**              | These machines are available for local testing, small-scale data processing, and early-stage model experimentation.                          |
| **Google Colab**                       | Provides additional computational power and GPU access for larger-scale model training and data processing tasks.                |
| **High-Performance PC**                | (i7-6700k, 8GB RAM, RTX 2070 Super with 8GB VRAM) Primarily used for processing large datasets, such as labeling the extensive comment data. |
| **Pocket change for API calls**        | Allocated to leverage external models like OpenAI’s GPT for labeling and validation, where suitable, especially for cases where local computational resources are insufficient.  |

#### 3.1.4 Key Hypotheses
| Hypothesis                    | Description                                                                                                                      |
|------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| **Event-Driven Spikes in Toxicity** | Sensitive topics like politics or race are expected to show heightened toxicity, especially around specific events.              |
| **Recurrent Toxic Contributors**    | A small subset of users may disproportionately contribute to harmful content.                                                   |
| **Temporal Patterns in Toxicity**   | Toxic comments may vary by time of day, with higher toxicity likely during late hours.                                          |
| **Trigger Comments**                | Certain comments may initiate toxic comment chains, allowing for targeted interventions.                                        |

#### 3.1.5 Data Quality
**Incomplete Moderation Data** 

Moderation indicators are inconsistent, limiting analysis of the link between moderation and toxicity. Despite this limitation, the dataset still provides a robust foundation for identifying general toxicity trends.

**Assumptions on Deleted Content**

Comments labeled as "[deleted]" or "[removed]" are assumed to be non-toxic, based on the rationale that these may have been removed for reasons unrelated to toxicity, such as user discretion or policy violations. However, this assumption may introduce a minor bias, as some deleted comments could have contained harmful content. We mitigate this by focusing primarily on available text data for trend analysis. 

### 3.2 Data

#### 3.2.1 Data Collection
The data for this project was sourced from Shaun Khoo's Google link, providing a comprehensive collection of comments from Singapore-related subreddits (`r/Singapore`, `r/SingaporeRaw`, `r/SingaporeHappenings`) from 2020 to 2023.

Under the column link, a part of the post title can be seen. However, the titles are truncated for most posts, making it difficult to analyse the post titles. We expanded our dataset by using the Reddit API with the PRAW library to retrieve full post titles. While gathering complete titles, we also collected additional data on each post for further analysis.

| Column Name      | Description                        |
|------------------|------------------------------------|
| post_timestamp   | Time at which the post was created |
| comment_count    | Total number of comments in that post |
| vote_score       | Vote score for the post            |
| post_title       | Post Title                         |


#### 3.2.2 Data Cleaning (`Phase Specific`)

**Phase 1: Identifying Hate and Toxicity** 

Comments labelled as [removed], [deleted], or with NaN values in the text field were excluded. The dataset had 333,164 [deleted] comments, out of which 110,009 comments were not specified with reasons for removal under moderation and 76,411 comments are only specified as deleted under moderation. These rows lacked substantive content and would not contribute to toxicity analysis. Removing them reduced noise and kept the focus on meaningful discussions. 

**Phase 2: Community & User Behaviour Analysis**

Post titles were cleaned by converting text to lowercase, removing punctuation, special characters, and extra spaces. Most titles were in standard English, so minimal cleaning was needed. Stop word removal and lemmatization weren’t required, as we used the RAKE keyword algorithm, which performs these steps. The cleaned titles were added as a new column, `post_title_cleaned`.

During topic analysis, we excluded periodic “Random Discussion and Small Questions” threads from r/Singapore, as they lack specific topics. For username analysis, `[deleted]`, `sneakpeek_bot`, and `AutoModerator` were also excluded—`[deleted]` denotes voluntarily removed comments, while the other two are Reddit bots, irrelevant to user-based insights.

**Phase 3: Trend & Driver Analysis**

Preprocessed text by lowercasing, removing non-alphanumeric characters, and applying lemmatization to reduce words to their base form. Additionally, common English stop words were removed to focus on meaningful terms, creating consistent language patterns for analysis. Comments that became empty after preprocessing were also removed (e.g., those with only punctuation) as they lacked relevant content. Duplicate rows of data were also removed.

**Phase 4: Non-hateful/toxic Trigger Analysis**

To preprocess Reddit comments, we created a pipeline to clean and standardize text, enhancing its effectiveness in detecting nuanced, non-hateful toxicity triggers. This step was essential for filtering irrelevant text and normalizing patterns, helping the model focus on meaningful content likely to trigger harmful responses.

We downloaded key NLTK resources and initialized a lemmatizer and stopword set. A dictionary was defined to expand common abbreviations (e.g., "idk" to "I don’t know") to clarify context, as abbreviations can obscure meaning in non-harmful triggers. The pipeline involved the following steps:

| Step                                  | Description                                                                                                        |
|---------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| Convert to lowercase                  | Text is converted to lowercase for consistency and to prevent duplicate representations due to capitalization.     |
| Replace abbreviations                 | Common abbreviations are expanded for clarity (e.g., "idk" to "I don’t know").                                    |
| Remove URLs, mentions, hashtags, etc. | URLs, mentions, hashtags, Markdown links, and punctuation are removed to filter out non-essential symbols.         |
| Convert emojis                        | Emojis are converted to text descriptions, as they can convey subtle tones relevant to toxicity analysis.          |
| Tokenize and lemmatize                | Text is tokenized, and each word is reduced to its base form for consistency, simplifying word variations.         |
| Filter stopwords and non-alphabetic tokens | Stopwords are removed, and only alphabetic tokens are retained to focus on essential words.                      |

We recombined the processed tokens into strings and added them as a new `processed_text` column, providing a standardized format for nuanced analysis of non-harmful triggers and enabling easier identification of subtle toxicity and hate patterns.

#### 3.2.3 Features (`Phase Specific`)
No feature engineering was applied in Phases 1, 2, or 3 of the analysis.

**Phase 4: Non-hateful/toxic Trigger Analysis**

For feature engineering, we created `data_for_xai` as a copy of `deberta_cleaned` to preserve the original data. We extracted the numeric portion of `parent_id` as `pure_parent_id` to uniquely identify parent comments. Next, we calculated the number of toxic child comments for each `pure_parent_id`, counting only those marked toxic by `BERT_2_hate`. A variable **threshold** was set, specifying the minimum toxic child comments needed for a parent to be flagged as a trigger. This count was merged into `data_for_xai`, with missing values filled as zero for parents with no toxic child comments. In `merged_df`, a `trigger` column was added to mark whether a parent met the toxic child threshold. Finally, we balanced the dataset by sampling 100,000 rows from each class (triggered and non-triggered) in `balanced_df`, ensuring equal representation based on the defined threshold.

#### 3.2.4 Data Splitting (`Phase Specific`)
No model training or data splitting was required in Phases 1, 2, or 3.

**Phase 4: Non-hateful/toxic Trigger Analysis**

We used an 80/20 split with Scikit-Learn’s `train_test_split`, dedicating 80% for training and 20% for testing to assess the model’s performance on unseen data, ensuring reliable generalization.

### 3.3 Experimental Design (`Phase Specific`)

#### 3.3.1 Phase 1: Identifying Hate and Toxicity

To accurately label the Reddit dataset for hatefulness and toxicity, we evaluated a range of language models. Our selection prioritized high accuracy, efficiency, and affordability, with the F1-score chosen as the main evaluation metric to balance precision and recall. A validation set of 300 comments was created to test models against nuanced language and Singaporean cultural references.

**Algorithms & Model Used**
| **Category**           | **Model(s)**                                                                                                                                                                                                                              | **Purpose**                                                                                                                             |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| **BERT-Based Models**  | *sileod/deberta-v3-base-tasksource-toxicity*, *unitary/toxic-bert*, *GroNLP/hateBERT*, *textdetox/xlmr-large-toxicity-classifier*, *facebook/roberta-hate-speech-dynabench-r4-target*, *cointegrated/rubert-tiny-toxicity*, *badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification*, *citizenlab/distilbert-base-multilingual-cased-toxicity*, *GANgstersDev/singlish-hate-offensive-finetuned-model-v2.0.1*, *Hate-speech-CNERG/dehatebert-mono-english*, *cardiffnlp/twitter-roberta-base-hate*, *Hate-speech-CNERG/bert-base-uncased-hatexplain*, *mrm8488/distilroberta-finetuned-tweets-hate-speech* | Chosen for strong text classification capabilities, sensitivity to hate speech detection, and support for threshold tuning.              |
| **LLaMA-Based Models** | *meta-llama/Llama-3.2-1B-Instruct*, *meta-llama/Llama-3.2-3B-Instruct*, *aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct*                                                                                                                | Evaluated for multilingual support and contextual understanding, particularly for detecting local slang and cultural nuances.           |
| **API-Based Models**   | *gpt-4o-mini*, *claude-3-haiku-20240307*, *Perspective API* (trained on Jigsaw toxicity dataset for multiclass classification)                                                                                                            | Selected for high-performance toxicity detection with flexible API usage; however, large-scale deployment limited by resource constraints. |

**Evaluation Metric**

We optimized models based on the F1-score to handle class imbalances effectively, ensuring that both toxic and non-toxic comments were accurately labeled.

#### 3.3.2 Phase 2: Community and User Behaviour Analysis

**Topic Extraction**

***RAKE***  
We initially tried RAKE for keyword extraction in Reddit post titles, but it failed to capture core topics effectively. For example, in “Singapore reports 73 new COVID-19 cases, new cluster involving PCF Sparkletots centre linked to 18 cases,” RAKE produced “reports new covid cases new cluster involving PCF Sparkletots centre linked,” which lacked specificity. RAKE’s reliance on co-occurrence patterns limits its accuracy with short texts, often resulting in generic phrases that miss essential keywords.

***KeyBERT***  
We then tested KeyBERT, hoping its semantic approach would improve accuracy. However, KeyBERT also struggled with short text, as limited context hindered its ability to capture primary themes effectively.

***BERTopic***  
Ultimately, we chose BERTopic due to its ability to capture contextual information, even in short texts. Unlike LDA, which requires longer documents, BERTopic’s BERT-based embeddings handle brief phrases well. Given the small dataset, we avoided fine-tuning to prevent overfitting, as some subreddits (e.g., r/SingaporeHappenings) have as few as 1,000 titles, making pretrained BERT ideal for capturing high-level topics.

**Top 100 Posts for Analysis**  
To ensure balanced representation across subreddits, we analyzed the top 100 posts from each. This sample size minimized subreddit bias, maintaining result sensitivity without skewing topic popularity or harmfulness analysis. Testing with different sample sizes showed consistent top topics, validating our choice of 100 posts.


#### 3.3.3 Phase 3: Trend and Driver Analysis

**Analyze Overall Trend in Hate and Toxicity**

To detect a trend in hate and toxicity, we calculate the **ratio of hate to non-hate comments** over time as an indicator of hate prevalence. Using **Ordinary Least Squares (OLS) linear regression**, it identifies any directional trend in this ratio. The weekly hate-to-non-hate ratio is then plotted alongside the regression trend line: an upward slope indicates an increase in hate content over time, while a flat or downward slope suggests stability or decline. This approach provides a clear visual and statistical analysis to assess changes in hate and toxicity over time.

**Analyze Trend in Hate and Toxicity per Topic**

To identify topics with rising trends in hate and toxicity, we first sampled a representative subset of 230,000 comments from the original dataset of approximately 1.5 million hate comments, selecting 5,000 hate comments per month. This approach preserves temporal diversity, ensuring a balanced representation of monthly variations in hate trends. By creating a manageable dataset for experimentation, we enable efficient topic modeling while retaining essential patterns and themes across time. We then tested three approaches for topic modeling:

| Technique                        | Description                                                                                           | Limitations                                                     |
|----------------------------------|-------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------|
| **Latent Dirichlet Allocation (LDA)** | A probabilistic model effective for large datasets, identifying topics based on word distributions. Performs well on structured text but struggles with short, nuanced language and requires a set topic count. | Limited adaptability with informal language and short text.     |
| **Non-Negative Matrix Factorization (NMF)** | Decomposes term matrices to extract topics, performing well on sparse data. However, it is less effective with informal or culturally specific language and requires a predefined topic count. | Lacks flexibility in handling informal language and short text. |
| **BERTopic**                     | Utilizes Sentence-BERT embeddings, UMAP for dimensionality reduction, and HDBSCAN for clustering, allowing dynamic topic discovery without a predefined count. Adaptable for short, informal text, capturing contextual nuances in diverse language. Extensive tuning provides deeper insights. | None significant; highly adaptable for online discussions.       |

LDA and NMF require processed text because they rely on word frequency, so stop words can skew topic results. Removing stop words ensures only meaningful terms shape topics.

With BERTopic, which uses transformer-based embeddings, there’s no need to preprocess the data by removing stop words, as the model captures context more accurately with the full text. Removing stop words beforehand could reduce embedding accuracy. Instead, after generating embeddings and clustering, `CountVectorizer` can remove stop words from topic labels, preserving context while refining topic representation.

Therefore, when we experimented on the three topic modelling techniques, LDA and NMF was given processed text data while BERTopic was mainly given the raw text.

***Final Topic Modelling Technique Choice***

BERTopic was chosen for its adaptability to short, informal comments and its ability to dynamically discover topics without requiring a predefined topic count. Its extensive tuning capabilities further support deep insights into diverse online discussions. While [OCTIS (Optimizing and Comparing Topic models Is Simple)](https://github.com/MIND-Lab/OCTIS) provides a robust framework for training, analyzing, and comparing topic models using Bayesian Optimization, BERTopic remained the preferred choice. This is because there is no single correct way to evaluate a topic model across all use cases, allowing users to select metrics suited to their specific goals rather than optimizing solely for general-purpose coherence metrics.

Furthermore, BERTopic 

***Fine-Tuning BERTopic***

To fine-tune BERTopic for optimal topic modeling results, we began by pre-calculating document embeddings for the ~230,000 hate comments using the SentenceTransformer `all-MiniLM-L6-v2` which is the default used by BERTopic. This step ensured that embeddings were readily available for multiple iterations of parameter tuning, significantly saving computation time. The table below shows the sets of hyperparameters used for tuning BERTopic.
| Row | `umap_n_neighbors` | `umap_min_dist` | `umap_n_components` | `hdbscan_min_cluster_size` | `hdbscan_min_samples` |
|------|---------------------|-----------------|----------------------|----------------------------|------------------------|
| 1    | 15                 | 0.1             | 5                    | 200                        | 1                      |
| 2    | 15                 | 0.1             | 5                    | 200                        | 5                      |
| 3    | 15                 | 0.1             | 5                    | 300                        | 1                      |
| 4    | 15                 | 0.1             | 5                    | 300                        | 5                      |
| 5    | 15                 | 0.3             | 5                    | 200                        | 1                      |
| 6    | 15                 | 0.3             | 5                    | 200                        | 5                      |
| 7    | 15                 | 0.3             | 5                    | 300                        | 1                      |
| 8    | 15                 | 0.3             | 5                    | 300                        | 5                      |
| 9    | 25                 | 0.1             | 5                    | 200                        | 1                      |
| 10   | 25                 | 0.1             | 5                    | 200                        | 5                      |
| 11   | 25                 | 0.1             | 5                    | 300                        | 1                      |
| 12   | 25                 | 0.1             | 5                    | 300                        | 5                      |
| 13   | 25                 | 0.3             | 5                    | 200                        | 1                      |
| 14   | 25                 | 0.3             | 5                    | 200                        | 5                      |
| 15   | 25                 | 0.3             | 5                    | 300                        | 1                      |
| 16   | 25                 | 0.3             | 5                    | 300                        | 5                      |

The table below shows the rationale behind different sets of hyperparameters.
| Row | Description |
|------|-------------|
| **1–4** | `umap_n_neighbors=15`, `umap_min_dist=0.1`: These configurations create compact, highly localized clusters with `n_neighbors` set to a lower value and `min_dist` set to 0.1. By varying `hdbscan_min_cluster_size` and `hdbscan_min_samples`, we can explore both fine-grained and strict clustering (with `min_samples=5`) and looser, small clusters (with `min_samples=1`). |
| **5–8** | `umap_n_neighbors=15`, `umap_min_dist=0.3`: These configurations test how increasing the minimum distance in UMAP affects cluster compactness for a smaller neighborhood size. The larger `min_dist` creates slightly looser clusters, allowing for more spread. Varying `min_cluster_size` and `min_samples` again tests the balance between fine-grained clusters and broader groupings. |
| **9–12** | `umap_n_neighbors=25`, `umap_min_dist=0.1`: With a higher `n_neighbors`, these configurations capture more global structure, creating broader clusters with compact boundaries (`min_dist=0.1`). By adjusting `min_cluster_size` and `min_samples`, we test for both strict (larger `min_samples` and `min_cluster_size`) and flexible clustering (smaller values). |
| **13–16** | `umap_n_neighbors=25`, `umap_min_dist=0.3`: These configurations use both a high `n_neighbors` and a larger `min_dist`, allowing for broad clusters with looser boundaries. This setup captures large-scale structure, which may be suitable for finding broader topics. Adjustments to `min_cluster_size` and `min_samples` ensure we capture both robust clusters (higher values) and more flexible clusters (lower values). |

This layout covers a range of parameter combinations that balances between local and global structures (`umap_n_neighbors`), compact and loose clusters (`umap_min_dist`), fine-grained and broad clustering (`hdbscan_min_cluster_size` and `hdbscan_min_samples`).

To improve topic interpretability, we incorporated multiple representation models. The table below shows the representation models used for tuning BERTopic.
| Representation Model | Description                                                                                                   |
|----------------------|---------------------------------------------------------------------------------------------------------------|
| Main                 | BERTopic's default topic representation.                                                                      |
| KeyBERTInspired              | Uses KeyBERT-inspired keyword extraction which emphasizes key terms, providing concise and straightforward topic representations.                                  |
| POS                  | Incorporates Part-of-Speech tagging with the "en_core_web_sm" model, focusing on nouns for topic clarity.    |
| KeyBERT_MMR          | Combines KeyBERT with Maximal Marginal Relevance (MMR) to balance keyword relevance and diversity in topic keywords. |

For evaluation, we calculated coherence scores using the c_v metric for each combination of parameters and representation models. This coherence scoring provided a metric-based assessment of topic clarity and relevance, facilitating efficient comparison across configurations. 

Through this structured process of parameter tuning, enhanced representation, and coherence scoring, we aim to obtain a BERTopic model that delivers high-quality, meaningful topics. 

***Merging Similar Topics***

We reduced the number of topics in our optimized BERTopic model by using its hierarchical clustering feature, which organizes topics into a tree structure. This allowed us to visualize and manually merge thematically similar topics, resulting in a refined set of cohesive, interpretable themes.

***Outlier Reduction***

To reduce comments with an outlier topic in our BERTopic model, we applied a two-step approach using c-TF-IDF followed by Distributions. First, c-TF-IDF helped reassign outlier comments by clustering them into clear, interpretable topics based on term frequency and distinct themes. For remaining comments still marked as outliers, we applied Distributions to assign them probabilistically to the most closely related topics, capturing nuanced relationships without sacrificing interpretability.

***Transforming the remaining hate comments that were not sampled***

After further optimizing our BERTopic model with topic merging and outlier reduction, we processed the remaining 1.3 million hate/toxic comments in batches of 200,000. For each batch, we generated embeddings with `all-MiniLM-L6-v2`, applied the BERTopic model, and refined outliers using **c-TF-IDF** followed by **Distributions**. This approach provided cohesive, refined topic labels for a unified analysis of hate/toxic content trends.

***Time Series Analysis***

We plotted the weekly count of hate and toxic comments related to a chosen topic on Reddit to identify time periods with rising trends or sudden spikes in hate and toxicity for that topic.

**Analyze Drivers of Trend/Spikes in Harmful Comments per Topic**

To identify drivers of increased hate and toxicity for specific topics and time periods, we used a targeted keyword extraction approach. This method reveals key terms that may contribute to spikes in toxic comments, highlighting underlying drivers of hate and toxicity within each topic and timeframe.

First, we filtered comments to a chosen topic (e.g., politics or race-related issues) and time period with observed spikes in toxicity. This narrowed dataset provided a targeted context for identifying potential drivers of hate and toxicity.

Next, we cleaned the comments by removing commonly associated topic words, using BERTopic-generated representations like POS tags, KeyBERT, and Maximal Marginal Relevance to exclude primary terms. This filtering step highlighted additional, context-specific drivers of toxicity rather than main topic indicators.

KeyBERT then extracted impactful keywords from the refined comments. By focusing on terms with high relevance and frequency, we identified themes or language that contributed significantly to toxicity spikes within the topic and timeframe.

Finally, we will visualize the top keywords in a word cloud, helping to detect recurring patterns and themes in toxic discourse. This method offered a clearer view of the dynamics and potential triggers of hate in specific online discussions, enhancing our understanding of toxicity drivers across topics and periods.

#### 3.3.4 Phase 4: Non-hateful & Non-toxic Trigger Analysis

**Defining Non-hateful & Non-toxic Triggers**

A "Trigger Comment" is a non-harmful comment that sparks harmful replies, identified by the number of harmful responses it receives. Our goal is to identify comments that unintentionally escalate hate and toxicity and uncover what drives these hostile exchanges.

**Thresholding Process**

We tested different threshold levels, varying the minimum count of harmful responses needed to label a parent comment as a trigger. By iterating from 0 to the maximum observed `toxic_child_count`, we created binary trigger labels at each threshold, allowing us to evaluate how varying thresholds impacted trigger identification.

**Evaluation of Trigger Counts**

At each threshold, we calculated the total number of comments classified as triggers to assess sensitivity. This evaluation helped balance inclusivity and specificity, identifying thresholds that captured relevant toxic interactions without overestimating toxicity triggers.

The results were plotted to visualize the relationship between thresholds and trigger counts, providing insight into how threshold sensitivity impacts trigger classification. This visualization was critical in selecting a threshold level for model training and evaluation.

---

**Baseline Experiment with TF-IDF**

To establish a baseline for predicting trigger comments, we implemented TF-IDF vectorization on text data, setting a threshold of 1 for the `toxic_child_count` feature and focusing on a balanced dataset.

**Data Preparation**

We loaded a balanced dataset with 100,000 samples for each class (trigger and non-trigger) to avoid class imbalance issues. This sample size ensured computational efficiency while providing ample data for model training.

**Text Vectorization**

For text vectorization, we applied TF-IDF with an n-gram range of 1 to 3, limiting the feature set to a maximum of 5,000 features to enhance efficiency. TF-IDF was chosen to capture the importance of words in Reddit comments by downweighting common terms. The selected n-gram range effectively captured contextual information within short comments, supporting more accurate toxicity detection.

---

**Model Training**

We selected accuracy as our primary metric to ensure balanced detection of triggers and non-triggers. With a balanced dataset, accuracy effectively measures the model’s correct classifications, aiming to capture true triggers while minimizing misclassifications. This approach provides a reliable benchmark for the model’s overall detection capability.

**Model Comparison**

To establish a multi-model approach for detecting trigger comments, we trained and evaluated four models: Ridge Classifier, Logistic Regression, XGBoost, and Neural Network. These models were chosen for their varied structures and ability to capture complex patterns, with accuracy as the primary metric on a balanced dataset.

| **Model Name**           | **Description**                                                                                                                                           | **Justification**                                         | **Hyperparameters & Grid Search**                                |
|--------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|------------------------------------------------------------------|
| Ridge Classifier         | Regularized with `alpha` to handle high-dimensional TF-IDF data, preventing overfitting.                                                                    | Effective for TF-IDF data                                  | `alpha`: [500, 800, 1000] (10-fold CV)                           |
| Logistic Regression      | L2 regularization, balancing bias and variance, with tuning on `C` and solver options.                                                                      | Simple and interpretable                                   | `C`: [0.1, 1.0, 10.0], `solver`: ['liblinear', 'saga'] (3-fold CV) |
| XGBoost                  | Captures complex relationships with tuning on tree depth, learning rate, and boosting rounds.                                                                | Handles nuanced toxicity patterns                          | `max_depth`: [3, 5], `learning_rate`: [0.1, 0.2], `n_estimators`: [100, 200] (3-fold CV) |
| Neural Network           | Three-layer architecture with ReLU in hidden layers and Sigmoid for binary output, tailored for high-dimensional TF-IDF.                                   | Deeper learning potential                                  | `learning_rate`: [0.0001], `hidden_units`: [128], `batch_size`: [64] |

After training each model across different thresholds, we assessed their performance using multiple metrics—Accuracy, Precision, Recall, F1-Score, and ROC-AUC. This approach provided a comprehensive evaluation of each model's effectiveness in detecting trigger comments, offering insights for further refinement and a clear view of sensitivity across thresholds.

**Threshold-wise Model Comparison and Visualization**

We evaluated each model's performance across different thresholds, storing accuracy and other metrics to analyze trigger detection sensitivity. Plotting accuracy scores for each model allowed us to observe threshold-specific sensitivities. To further aid in analysis, we created an interactive table where users can select thresholds and models, dynamically viewing detailed performance metrics to identify optimal configurations. This evaluation phase provided valuable insights into each model's robustness and sensitivity across thresholds, guiding the selection of the most effective models and thresholds for detecting triggers.

---

**Training the Optimal Model and Applying Explainable AI**

With the optimal threshold (threshold = 4) and model configuration identified, we trained the neural network using the best hyperparameters for reliable performance. Explainable AI techniques were then applied to interpret predictions, highlighting key features driving trigger comments.

**Explainable AI Techniques**

| **Technique**            | **Description**                                                                                                                    | **Justification**                                                                                               | **Visualization Methods**                         |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| **SHAP**                 | Calculates feature importance, assigning each word a SHAP value to indicate its impact on model predictions.                       | Provides a comprehensive view of words most influencing toxicity classification across the dataset.              | Bar chart, word cloud                             |
| **LIME**                 | Model-agnostic method that perturbs input data, training local models to interpret specific predictions and word contributions.    | Enables localized insights, interpreting each word’s effect on individual predictions.                           | Highlighted words with per-comment contributions  |
| **Integrated Gradients** | Attribution for neural networks, calculating cumulative gradients from baseline to input to indicate each word's influence.       | Offers both global and local feature importance for understanding cumulative word impact on predictions.        | Color-coded highlights, bar chart of top tokens   |

This final step, integrating model optimization with interpretability techniques, provided a robust, transparent model for detecting harmful triggers, offering insights into the linguistic factors driving harmful content in Reddit comments.

## Section 4: Findings

### 4.1 Results (`Phase Specific`)
#### 4.1.1 Phase 1: Identifying Hate and Toxicity
These are the results of the models we tested against our 300 expertly labelled data. 

| Model                                         | Best Toxic F1 Score | Toxic Threshold | Best Hate F1 Score | Hate Threshold | Combined Best F1 Score | Combined Threshold | Time Taken |
|-----------------------------------------------|----------------------|-----------------|--------------------|----------------|------------------------|--------------------|------------|
| sileod/deberta-v3-base-tasksource-toxicity    | 0.547368            | 0.01            | 0.573034           | 0.04          | 0.675079               | 0.01               | 12s        |
| unitary/toxic-bert                            | 0.543689            | 0.00            | 0.513889           | 0.40          | 0.648649               | 0.00               | 4s         |
| GroNLP/hateBERT                               | 0.554455            | 0.40            | 0.397849           | 0.38          | 0.651584               | 0.38               | 4s         |
| textdetox/xlmr-large-toxicity-classifier      | 0.540146            | 0.00            | 0.493671           | 0.05          | 0.645598               | 0.00               | 4s         |
| facebook/roberta-hate-speech-dynabench-r4-target | 0.540146         | 0.00            | 0.422360           | 0.04          | 0.645598               | 0.00               | 4s         |
| cointegrated/rubert-tiny-toxicity             | 0.540146            | 0.00            | 0.429268           | 0.04          | 0.645598               | 0.00               | 1s         |
| badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification | 0.540146 | 0.00 | 0.391421 | 0.00 | 0.645598 | 0.00 | 2s         |
| citizenlab/distilbert-base-multilingual-cased-toxicity | 0.540146 | 0.00 | 0.48062  | 0.57 | 0.645598 | 0.00 | 3s         |
| GANgstersDev/singlish-hate-offensive-finetuned-model-v2.0.1 | 0.543689 | 0.00 | 0.395722 | 0.00 | 0.648649 | 0.00 | 3s         |
| Hate-speech-CNERG/dehatebert-mono-english     | 0.543689            | 0.00            | 0.476190           | 0.08          | 0.648649               | 0.00               | 4s         |
| cardiffnlp/twitter-roberta-base-hate          | 0.540146            | 0.00            | 0.423077           | 0.04          | 0.645598               | 0.00               | 4s         |
| Hate-speech-CNERG/bert-base-uncased-hatexplain | 0.571429           | 0.04            | 0.444444           | 0.04          | 0.658824               | 0.04               | 6s         |
| mrm8488/distilroberta-finetuned-tweets-hate-speech | 0.556122          | 0.04            | 0.391421           | 0.00          | 0.646226               | 0.04               | 2s         |
| meta-llama/Llama-3.2-1B-Instruct              | 0.0992908           | NaN             | 0.211382           | NaN           | 0.364341               | NaN                | 30s        |
| meta-llama/Llama-3.2-3B-Instruct              | 0.493023            | NaN             | 0.390244           | NaN           | 0.534483               | NaN                | 14min      |
| aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct | 0.514286         | NaN             | 0.324786           | NaN           | 0.517857               | NaN                | 53min      |
| gpt-4o-mini-2024-07-18                        | -                   | -               | -                  | -             | 0.694534               | -                  | 6min26s    |
| claude-3-haiku-20240307                       | -                   | -               | -                  | -             | 0.356757               | -                  | 5min19s    |
| Perspective API                               | -                   | -               | -                  | -             | 0.648649               | 0.09               | 6min28s    |

**Highest-Performing Model**

`gpt-4o-mini` achieved the highest F1-score (0.70) but was cost-prohibitive for large-scale labeling. We selected `sileod/deberta-v3-base-tasksource-toxicity` with an F1-score of 0.68 as a balanced solution for labeling efficiency and affordability.

**Final Model Choice**

We selected `sileod/deberta-v3-base-tasksource-toxicity` as our primary labeling tool, achieving an F1-score of 0.68. This model offers a balanced solution, providing efficient, cost-effective labeling while reliably identifying both toxic and non-toxic content across the dataset.

#### 4.1.2 Phase 2: Community and User Behaviour Analysis
**General Overview across different subreddits**

| Subreddit            | Total Posts | Total Text | Total Hate | Text per Post | Hate per Text |
|----------------------|-------------|------------|------------|---------------|---------------|
| r/Singapore          | 86,241      | 4,047,924  | 1,359,613  | 46.94         | 0.3359        |
| r/SingaporeRaw       | 18,482      | 409,386    | 189,920    | 22.15         | 0.4639        |
| r/SingaporeHappenings| 1,289       | 47,019     | 24,147     | 36.48         | 0.5136        |

Our dataset is imbalanced, with most posts and comments from r/Singapore, followed by r/SingaporeRaw and r/SingaporeHappenings. To address this, our post and user analyses are conducted by subreddit.

r/Singapore is the most active, averaging 46 comments per post, and has the lowest percentage of harmful comments. In contrast, r/SingaporeHappenings is the most toxic, with about half of the comments deemed harmful. This suggests a negative correlation between moderation level and harmful content, as r/Singapore has strict moderation, r/SingaporeRaw enforces basic Reddiquette, and r/SingaporeHappenings has no moderation.


**Most commonly discussed topics using comment_count obtained by Reddit API**

To analyze subreddit characteristics, we used `comment_count` data from the Reddit API to identify the most discussed topics in each subreddit. This approach provides the latest information, as `comment_count` reflects updated engagement without needing individual comments. We identified popular topics by selecting the top 100 posts with the highest comment counts for each subreddit and counting topic frequency among them.



| r/Singapore                                         | r/SingaporeRaw                               | r/SingaporeHappenings              |
|-----------------------------------------------------|----------------------------------------------|------------------------------------|
| man spore singapore covid19                         | singapore like new man                       | man singapore woman police        |
| lgbtq lgbt gay transgender                          | penalty death penalty death drug             | conflict gaza israelhamas israel  |
| pritam raeesah pritam singh singh                   | china chinese singaporean chinese taiwan     | man spore singapore covid19       |
| racism racist racial racism singapore               | singapore singapore singapore singaporeans singaporean |                                    |
| pm lee pm lee hsien loong                           | sg sg sg sgs sgd                             |                                    |
| live discussion constituency political 2020 july political broadcasts | men women singaporean girls |                                    |
| nicole nicole seah seah leon                        | racist racism minorities racial              |                                    |
| dining eateries pax groups                          | man spore singapore covid19                  |                                    |
| election elections singapore general presidential   | teachers teacher schools school              |                                    |
| singaporeans stereotypes singaporean young singaporeans | women ns serve ns serve           |                                    |

The table shows distinct topic trends across subreddits: r/Singapore focuses on political content like PM Lee and elections, r/SingaporeRaw centers on social issues like the death penalty, racism, and national service, while r/SingaporeHappenings features varied topics like gender and war. This insight helps stakeholders better target specific subreddits for sensitive topics.

**Popular & Harmful Topics by Subreddit**

Popular and Harmful Topics were found using similar methods. We took the top 100 posts with the most number of comments and most number of harmful comments respectively for each subreddit and then count the frequency of the topics among these 100 posts.  

| Subreddit              | Most Popular Topics 1               | Most Popular Topics 2             | Most Popular Topics 3           |
|------------------------|-------------------------------------|-----------------------------------|---------------------------------|
| r/Singapore            | Man, Singapore, Covid19            | Pritam Singh, Raeesah             | PM, PM Lee, Lee Hsien Loong     |
| r/SingaporeRaw         | Singapore, New                     | Singapore, Singaporeans           | Death Penalty, Drug             |
| r/SingaporeHappenings  | Man, Singapore, Women, Police      | Conflict, Gaza, Israel, Hamas     |                                 |

| Subreddit              | Most Harmful Topics 1               | Most Harmful Topics 2                | Most Harmful Topics 3           |
|------------------------|-------------------------------------|--------------------------------------|---------------------------------|
| r/Singapore            | Man, Singapore, Covid19            | Racist, Racism, Singapore            | LGBTQ, Gay, Transgender         |
| r/SingaporeRaw         | Singapore, New                     | Men, Women, Singaporeans, Girls      | Singapore, Singaporeans         |
| r/SingaporeHappenings  | Man, Singapore, Women, Police      | Conflict, Gaza, Israel, Hamas        |                                 |

From the table, we can observe that quite a lot of topics appearing in popular topics also appear in harmful topics, which suggests the likelihood that popular topics also tends to have more harmful contents. 

**Most Toxic Usernames & Most Toxic Username Count accounting for 10% of Harmfulness in each subreddit**

| subreddit             | username            | hate_comment_count | hate_percentage |
|-----------------------|---------------------|---------------------|-----------------|
| r/Singapore           | deangsana           | 9268               | 0.712183        |
| r/Singapore           | blackwoodsix        | 9004               | 0.691896        |
| r/Singapore           | FitCranberry        | 8734               | 0.671149        |
| r/SingaporeRaw        | AyamBrandCurryTuna  | 2536               | 1.380722        |
| r/SingaporeRaw        | laglory             | 1710               | 0.931007        |
| r/SingaporeRaw        | jypt98              | 1707               | 0.929374        |
| r/SingaporeHappenings | adamlau8899         | 125                | 0.519103        |
| r/SingaporeHappenings | ZXcvk123            | 121                | 0.502492        |
| r/SingaporeHappenings | logicnreason93      | 103                | 0.427741        |




| Subreddit             | Username Count | Total No. of Harmful Comments |
|-----------------------|----------------|------------------------------|
| r/Singapore           | 27             | 135961                       |
| r/SingaporeRaw        | 37             | 18992                        |
| r/SingaporeHappenings | 17             | 2415                         |

The 2 tables shows that quite a significant portion of the toxic comments are contributed by a small number of usernames. For example, the top 27 harmful users in r/Singapore account for 10% of the harmful comments which is 135961 comments in absolute numbers. 

The `Reddit_Report4.ipynb` shares a cumulative plot and a function to output the number of usernames involved for specified harmful comments percentage level. 


#### 4.1.3 Phase 3: Trend and Driver Analysis

**Overall Trend in Hate and Toxicity**

The plot of weekly harmful-to-non-harmful comment ratio with trend line showed a 26.31% increase in weekly proportion of harmful to non-harmful comments from Jan 2020 to Sep 2023.

**BERTopic Hyperparameter Tuning Results**

| umap_n_neighbors | umap_n_components | umap_min_dist | hdbscan_min_cluster_size | hdbscan_min_samples | vectorizer_min_df | vectorizer_max_df | vectorizer_ngram_range | representation_model | coherence_score |
|------------------|-------------------|---------------|--------------------------|----------------------|--------------------|--------------------|------------------------|----------------------|-----------------|
| 15               | 5                 | 0.1           | 200                      | 1                    | 0.001             | 0.8               | (1, 1)                 | POS                  | 0.491625        |
| 15               | 5                 | 0.1           | 200                      | 1                    | 0.001             | 0.8               | (1, 1)                 | Main                 | 0.490929        |
| 15               | 5                 | 0.1           | 200                      | 5                    | 0.001             | 0.8               | (1, 1)                 | Main                 | 0.485083        |
| 25               | 5                 | 0.1           | 200                      | 1                    | 0.001             | 0.8               | (1, 1)                 | POS                  | 0.484493        |
| 25               | 5                 | 0.1           | 200                      | 5                    | 0.001             | 0.8               | (1, 1)                 | POS                  | 0.484057        |

The top coherence score was achieved with `umap_n_neighbors=15`, `umap_n_components=5`, `umap_min_dist=0.1`, `hdbscan_min_cluster_size=200`, `hdbscan_min_samples=1` and using POS representation. This configuration was selected for the final BERTopic model. 

The table above only shows the first 5 rows.

**Topic Modeling Results**

The table below presents an overview of topics from the tuned BERTopic model, initially identifying 166 topics, including a significant outlier topic with a high comment count. To address this, we plan to refine the model through topic merging and outlier reduction.

| Topic | Count | Name                                  | Representation                                 | KeyBERT                                          | POS                                              | KeyBERT_MMR                                      | Representative_Docs                              |
|-------|-------|---------------------------------------|------------------------------------------------|--------------------------------------------------|--------------------------------------------------|--------------------------------------------------|--------------------------------------------------|
| -1    | 131216| -1_singapore_chinese_covid_government | [singapore, chinese, covid, government, ...]    | [singaporeans, singaporean, covid, ...]           | [chinese, government, police, ...]                | [singaporeans, covid, ...]                       | [Fight at Taman Jurong coffeeshop ...]           |
| 0     | 4703  | 0_gay_lgbt_lgbtq_religion             | [gay, lgbt, lgbtq, religion, ...]              | [lgbt, homophobic, ...]                           | [gay, lgbtq, religious, ...]                      | [homophobic, gay, ...]                           | [Not all of them unfortunately...]               |
| 1     | 3137  | 1_sia_siao_lao_jiak                   | [sia, siao, lao, ...]                          | [sia, sial, sian, ...]                            | [sia, nasi, peng, ...]                            | [sia, sian, siao, ...]                           | [Lol idk what to watch sia...]                   |
| 2     | 2714  | 2_road_car_cyclists_driver            | [road, car, cyclists, ...]                     | [cyclists, cycling, ...]                          | [road, car, drivers, ...]                         | [drivers, road, cyclist, ...]                    | [Full of excuses...]                             |
| 3     | 2454  | 3_hahaha_joke_funny_omg               | [hahaha, joke, funny, ...]                     | [hahah, joke, ...]                                | [joke, funny, omg, ...]                           | [hahaha, joke, ...]                              | [HAHAHA i guess i just...]                       |

#### Hierarchical Topic Clustering and Merging

Our optimized BERTopic model generated a hierarchical clustering tree that allowed us to identify and merge similar topics. A truncated part of the tree is shown below:

```
├─ pap_vote_opposition_wp_parliament
│    ├─ pap_vote_opposition_tharman_tkl
│    │    ├─ tharman_tkl_vote_nks_election
│    │    │    ├─ ■── tharman_tkl_nks_vote_pap ── Topic: 90
│    │    │    └─ ■── vote_election_voting_votes_polling ── Topic: 121
│    │    └─ pap_opposition_vote_wp_party
│    │         ├─ ■── pap_opposition_vote_party_election ── Topic: 115
│    │         └─ ■── pap_opposition_vote_wp_oppo ── Topic: 16
│    └─ rk_parliament_wp_khan_ps
│         ├─ ■── rk_wp_khan_ps_pritam ── Topic: 87
│         └─ ■── ministers_mp_parliament_minister_mps ── Topic: 62
```

In the example above, we decided to merge these topics (90, 121, 115, 16, 87, 62) to form the overarching topic `pap_vote_opposition_wp_parliament` This merging refined the model by consolidating overlapping topics, providing clearer, more interpretable themes. This is done for other similar topics as well.

#### Outlier Reduction

Out of ~230,000 comments, 131,216 were initially classified as outliers. After applying the outlier reduction framework, 127,680 comments were reassigned to existing topics, significantly reducing the outlier pool and enhancing topic relevance. Examples of reclassified comments include:

```
Original Comment: “bring an umbrella! it gets really hot during the day.”
New Topic: weather, rain, climate, heat

Original Comment: “Ahhh yes, the worst Smash character.”
New Topic: movie, netflix, ads, watch

Original Comment: “I used to hate going to school cos I was bullied...”
New Topic: bully, bullying, teachers
```

In the end, there are a total of 119 topics, one of which is the outlier topic, for the 1,547,105 hate and toxic comments.

---

For the results of (2) identifying topics with rising trends in hate and toxicity and (3) pinpointing drivers of increased hate and toxicity for each topic, please refer to the *Time Series Analysis* and *Identify Correlations with External Events (using LLMs)* sections in the `TopicModelling_TimeSeries.ipynb` notebook. These sections contain detailed findings supported by numerous visualizations and further explanations.

#### 4.1.4 Phase 4: Non-hateful/Toxic Trigger Analysis

**Threshold Impact on Classification**

We analyzed how varying thresholds for harmful child comments affected parent comment classification as triggers, allowing us to assess trigger classification sensitivity based on harmful child counts.

***Trigger Counts at Various Thresholds (1-15)***
| Threshold       | 1       | 2       | 3       | 4       | 5      | 6     | 7     | 8     | 9    | 10   | 11  | 12 | 13 | 14 | 15 |
|-----------------|---------|---------|---------|---------|--------|-------|-------|-------|------|------|-----|----|----|----|----|
| Trigger Count   | 407,343   | 57,131   | 14,891   | 5,375    | 2,257   | 1,044  | 517   | 297   | 159  | 92   | 56  | 33 | 24 | 17 | 12 |

***Trigger Counts at Various Thresholds (16-30)***
| Threshold       | 16  | 17  | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 |
|-----------------|-----|-----|----|----|----|----|----|----|----|----|----|----|----|----|----|
| Trigger Count   | 8   | 7   | 6  | 4  | 4  | 3  | 3  | 2  | 2  | 2  | 2  | 2  | 1  | 1  | 0  |

***Observations***
| **Aspect**           | **Description**                                                                                                                                                                |
|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Lower thresholds** | Captured a larger number of parent comments as triggers, encompassing a wide range of potential toxicity but risking higher false positives.                                   |
| **Higher thresholds**| Reduced false positives by being more selective, though potentially missing less overt toxic interactions.                                                                    |
| **Maximum Threshold**| Identified the maximum threshold that balanced sensitivity (capturing true triggers) and specificity (minimizing false positives), providing reliable classification of triggers aligned with project goals. |

This threshold and trigger analysis was crucial for setting a baseline threshold that would guide the following model training and evaluation phases, ensuring our classification of triggers was both inclusive and precise.

**Model Experiments**

| Model              | Best Parameters                              | Performance (Precision, Recall, F1-Score, Accuracy, ROC-AUC)         | Technique & Top 5 Words Contributing to Trigger Class (Values)            |
|--------------------|----------------------------------------------|----------------------------------------------------------------------|------------------------------------------------------|
| **Ridge Classifier**      | `alpha = 500`                           | Precision: 0.57 <br> Recall: 0.56 <br> F1-Score: 0.56 <br> Accuracy: 0.56 <br> ROC-AUC: 0.5648 | **SHAP** <br> `terminal`: 0.0039, <br> `guy`: 0.0026, <br> `industry`: 0.0025, <br> `frequently`: 0.0024, <br> `outcome`: 0.0024 |
| **Logistic Regression**   | `C = 0.1`, `solver = liblinear`         | Precision: 0.57 <br> Recall: 0.57 <br> F1-Score: 0.57 <br> Accuracy: 0.57 <br> ROC-AUC: 0.5664 | **SHAP** <br> `guy`: 0.0050, <br> `industry`: 0.0042, <br> `yeah`: 0.0039, <br> `terminal`: 0.0036, <br> `contract`: 0.0028 |
| **XGBoost**        | `learning_rate = 0.1`, `max_depth = 3`, `n_estimators = 100` | Precision: 0.55 <br> Recall: 0.54 <br> F1-Score: 0.53 <br> Accuracy: 0.54 <br> ROC-AUC: 0.5420 | **SHAP** <br> `guy`: 0.0016, <br> `girl`: 0.0009, <br> `remindme`: 0.0007, <br> `price`: 0.0006, <br> `thanks`: 0.0005 |
| **Neural Network** | Final Epoch (10/10) | Precision: 0.59 <br> Recall: 0.60 <br> F1-Score: 0.59 <br> Accuracy: 0.60 | **Integrated Gradients** <br> `people`: 19.73, <br> `like`: 16.80, <br> `laugh`: 10.82, <br> `want`: 10.81, <br> `loud`: 9.29 |

Based on initial testing, the Neural Network performed the best, achieving the highest scores in precision (0.59), recall (0.60), F1-score (0.59), and accuracy (0.60). Its use of Integrated Gradients provided detailed insights into influential words, making it the most effective model in this experiment for identifying toxicity triggers.

**Applying Explainable AI (XAI)**

| **XAI Technique**          | **Description**                                                                                                                                                                                                                              |
|----------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **SHAP (SHapley Additive exPlanations)** | SHAP values quantify each word’s impact on predictions for models like Ridge Classifier, Logistic Regression, and XGBoost. Positive values increase the likelihood of a comment being classified as a trigger. For example, in Logistic Regression, the word "guy" with a SHAP value of 0.0050 contributes positively to trigger classification. SHAP values are additive, meaning individual contributions sum up to the model’s prediction. |
| **Integrated Gradients**   | For Neural Networks, Integrated Gradients values reflect each word’s cumulative influence on the trigger classification. These values, often larger than SHAP values, accumulate from a baseline to the input. Words like "people" (19.73) and "like" (16.80) strongly influence predictions, capturing cumulative impact across multiple steps. |

**Model Performance Across Varying Thresholds**

We evaluated each model’s performance across thresholds (1 to 5) for toxic child comments to analyze precision, recall, and accuracy trade-offs in trigger classification. Metrics like Accuracy, Precision, Recall, F1-Score, and ROC-AUC provided insights into each model's sensitivity and specificity at different thresholds.

| **Threshold** | **Model**           | **Precision** | **Recall** | **F1-Score** | **Accuracy** | **ROC-AUC** |
|---------------|---------------------|---------------|------------|--------------|--------------|-------------|
| 1             | Neural Network      | 0.6085       | 0.5954     | 0.5980       | 0.5996       | 0.5996      |
|               | Logistic Regression | 0.6228       | 0.5933     | 0.6076       | 0.6104       | 0.6141      |
|               | Ridge Classifier    | 0.6224       | 0.5525     | 0.5696       | 0.5838       | 0.6046      |
|               | XGB Classifier      | 0.6302       | 0.5113     | 0.5611       | 0.6126       | 0.6031      |
| 2             | Neural Network      | 0.6474       | 0.6616     | 0.6494       | 0.6516       | 0.6137      |
|               | Logistic Regression | 0.6519       | 0.6081     | 0.6292       | 0.6387       | 0.6390      |
|               | Ridge Classifier    | 0.6622       | 0.5525     | 0.6597       | 0.6332       | 0.6341      |
|               | XGB Classifier      | 0.6519       | 0.5112     | 0.5097       | 0.6216       | 0.6291      |
| 3             | Neural Network      | 0.6474       | 0.6632     | 0.6494       | 0.6459       | 0.6467      |
|               | Logistic Regression | 0.6641       | 0.5341     | 0.6479       | 0.6395       | 0.6390      |
|               | Ridge Classifier    | 0.6950       | 0.6353     | 0.6157       | 0.6389       | 0.6592      |
|               | XGB Classifier      | 0.6889       | 0.5364     | 0.6494       | 0.6379       | 0.6516      |
| 4             | Neural Network      | 0.6474       | 0.6652     | **0.6743**   | **0.6678**   | 0.6516      |
|               | Logistic Regression | 0.6799       | 0.5637     | 0.6157       | 0.6389       | 0.6592      |
|               | Ridge Classifier    | 0.7852       | 0.4808     | 0.5171       | 0.6404       | 0.6349      |
|               | XGB Classifier      | 0.6889       | 0.5364     | 0.6028       | 0.6379       | 0.6208      |
| 5             | Neural Network      | 0.6545       | 0.5925     | 0.6243       | 0.6438       | 0.6137      |
|               | Logistic Regression | 0.6724       | 0.5467     | 0.6031       | 0.6316       | 0.6337      |
|               | Ridge Classifier    | 0.7244       | 0.4929     | 0.5039       | 0.6244       | 0.6251      |
|               | XGB Classifier      | 0.6724       | 0.5467     | 0.6031       | 0.6316       | 0.6337      |

The Neural Network performed best at threshold 4, achieving 66.78% accuracy with balanced precision and recall, making it the top performer for reliable trigger classification. While other models like Logistic Regression and XGB Classifier performed well at different thresholds, the Neural Network at threshold 4 proved optimal for accuracy, precision, and recall balance.

Using the Neural Network with optimized hyperparameters (learning rate: 0.0001, 128 hidden units, batch size: 64), we achieved stable performance (Precision: 0.6895, Recall: 0.6523, Accuracy: 0.6711). Applying Explainable AI further supported high accuracy and interpretability, providing valuable insights into harmful language patterns in Reddit comments, guiding ongoing refinement.

### 4.2 Discussion

#### 4.2.1 Limitations
In applying DeBERTa and BERTopic, current limitations arise in contextual understanding, especially for sarcasm, Singlish, and cultural nuances. This restricts accurate harmful classification, as sarcasm and culturally specific expressions may be misinterpreted. Additionally, using binary harmful and non-harmful labels simplifies complex emotional expressions, potentially overlooking subtleties that could yield more precise insights.

#### 4.2.2 Future Work
Future improvements aim to address these limitations. Fine-tuning DeBERTa for cultural and linguistic nuances will improve harmful comments recognition for the unique expressions in local dialects like Singlish. Shifting to multi-class classification will enable richer hate and toxicity insights beyond binary categories. Training an embedding model tailored to Singlish will facilitate more accurate word representations, while enhancing BERTopic’s topic modeling with LLMs or GenAI will allow more sophisticated topic detection and grouping.

### 4.3 Recommendations

| **Insight**                                | **Recommendation for MDDI**                                                                                                                                                                  | **Rationale**                                                                                                                                                                 |
|--------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Moderation Impact on Toxicity Levels**   | Encourage **Differentiated Moderation** by subreddit. Recommend stricter rules for high-toxicity spaces like r/SingaporeHappenings, implementing peak-time moderation for repeat offenders. | Toxicity levels are higher in unmoderated spaces, and stricter, targeted moderation can help reduce hateful comments.                                                        |
| **High-Volume Toxic Topics**               | **Topic-Specific Moderation** for subjects like politics and social issues. Suggest custom guidelines and stricter keyword-based monitoring for divisive topics.                             | Political and socio-economic discussions are highly toxic; custom guidelines can promote civil discourse.                                                                    |
| **Event-Driven Spikes in Toxicity**        | Implement **Event-Triggered Moderation Alerts** during significant events (e.g., elections, policy announcements). Recommend heightened monitoring by moderators during these times.        | Real-world events fuel toxic spikes; increased moderation during these periods can mitigate community harm.                                                                  |
| **Influence of Super-Spreaders**           | Suggest **Targeted Interventions** for frequent toxic commenters, such as probation periods, temporary posting restrictions, or behavioral nudges like community guideline reminders.       | A small group contributes disproportionately to toxicity. Targeted restrictions and reminders can curb the impact of repeat offenders.                                       |
| **Identifying Subtle Toxic Triggers**      | Promote **Pre-Comment Warnings** with predictive models to flag comments likely to incite toxicity. Display on-screen reminders to encourage rephrasing before posting.                     | Flagging potentially inciteful comments preemptively can prevent escalating conflicts, fostering a healthier community environment.                                          |
| **Trend of Rising Toxicity**               | Recommend **Ongoing Toxicity Tracking** to monitor evolving hate and toxicity trends. Advise MDDI to work with platforms to regularly track toxicity patterns and emerging trends.          | Continuous tracking allows MDDI to assess the impact of its policies and guide proactive interventions.                                                                      |
| **Community-Driven Moderation**            | Encourage **Platform and Community Partnership** by enabling subreddit moderators to work with MDDI, supported by data-driven insights on high-risk users and peak times.                   | Empowering community leaders enhances moderation effectiveness, especially when combined with platform insights on frequent offenders and high-risk periods.


These recommendations target key drivers of hate and toxicity while addressing both event-driven spikes and persistent user behavior patterns. By using data-driven insights, MDDI can help social media platforms reduce harmful content and foster healthier online discussions.