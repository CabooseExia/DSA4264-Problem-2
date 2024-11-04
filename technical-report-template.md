# Technical Report

**Project: DSA4264 Problem 2**  
**Members:**  
**Bae Soo Youn, A0237****, e077****@u.nus.edu**  
**Ethan Loh Zhi Kai, A0233****, e072****@u.nus.edu**  
**Muhammad Irfan Bin Salleh, A0234****, e072****@u.nus.edu**  
**Wee Wei Kit, Glenn, A0234****, e072****@u.nus.edu**  
**Wong Si Yuan, A0239****, e077****@u.nus.edu**  

<!-- i'm scared this counts as sensetive data. we need to edit this b4 submitting -->

Last updated on 05/11/2024

## Section 1: Context
There is a growing concern of toxic and hateful content on social media platforms, particularly within Singapore-related subreddits on Reddit. We are to report our findings to stakeholders, especially within the Ministry of Digital Development and Innovation (MDDI), which oversees online trust and safety. A significant portion of this concern is aimed at protecting vulnerable groups, such as children, who might be exposed to such content.

The Online Trust and Safety department within MDDI seeks to understand the trend's trajectory and underlying causes. Through a computational analysis of Reddit comments, the team aims to quantify the increase in toxic discourse and identify key drivers behind it. This research is essential to inform policy actions and collaborations with social media companies to curb the spread of hateful content. However, limitations include restricted access to original thread texts and a requirement for confidentiality in handling the provided data. Success in this endeavor would mean actionable insights that help shape policy recommendations to foster safer online environments.

<!-- *In this section, you should explain how this project came about. Retain all relevant details about the project’s history and context, especially if this is a continuation of a previous project.*

*If there are any slide decks or email threads that started before this project, you should include them as well.* -->

## Section 2: Scope

### 2.1 Problem
The business problem focuses on understanding the nature and potential increase of hate and toxicity in Singapore-related subreddits on Reddit. The Online Trust and Safety department within the MDDI is concerned about whether these negative elements are rising, as it could hinder their mission to ensure safe digital spaces for Singaporean users. Given that social media platforms are often breeding grounds for extreme viewpoints and harmful rhetoric, this issue directly impacts MDDI’s ability to create a healthier online environment.

The significance of this problem lies in its potential impact on social cohesion. In a multi-ethnic and multi-religious country like Singapore, an increase in toxic discourse can lead to social fragmentation and heightened community tensions. Metrics that could help quantify this impact include the proportion of users exposed to or affected by hateful content, alongside the frequency of reports related to online harassment. Addressing these issues proactively could mitigate the risk of further societal polarization.

Data science and machine learning offer the most feasible solution due to the large volume of Reddit data involved. Manual examination of comments for hatefulness or toxicity is impractical; however, Natural Language Processing (NLP) techniques can facilitate efficient, large-scale analysis to detect trends and measure sentiment. This approach enables MDDI to determine if toxicity levels are indeed rising and to uncover underlying causes. Machine learning models could further identify patterns in content and user interactions, providing MDDI with actionable insights that support informed policy-making and collaboration with social media platforms.

<!-- *In this subsection, you should explain what is the key business problem that you are trying to solve through your data science project. You should aim to answer the following questions:*

* *What is the problem that the business unit faces? Be specific about who faces the problem, how frequently it occurs, and how it affects their ability to meet their desired goals.*
* *What is the significance or impact of this problem? Provide tangible metrics that demonstrate the cost of not addressing this problem.*
* *Why is data science / machine learning the appropriate solution to the problem?* -->

### 2.2 Success Criteria
For this data science project to be successful, it will need to meet key business goals that support the Online Trust and Safety department's objectives in ensuring a safer digital environment for Singaporean users on Reddit. 

**Business Goals:**
1. **Insight into Toxicity Trends:** A primary business goal is to gain clear, data-driven insights into the levels of hate and toxicity within Singapore-related subreddits. By understanding whether toxicity is rising, stable, or declining, MDDI can tailor policy recommendations and interventions to social media platforms more effectively. Success here will be measured by the ability to quantify the extent and trend of toxic content, providing actionable insights that can directly inform policy decisions and collaborative initiatives with social media companies.

2. **Identification of Key Drivers of Toxicity:** Another business goal is to identify the main factors contributing to hate and toxicity on these platforms. This includes determining whether specific topics, user demographics, or external events correlate with higher levels of toxic discourse. Success will be defined by the ability to pinpoint and explain significant drivers of hateful content, enabling targeted measures to address these root causes.


<!-- *In this subsection, you should explain how you will measure or assess success for your data science project. You need to specify at least 2 business and/or operational goals that will be met if this project is successful. Business goals directly relate to the business’s objectives, such as reduced fraud rates or improved customer satisfaction. Operational goals relate to the system’s needs, such as better reliability, faster operations, etc.* -->

### 2.3 Assumptions
Our team assumed that we are experts at recognising hate and toxic comments. (help)

<!-- *In this subsection, you should set out the key assumptions for this data science project that, if changed, will affect the problem statement, success criteria, or feasibility. You do not need to detail out every single assumption if the expected impact is not significant.*

*For example, if we are building an automated fraud detection model, one important assumption may be whether there is enough manpower to review each individual decision before proceeding with it.* -->

## Section 3: Methodology

### 3.1 Technical Assumptions
Defining Hate and Toxicity: For this project, we define hate as language or content that intentionally demeans or discriminates against an individual or group based on characteristics such as race, religion, gender, or other protected attributes. Toxicity refers to hostile, inflammatory, or harmful language that may provoke or offend, even if it does not directly target a specific group.These definitions are essential to distinguish between general negative sentiment and content that is considered harmful by moderation standards. Furthermore, as the reccomendations at the end would apply to deal with both toxic and hateful comments in general, we opted to merge the 2 into one feature. 


#### Data Features
The data features available in the dataset include:
- **text:** The content of the Reddit comment.
- **timestamp:** The date and time the comment was posted.
- **username:** The username of the person who posted the comment.
- **link:** URL to the Reddit thread where the comment was posted.
- **link_id, parent_id, id, subreddit_id:** Metadata identifiers for the comment, parent comment, link, and subreddit.
- **moderation:** Moderation information, including details such as whether the comment was flagged as controversial.


#### Computational Resources
For this project, we have access to the following computational resources:
1. Five Personal Laptops
2. Google Colab
3. An old PC with a RTX 2070 super
4. Some money for API calls


#### Key Hypotheses
1. **Increasing Toxicity Over Time:** The volume and intensity of toxic and hateful comments in Singapore-related subreddits have increased over the past few years. This hypothesis can be tested by analyzing trends in toxicity levels over time.

2. **Correlation with Specific Topics or Events:** Certain topics, such as political or social issues, are more likely to contain toxic or hateful content. This hypothesis could be tested by examining the frequency of toxicity within comments on these topics or around major events.

*In this subsection, you should set out the assumptions that are directly related to your model development process. Some general categories include:*
3. **User Behavior and Recurrence:** A small group of users may disproportionately contribute to toxic and hateful content. Testing this hypothesis would involve analyzing user-level data to determine if certain users have a recurring pattern of posting harmful content.

4. **Impact of Moderation Features on Toxicity Levels:** Moderated comments may have different toxicity levels compared to unmoderated comments. This hypothesis could be examined by comparing toxicity across moderated and unmoderated comments.

5. **Effect of Time of Day on Toxicity:** Toxic or hateful comments may be more prevalent at certain times of day. This hypothesis could be explored by analyzing toxicity trends in relation to posting times. (idea)

6. **Trigger comments:** Certain comments might trigger more toxic comments. Analysing these comments could lead us to determine the root cause of toxicity.


#### Data Quality
The dataset is mostly complete, with key features such as text, timestamp, and user metadata well-represented. However, the **moderation** feature contains significant gaps, with missing data across many entries. This missing information may limit the analysis of moderation’s impact on toxicity levels, as incomplete moderation data reduces our ability to compare toxic comments against moderation actions. Despite this limitation, other core features are sufficiently complete for meaningful analysis.

<!-- *In this subsection, you should set out the assumptions that are directly related to your model development process. Some general categories include:*
* *How to define certain terms as variables*
* *What features are available / not available*
* *What kind of computational resources are available to you (ie on-premise vs cloud, GPU vs CPU, RAM availability)*
* *What the key hypotheses of interest are*
* *What the data quality is like (especially if incomplete / unreliable)* -->

### 3.2 Data

#### Data collection
we collected the data from Shuan Khoo's google link. 


#### Data Cleaning
For the initial part of the project, we undertook the task of labeling the text data as either *toxic/hateful* or *not toxic/not hateful*. During this process, we removed rows where the text content was `[removed]`, `[deleted]`, or `NaN`, as these entries lacked meaningful information for toxicity and hatefulness analysis. This step ensured that the remaining dataset contained only relevant text data for accurate labeling and analysis.


#### Features
We labelled the data as `True` or `False` to indicate whether each comment was *toxic* or *hateful* (True for toxic/hateful, False for not toxic/not hateful). These binary labels serve as the target variables for analyzing patterns and trends in harmful content.


### Data Splitting
We did not train any models, so there was no train-test split applied to the dataset. However, we used a subset of 300 rows as a validation set to help choose the best model for labeling our data. This validation set enabled us to evaluate model performance and select the most accurate approach for labeling toxic and hateful content.


<!-- *In this subsection, you should provide a clear and detailed explanation of how your data is collected, processed, and used. Some specific parts you should explain are:*
* *Collection: What datasets did you use and how are they collected?*
* *Cleaning: How did you clean the data? How did you treat outliers or missing values?*
* *Features: What feature engineering did you do? Was anything dropped?*
* *Splitting: How did you split the data between training and test sets?* -->

### 3.3 Experimental Design
For labeling the data, we tested various models and used the F1-score as the primary metric to determine the best model for the task. To validate the models, the team labeled 300 challenging texts—entries we anticipated might be harder for the models to classify accurately. The highest-performing model was OpenAI's **gpt-4o-mini**, achieving an F1-score of 0.77. However, due to limited funds and time constraints, we opted to use **sileod/deberta-v3-base-tasksource-toxicity** for labeling the entire dataset. This model provided a relatively quick labeling process with a reasonable F1-score of 0.68.



<!-- *In this subsection, you should clearly explain the key steps of your model development process, such as:*
* *Algorithms: Which ML algorithms did you choose to experiment with, and why?*
* *Evaluation: Which evaluation metric did you optimise and assess the model on? Why is this the most appropriate?*
* *Training: How did you arrive at the final set of hyperparameters? How did you manage imbalanced data or regularisation?* -->

## Section 4: Findings

### 4.1 Results


Singapore is doomed. Recommend calling Poseidon to cleanse the land. Get in the water. 

<!-- *In this subsection, you should report the results from your experiments in a summary table, keeping only the most relevant results for your experiment (ie your best model, and two or three other options which you explored). You should also briefly explain the summary table and highlight key results.*

*Interpretability methods like LIME or SHAP should also be reported here, using the appropriate tables or charts.* -->

### 4.2 Discussion


<!-- *In this subsection, you should discuss what the results mean for the business user – specifically how the technical metrics translate into business value and costs, and whether this has sufficiently addressed the business problem.*

*You should also discuss or highlight other important issues like interpretability, fairness, and deployability.* -->

### 4.3 Recommendations


<!-- *In this subsection, you should highlight your recommendations for what to do next. For most projects, what to do next is either to deploy the model into production or to close off this project and move on to something else. Reasoning about this involves understanding the business value, and the potential IT costs of deploying and integrating the model.*

*Other things you can recommend would typically relate to data quality and availability, or other areas of experimentation that you did not have time or resources to do this time round.* -->