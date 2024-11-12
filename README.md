# DSA4264 Problem 2
### Hate Speech Detection and Analysis

---

## Setup

To get started with this project, you'll need to set up your development environment with the necessary libraries and dependencies. Follow the steps below to install the required packages and configure your environment.

### 1. Clone the Repository

If this project is hosted in a Git repository, start by cloning the repository to your local machine:


### 2. Create a Virtual Environment and Install Dependencies

To keep dependencies isolated and ensure compatibility, create a virtual environment named `.venv` and install the required packages from `requirements.txt` as follows:

1. **Create the Virtual Environment**  
   Run the following command to create a virtual environment in the root directory:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate #for macOS/Linux
   .venv\Scripts\activate #for windows

   pip install -r requirements.txt # to install libraries needed
### 3. Necessary Data

To run all the code in this project, please ensure you have a folder named `data` in the root directory of this repository. This folder should contain the following files:

- **Reddit Threads:**
  - `Reddit-Threads_2020-2021.csv`
  - `Reddit-Threads_2022-2023.csv`

- **Labeled Data:**
  - `labeled_data_2.csv`
  - `deberta_v3_labelled_3_1.csv`
  - `deberta_v3_labelled_3_2.csv`
  - `deberta_v3_labelled_3_3.csv`
  - `deberta_v3_labelled_3_4.csv`
  - `deberta_v3_labelled_3_5.csv`

- **Further Cleaned Data:**
  - `Data_1.csv`
  - `Data_2.csv`
  - `Data_3.csv`
  - `Data_4.csv`
  - `Data_5.csv`
  - `Data_6.csv`
  - `Data_7.csv`
  - `deberta_v3_labelled_3_ALL.csv`

- **Parquet Files:**
  - `glenn_and_sy.parquet`
  - `hate_toxic_topical_comments.parquet`
  - `post.parquet`

Ensure all these files are in the `data` folder to avoid issues when running the code.

### 4. .env API Keys

To access various APIs used in this project, create a `.env` file in the root directory of this repository and include the following keys:

- **Hugging Face API Key**  
  `HUGGINGFACE_API_KEY`

- **OpenAI API Key**  
  `OPENAI_API_KEY`

- **Claude API Key**  
  `CLAUDE_API_KEY`

- **Perspective API Key**  
  `PERSPECTIVE_API_KEY`

- **Reddit API Credentials**  
  - `REDDIT_CLIENT_ID`
  - `REDDIT_CLIENT_SECRET`

---

These keys are essential for accessing the respective APIs. Make sure to obtain the keys from each service provider and keep the `.env` file secure and private.


### 5. Model for Dashboard

To use the fine-tuned model in the dashboard, please contact `mibs862` to obtain the `neural_net_threshold_4.pth` and `vectorizer.pkl` files. Once you have it, place the files inside the `dashboard-reddit` directory.

---

This model is required for the dashboard functionality. Ensure it is placed correctly to avoid issues when running the dashboard.
