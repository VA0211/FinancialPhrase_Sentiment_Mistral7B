import pandas as pd
from sklearn.model_selection import train_test_split

def generate_prompt(data_point):
    return f"""
            [INST]Analyze the sentiment of the news headline enclosed in square brackets, 
            determine if it is positive, neutral, or negative, and return the answer as 
            the corresponding sentiment label "positive" or "neutral" or "negative"[/INST]

            [{data_point["text"]}] = {data_point["sentiment"]}
            """.strip()

def generate_test_prompt(data_point):
    return f"""
            [INST]Analyze the sentiment of the news headline enclosed in square brackets, 
            determine if it is positive, neutral, or negative, and return the answer as 
            the corresponding sentiment label "positive" or "neutral" or "negative"[/INST]

            [{data_point["text"]}] = """.strip()

def generate_dataset(file_path='sentiment-analysis-for-financial-news/all-data.csv', 
                     train_size=275,
                     val_size=50,
                     test_size=275):

    df = pd.read_csv(file_path, 
                     names=['sentiment', 'text'],
                     encoding='utf-8', 
                     encoding_errors='replace')

    # Initialize lists to hold DataFrames for train and test
    X_train = list()
    X_test = list()

    for sentiment in ["positive", "neutral", "negative"]:
        # Sample for training and testing sets
        train, test = train_test_split(df[df.sentiment == sentiment], 
                                       train_size=train_size,
                                       test_size=test_size, 
                                       random_state=42)

        # Append the samples to the lists
        X_train.append(train)
        X_test.append(test)

    # Concatenate all sentiment DataFrames into a single train and test set
    X_train = pd.concat(X_train).sample(frac=1, random_state=42).reset_index(drop=True)
    X_test = pd.concat(X_test).reset_index(drop=True)

    # Create a combined DataFrame for checking duplicates
    combined_df = pd.concat([X_train, X_test]).drop_duplicates(subset=['text'], keep=False)

    # Create an evaluation set
    # Filter the original DataFrame to exclude the texts in X_train and X_test
    eval_df = df[~df['text'].isin(combined_df['text'])]

    # Sample the evaluation set
    X_eval = (eval_df
              .groupby('sentiment', group_keys=False)
              .apply(lambda x: x.sample(n=val_size, 
                                         random_state=10, 
                                         replace=False)))  # No replacement to avoid duplicates

    # Return the final DataFrames
    y_true = X_test.sentiment
    return X_train.reset_index(drop=True), X_eval.reset_index(drop=True), X_test.reset_index(drop=True), y_true
