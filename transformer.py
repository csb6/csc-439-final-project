# Author: Cole Blakley
# Description: This program builds/runs models to predict the state that Tweets
#  originate from based solely on their text. The baseline is a logistic regression
#  model using unigrams, while the primary model is a DistilBERT model, fine-tuned
#  for the Twitter dataset.
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, \
  DataCollatorWithPadding, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, Features, load_metric
import datasets
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
import re, itertools, os, time, os.path, sys
from tzwhere import tzwhere

os.environ["TZ"] = "US/Pacific"
time.tzset()

tz = tzwhere.tzwhere()

# These are indices into the parsed rows of data read from the text file of Tweets
#Row 0 - User Data
# The user's Twitter handle
Username = 0
# The location of the user (no specific format, could be: "Rome, Italy", "LA", "somewhere", etc.)
User_Location = 3
# The timezone of the user
User_Timezone = 6

#Row 1 - Post Data
# The time that the post was made (e.g. "Wed Oct 02 16:11:47 PDT 2013")
# Always in PDT or MST.
Post_Time = 0
# The timezone of the post. In the dataset, this is represented as lat/lon coordinates,
# which we then normalize to a timezone name
Post_Timezone = 1

# Split into an array, replacing all NIL values with None
def convert_to_row(line):
    result = []
    for item in line.strip().split("\t"):
        if item == "NIL":
            result.append(None)
        else:
            result.append(item)
    return result

# Converts lat/lon coordinates to a timezone name
def to_timezone(lat_long_str):
    if not lat_long_str:
        return None
    try:
        latitude, longitude = [float(n) for n in lat_long_str.split("|", 2)]
        return tz.tzNameAt(latitude, longitude)
    except:
        print("Error converting lat/lon:", lat_long_str, file=sys.stderr)
    return None

# Build dictionary mapping state abbreviations -> list of regex patterns for
# matching variations of that state's name
def load_state_patterns(path):
    state_patterns = {}
    with open(path) as state_file:
        for line in state_file:
            bits = line.split("\t")
            abbrev = bits[0].strip()
            pattern_list = []
            for i in range(1, len(bits)):
                core = bits[i].strip()
                pattern_list.append(re.compile(r"\s+" + core + r"\s*$", re.I))
                pattern_list.append(re.compile(r"^" + core + r"\s*((,\s*)?" \
                                               + abbrev + r")?$", re.I))
            state_patterns[abbrev] = pattern_list
    return state_patterns

state_patterns = load_state_patterns("states.txt")

USA_Suffix = re.compile("([\\s*,]?\\s*USA?\\s*$)|([\\s*,]?\\s*united\\s*states\\s*(of\\s*america)?\\s*$)", re.I)

# Given a user location (an unformatted string) and optionally a timezone,
# attempt to convert the location to a U.S. state abbreviation
def to_us_state_abbrev(location, timezone):
    location = location.lower()
    # Remove the suffix "USA", "U.S.", etc. from the location
    match = USA_Suffix.search(location)
    if match:
        location = location[:match.start()]

    if location == "la":
        # Disambiguate between Los Angeles and Louisiana
        if not timezone:
            return None
        elif "pacific" in timezone:
            return "CA"
        elif "central" in timezone:
            return "LA"
        else:
            return None

    for abbrev, pattern_list in state_patterns.items():
        for pattern in pattern_list:
            if pattern.search(location):
                return abbrev
    return None

# Attempts to match a timezone to a normalized timezone string
def to_us_tz_abbrev(name):
    if not name:
        return None
    elif "/" not in name:
        state_name = name
    else:
        state_name = name.split("/")[1]

    for abbrev, pattern_list in state_patterns.items():
        for pattern in pattern_list:
            if pattern.search(state_name):
                return abbrev
    return None

# Converts a lat/lon position into a U.S. timezone
def to_us_timezone(timezone_str):
    return to_us_tz_abbrev(to_timezone(timezone_str))

# Iterator that yields user_metadata, tweet_metadata, tweet with some cleanup.
def tweet_iter(input_file, count=-1):
    if count < 0:
        counter = itertools.repeat(None)
    else:
        counter = range(count*3)

    for _ in counter:
        user_metadata_line = input_file.readline()
        tweet_metadata_line = input_file.readline()
        tweet = input_file.readline().strip()
        if tweet == "":
            print("EOF", file=sys.stderr)
            return

        user_metadata = convert_to_row(user_metadata_line)
        tweet_metadata = convert_to_row(tweet_metadata_line)

        if len(user_metadata) <= User_Location or len(user_metadata) <= User_Timezone:
            # Skip garbled UTF-8 lines
            input_file.readline()
            input_file.readline()
            continue

        # Try to identify a U.S. timezone
        timezone = None
        if tweet_metadata[Post_Timezone]:
            tweet_metadata[Post_Timezone] = to_us_timezone(tweet_metadata[Post_Timezone])
            timezone = tweet_metadata[Post_Timezone]
        if user_metadata[User_Timezone]:
            user_metadata[User_Timezone] = user_metadata[User_Timezone].lower()
            timezone = user_metadata[User_Timezone]

        # Try to normalize the user's provided location to a state abbreviation
        if user_metadata[User_Location]:
            user_metadata[User_Location] = to_us_state_abbrev(user_metadata[User_Location],
                                                              timezone)

        if not user_metadata[User_Location]:
            # If we can't localize the Tweet to a U.S. state, then we can't use it
            continue

        try:
            time_str = tweet_metadata[Post_Time]
            # Remove timezone code
            time_str = time_str[:-8] + time_str[-4:]
            tweet_metadata[Post_Time] = time.mktime(time.strptime(
                time_str, "%a %b %d %H:%M:%S %Y"))
        except ValueError as err:
            print(err, file=sys.stderr)
            continue

        yield user_metadata, tweet_metadata, tweet

f1_metric = load_metric("f1")
bootstrap_metric = load_metric("bootstrap_resample")

# Get measures indicating performance
def compute_metrics(preds):
    logits, labels = preds
    predictions = numpy.argmax(logits, axis=-1)
    f1_score = f1_metric.compute(predictions=predictions, references=labels, average="micro")["f1"]
    return {"f1": f1_score}

# tokens that start with punctuation are not useful for baseline classifier,
# so discard them to reduce noise
punct = re.escape("-+,.?!\"'{}[]_&^();:/\\…")
unwanted_token_pattern = re.compile(f"^[{punct}]+")

# Filter tokens for baseline model
def filter_tokens(tokens):
    result = []
    for token in tokens:
        if token.startswith("http"):
            # Many tweets end with a URL of some kind, so ignore this part of tweets
            break
        elif unwanted_token_pattern.match(token):
            continue
        else:
            result.append(token)
    return result

# If not written to disk already, need to generate the dataset by reading/tokenizing the
# Tweet data.
def build_dataset(tweet_file_path, tokenizer):
    # Tweet Dataset:
    #  Features:
    #   text: the text of the Tweet
    #  Labels:
    #   the state abbreviation
    data = {"text": [], "labels": []}
    data_by_state = {}
    with open(tweet_file_path) as tweet_file:
        print("Gathering training data...")
        for user_metadata, tweet_metadata, tweet in tweet_iter(tweet_file):
            state_label = user_metadata[User_Location]
            assert(state_label)
            if state_label not in data_by_state:
                data_by_state[state_label] = [tweet]
            else:
                data_by_state[state_label].append(tweet)
    for state_label, state_tweets in data_by_state.items():
        i = 0
        while i < len(state_tweets):
            data["text"].append(state_tweets[i])
            data["labels"].append(state_label)
            i += 1

    label_set = set(data["labels"])
    print(f"  Label set: {label_set}")

    print("Building/tokenizing dataset...")
    dataset = Dataset.from_dict(data,
                                features=Features(
                                    {"text": datasets.Value("string"),
                                     "labels": datasets.ClassLabel(names=list(label_set))}
                                ))
    tokenized_dataset = dataset.map(lambda samples: tokenizer(samples["text"], truncation=True),
                                    batched=True)

    return len(label_set), tokenized_dataset

# Runs/tunes (if needed)/evaluates a DistilBERT model
def run_distilbert_model(tokenizer, data_collator, label_count,
                         train_dataset, dev_dataset, test_dataset,
                         eval_test=False):
    args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01
    )

    needs_training = False
    if not os.path.exists("./model"):
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased",
                                                                    num_labels=label_count)
        needs_training = True
    else:
        model = DistilBertForSequenceClassification.from_pretrained("./model")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    if needs_training:
        print("Training model...")
        trainer.train()
        print("Saving model to disk...")
        try:
            os.mkdir("./model")
        except FileExistsError:
            print("./model directory already exists")
        model.save_pretrained("./model")
    #print("Evaluating model on dev dataset...")
    #print("DistilBERT f1 score (Development):", trainer.evaluate())
    if eval_test:
        print("Evaluating model on test dataset")
        embeddings, predictions, metrics = trainer.predict(test_dataset)
    return test_dataset.features["labels"].int2str(predictions), metrics

def dummy(doc):
    return doc

# For each sample in the dataset, converts the tokenized samples back into
# strings and labels, cleaning up tokens to be more suitable for a unigram model
def gather_tweets_and_labels(tokenizer, dataset):
    tweets = []
    labels = []
    for sample in dataset:
        tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"])
        tweets.append(filter_tokens(tokens))
        labels.append(sample["labels"])
    return tweets, labels

# Runs and evaluates a simple unigram logisitic regression model using tf values as features.
def run_baseline_model(tokenizer, data_collator, train_dataset, dev_dataset, test_dataset,
                       eval_test=False):
    print("Converting/filtering tokens of tweets...")
    print(" Train data...")
    train_tweets, y_train = gather_tweets_and_labels(tokenizer, train_dataset)
    print(" Dev data...")
    dev_tweets, y_dev = gather_tweets_and_labels(tokenizer, dev_dataset)
    if eval_test:
        print(" Test data...")
        test_tweets, y_test = gather_tweets_and_labels(tokenizer, test_dataset)

    print("Gathering tf counts...")
    tf_vectorizer = CountVectorizer(preprocessor=dummy, tokenizer=dummy)
    X_train = tf_vectorizer.fit_transform(train_tweets)
    X_dev = tf_vectorizer.transform(dev_tweets)
    if eval_test:
        X_test = tf_vectorizer.transform(test_dataset)

    print("Training baseline model...")
    model = LogisticRegression(C=20.0, max_iter=1000)
    model.fit(X_train, y_train)

    y_dev_pred = model.predict(X_dev)
    print("Baseline f1 score (development):", f1_score(y_dev, y_dev_pred, average="micro"))

    if eval_test:
        y_test_pred = model.predict(X_test)
        print("Baseline f1 score (test):", f1_score(y_test, y_test_pred, average="micro"))
        return y_test_pred, y_test

def main():
    eval_test = True

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if not os.path.exists("./dataset"):
        os.mkdir("./dataset")
        label_count, dataset = build_dataset("2020-food-tweets.txt", tokenizer)
        print(f"{label_count} labels")
        print("Saving dataset to disk...")
        dataset.save_to_disk("./dataset")
    else:
        label_count = 51
        dataset = Dataset.load_from_disk("./dataset")
    print("Splitting dataset...")
    # 0.1 for test data; 0.9 for train+dev
    train_test_split_dataset = dataset.train_test_split(seed=41, test_size=0.1)
    test_dataset = train_test_split_dataset["test"]
    # 0.1 for dev data; 0.8 for train
    train_dev_split_dataset = train_test_split_dataset["train"].train_test_split(seed=41,
                                                                                 test_size=0.1)
    dev_dataset = train_dev_split_dataset["test"]
    train_dataset = train_dev_split_dataset["train"]
    print(" # of Tweets in test data:", len(test_dataset))
    print(" # of Tweets in train data:", len(train_dataset))
    print(" # of Tweets in dev data:", len(dev_dataset))

    predictions, metrics = run_distilbert_model(tokenizer, data_collator, label_count,
                                                train_dataset, dev_dataset, test_dataset,
                                                eval_test)
    print(metrics)
    baseline_predictions = run_baseline_model(tokenizer, data_collator,
                                              train_dataset, dev_dataset, test_dataset,
                                              eval_test)
    if eval_test:
        baseline_labels, true_labels = baseline_predictions
        print("DistilBERT preds:", baseline_labels[:10])
        print("Baseline preds:", predictions[:10])
        print("True preds:", true_labels[:10])
        p_value = bootstrap_metric.compute(predictions=test_dataset.features["labels"].str2int(predictions),
                                           references=test_dataset.features["labels"].str2int(baseline_labels),
                                           true_labels=test_dataset.features["labels"].str2int(true_labels))
        print("p (compared to baseline model on test set):", p_value)

main()
