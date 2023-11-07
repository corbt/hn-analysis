from shared import serialize_input, deserialize_chat_output
import json
import polars as pl
from tqdm import tqdm
from vllm import LLM, SamplingParams, RequestOutput
import os
import logging


logging.info("Loading comments...")
comments = pl.read_ndjson("data/comments-to-classify.ndjson").sample(
    fraction=1, shuffle=True, seed=42
)

BATCH_SIZE = 10000

logging.info("Loading cached sentiments...")

tags = comments["tag"].unique().to_list()

if os.path.exists("data/comment-sentiments.json"):
    sentiments = json.load(open("data/comment-sentiments.json"))
else:
    sentiments = {tag: {} for tag in tags}

unlabeled_comments = comments

logging.info("Filtering out already classified comments...")

for tag in tags:
    keys = [int(k) for k in sentiments[tag].keys()]
    # print(tag)
    # print(len(sentiments[tag].keys()))
    # print(next(sentiments[tag].keys()))
    unlabeled_comments = unlabeled_comments.filter(
        (pl.col("tag").eq(tag) & pl.col("id").is_in(keys)).not_()
    )

logging.info(f"Classifying {len(unlabeled_comments)}/{len(comments)} comments")

logging.info("Loading model...")
model = LLM(
    model="OpenPipe/ft-development-604a4cf0-b954-4ea5-b5c4-2d13541f7d3e-classify-hn-comments-v4"
)
sampling_params = SamplingParams(temperature=0, max_tokens=20)


def parse_sentiment(output: RequestOutput):
    parsed = deserialize_chat_output(output.outputs[0].text)
    try:
        return json.loads(parsed["function_call"]["arguments"])["sentiment"]
    except Exception as e:
        logging.info(e)
        return None


def classify_batch(i, batch_size):
    batch = unlabeled_comments.slice(i, batch_size)

    inputs = [serialize_input(row["input"]) for row in batch.to_dicts()]
    outputs = model.generate(inputs, sampling_params=sampling_params)

    batch_sentiments = [parse_sentiment(output) for output in outputs]

    for i in range(len(batch)):
        sentiments[batch["tag"][i]][batch["id"][i]] = batch_sentiments[i]
    #     if batch_sentiments[i] == "positive":
    #         print("Tag:", batch["tag"][i])
    #         print("Comment:", batch["input"][i])
    #         print(f"Link: https://news.ycombinator.com/item?id={batch['id'][i]}")

    # exit()


for i in tqdm(range(0, len(unlabeled_comments), BATCH_SIZE)):
    classify_batch(i, BATCH_SIZE)

    with open("data/comment-sentiments.json", "w") as f:
        json.dump(sentiments, f)

logging.info("Finished!")
