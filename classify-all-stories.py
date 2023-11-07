import os
from dotenv import load_dotenv
import json
import polars as pl
from joblib import Memory, Parallel, delayed
import openpipe
from tqdm import tqdm

load_dotenv()

print("Loading stories...")
stories = pl.read_parquet("data/stories.parquet")

memory = Memory("/workspace/cache", verbose=0)

openpipe.configure_openpipe(api_key=os.getenv("OPENPIPE_API_KEY"))
openpipe.openai.api_key = os.getenv("OPENAI_API_KEY")


@memory.cache
def classify_story_with_mistral(row):
    resp = openpipe.openai.ChatCompletion.create(
        model="openpipe:classify-hn-stories",
        messages=[
            {
                "role": "system",
                "content": "This is an HN headline and top comment. Your job is to determine whether the story is likely related to any of the given subjects. Multiple subjects may apply.",
            },
            {
                "role": "user",
                "content": f"Headline: {row['title']}\nURL: {row['url']}\nTop comment: {row['top_comment']}",
            },
        ],
        functions=[
            {
                "name": "classify",
                "description": "Important: only return true for a label if the story is likely related to the subject. If it isn't related, or if you aren't sure, return false.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ai_ml": {
                            "type": "boolean",
                            "description": "Related to AI, ML, AGI, etc.",
                        },
                        "crypto": {
                            "type": "boolean",
                            "description": "Related to crypto, blockchain, etc.",
                        },
                        "remote_work": {
                            "type": "boolean",
                            "description": "Related to remote work, WFH, etc.",
                        },
                        "rust": {
                            "type": "boolean",
                            "description": "Related to the Rust language.",
                        },
                    },
                    "requred": ["ai_ml", "crypto", "remote_work", "rust"],
                },
            }
        ],
        function_call={"name": "classify"},
    )
    json.loads(resp.choices[0].message.function_call.arguments)
    return resp


print("Classifying sample story...")
sample_class = classify_story_with_mistral(stories[0].to_dicts()[0])

print("Sample classification:")
print(sample_class)


def process_story(row):
    try:
        output = classify_story_with_mistral(row)

        row["tags"] = json.loads(
            output.choices[0].message["function_call"]["arguments"]
        )

        row["prompt_tokens"] = output.usage.prompt_tokens
        row["completion_tokens"] = output.usage.completion_tokens
        return row
    except Exception as e:
        print(e)
        return None


# Parallelize using joblib
results = Parallel(n_jobs=500)(
    delayed(process_story)(row) for row in tqdm(stories.rows(named=True))
)

print("Collating results")
results = [r for r in results if r is not None]

results = pl.DataFrame(results).unnest("tags")

print("Writing results")
results.write_parquet("data/stories-classified.parquet")

print("Done!")
