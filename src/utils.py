from tqdm import tqdm
from faker import Faker
from collections import defaultdict
import pandas as pd
import random


def split_sizes(lst_length, n):
    return [lst_length // n + (i < lst_length % n) for i in range(n)]

def gen_n_random_samples(funcs, n):
    generated_samples = []
    chunks_sizes = split_sizes(n, len(funcs))
    for func, chunk_size in zip(funcs, chunks_sizes):
        generated_samples = generated_samples + [func() for _ in range(chunk_size)]
    random.shuffle(generated_samples)
    return generated_samples

def generate_profiles(funcs_dict: dict, n: int):
    profiles = {ent_type: gen_n_random_samples(funcs, n) for ent_type, funcs in funcs_dict.items()}
    return pd.DataFrame().from_dict(profiles)

# def replace_placeholders_with_entities(row, entities_dict):
#     tokens = row['tokens']
#     ent_tags = row['ent_tags']
#     trailing_whitespaces = row['trailing_whitespaces']

#     modified_tokens = []
#     modified_tags = []
#     modified_whitespaces = []

#     for token, tag, whitespace in zip(tokens, ent_tags, trailing_whitespaces):
#         if tag.startswith("B-"):
#             entity_type = tag[2:]
#             entity_value = entities_dict.get(entity_type, "")

#             if entity_value:
#                 entity_words = entity_value.split()

#                 for i, word in enumerate(entity_words):
#                     modified_tokens.append(word)
#                     modified_tags.append(f"B-{entity_type}" if i == 0 else f"I-{entity_type}")
#                     modified_whitespaces.append(True if i < len(entity_words) - 1 else whitespace)
#             else:
#                 modified_tokens.append(token)
#                 modified_tags.append(tag)
#                 modified_whitespaces.append(whitespace)
#         else:
#             modified_tokens.append(token)
#             modified_tags.append(tag)
#             modified_whitespaces.append(whitespace)

#     return {
#         'tokens': modified_tokens,
#         'ent_tags': modified_tags,
#         'trailing_whitespaces': modified_whitespaces
#     }


def replace_placeholders_with_entities(df, entities_df):
    new_data = []

    for index, row in df.iterrows():
        tokens = row['tokens']
        ent_tags = row['ent_tags']
        trailing_whitespaces = row['trailing_whitespaces']

        # Use the corresponding row in entities_df
        entity_row = entities_df.iloc[index]
        entities_dict = {col: entity_row[col].split() for col in entities_df.columns}

        modified_tokens = []
        modified_tags = []
        modified_whitespaces = []

        for token, tag, whitespace in zip(tokens, ent_tags, trailing_whitespaces):
            if tag.startswith("B-"):
                entity_type = tag[2:]
                entity_words = entities_dict.get(entity_type, [token])

                for i, word in enumerate(entity_words):
                    modified_tokens.append(word)
                    modified_tags.append(f"B-{entity_type}" if i == 0 else f"I-{entity_type}")
                    modified_whitespaces.append(True if i < len(entity_words) - 1 else whitespace)
            else:
                modified_tokens.append(token)
                modified_tags.append(tag)
                modified_whitespaces.append(whitespace)

        new_data.append({
            'tokens': modified_tokens,
            'ent_tags': modified_tags,
            'trailing_whitespaces': modified_whitespaces
        })

    return pd.DataFrame(new_data)

def reconstruct_text(df):
    new_data = []
    for _, row in df.iterrows():
        tokens = row['tokens']
        trailing_whitespaces = row['trailing_whitespaces']
        text = "".join(token + (" " if whitespace else "") for token, whitespace in zip(tokens, trailing_whitespaces)).strip()
        
        new_data.append({
            **row.to_dict()
            'text': text
        })

    return pd.DataFrame(new_data)


def map_ent_tags_to_labels(df, ent_tags):
    label_map = {tag: index for index, tag in enumerate(ent_tags)}
    new_data = []

    for _, row in df.iterrows():
        labels = [label_map.get(tag, -1) for tag in row['ent_tags']]
        new_data.append({
            **row.to_dict()
            'labels': labels
        })

    return pd.DataFrame(new_data)