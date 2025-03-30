import json
from typing import List
import pandas as pd
import datasets
from faker import Faker
from src.utils import replace_placeholders_with_entities, generate_profiles, reconstruct_text, map_ent_tags_to_labels

_DESCRIPTION = ""

_HOMEPAGE = ""
    
_URLS_TO_DOWNLOAD = {
    "legal_questions": "./data/legal_questions_placeholders.jsonl",
    "medical_consultations": "./data/medical_consultations_placeholders.jsonl",
}

_ENT_TAGS = [
    'B-EMAIL',
    'B-ID_NUM', 
    'B-NAME',  
    'B-PHONE_NUM', 
    'B-ADDRESS', 
    'B-URL', 
    'B-USERNAME', 
    'I-EMAIL',
    'I-ID_NUM', 
    'I-NAME',  
    'I-PHONE_NUM', 
    'I-ADDRESS', 
    'I-URL', 
    'I-USERNAME', 
    'O',
]

_LICENSE = ""

_CITATION = ""

faker = Faker()
PII_ENT_FUNCS = {
    "EMAIL": [faker.ascii_email, faker.ascii_free_email],
    "NAME": [faker.name],
    "URL": [lambda: faker.uri(deep=1), lambda: faker.uri(deep=2)],
    "PHONE_NUM": [faker.phone_number],
    "ID_NUM": [
        # faker.doi, 
        faker.ripe_id, 
        faker.msisdn, 
        faker.ssn, 
        faker.sbn9, 
        faker.isbn10, 
        faker.isbn13, 
        faker.credit_card_number, 
        faker.aba, 
        faker.bban, 
        faker.iban
    ],
    "ADDRESS": [faker.street_address], 
    "USERNAME": [faker.user_name],
}


class SPYConfig(datasets.BuilderConfig):
    def __init__(self, faker_random_seed: int = 0, **kwargs):
        """BuilderConfig for SPY dataset."""
        super(SPYConfig, self).__init__(**kwargs)
        # self.random_seed = random_seed
        Faker.seed(faker_random_seed)

class SPY(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        SPYConfig(
            name="legal_questions",
            description="Legal questions domain",
            faker_random_seed=0
        ),
        SPYConfig(
            name="medical_consultations",
            description="Medical consultations domain",
            faker_random_seed=0
        ),
    ]
    DEFAULT_CONFIG_NAME = "legal_questions"

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "tokens": datasets.Sequence(feature=datasets.Value("string")),
                    "trailing_whitespaces": datasets.Sequence(feature=datasets.Value("bool")),
                    "labels": datasets.Sequence(
                        feature=datasets.features.ClassLabel(num_classes=len(_ENT_TAGS), names=_ENT_TAGS)
                    ),
                    "ent_tags": datasets.Sequence(feature=datasets.Value("string"))
                }
            ),
            supervised_keys=None,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        downloaded_files = {
            name: dl_manager.download(url) 
            for name, url in _URLS_TO_DOWNLOAD.items()
        }

        return [
            datasets.SplitGenerator(
                name=datasets.Split(name),
                gen_kwargs={"filepath": path}
            )
            for name, path in downloaded_files.items()
        ]

    def _generate_examples(self, filepath):      
        df = pd.read_json(filepath, lines=True)
        entities_df = generate_profiles(PII_ENT_FUNCS, len(df))

        df = replace_placeholders_with_entities(df, entities_df)
        df = map_ent_tags_to_labels(df, _ENT_TAGS)
        for key, (_, row) in enumerate(df.iterrows()):
            yield key, row.to_dict()