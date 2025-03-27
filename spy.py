import json
from typing import List

import datasets

_DESCRIPTION = ""

_HOMEPAGE = ""
    
_URLS_TO_DOWNLOAD = {
    "legal_questions": "./data/legal_questions_all.jsonl",
    "medical_consultations": "./data/medical_consultations_all.jsonl",
}

_ENT_TAGS = [
    'B-EMAIL',
    'B-ID_NUM', 
    'B-NAME_STUDENT',  
    'B-PHONE_NUM', 
    'B-STREET_ADDRESS', 
    'B-URL_PERSONAL', 
    'B-USERNAME', 
    'I-EMAIL',
    'I-ID_NUM', 
    'I-NAME_STUDENT',  
    'I-PHONE_NUM', 
    'I-STREET_ADDRESS', 
    'I-URL_PERSONAL', 
    'I-USERNAME', 
    'O',
]

_LICENSE = ""

_CITATION = ""


class SPYConfig(datasets.BuilderConfig):
    def __init__(self, random_seed: int = 0, **kwargs):
        """BuilderConfig for SPY dataset."""
        super(SPYConfig, self).__init__(**kwargs)
        self.random_seed = random_seed

class SPY(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        SPYConfig(
            name="legal_questions",
            description="Legal questions domain",
            random_seed=0
        ),
        SPYConfig(
            name="medical_consultations",
            description="Medical consultations domain",
            random_seed=0
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
                    "trailing_whitespace": datasets.Sequence(feature=datasets.Value("bool")),
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
        with open(filepath, encoding="utf-8") as f:
            for key, row in enumerate(f):
                data = json.loads(row)
                yield key, {
                    **data,
                    "ent_tags": data["labels"]
                }