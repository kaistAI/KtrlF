import pickle
import argparse
import datasets
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm

import random
random.seed(42)

from google.cloud import language_v1
from google.cloud.language_v1.types import Entity as GcpEntity
from urllib.parse import urlparse, unquote_plus


#### Entity Linking from Target Text
def get_entities_from_text_using_gcp(
    text_content,
    gcp_client_options,
    accepted_entity_type_list=GcpEntity.Type
    ):
    """
    Analyzing Entities in a String

    Args:
      text_content: The text content to analyze
      gcp_client_options
      accepted_entity_type_list: https://cloud.google.com/natural-language/docs/reference/rest/v1/Entity#Type (language_v1.Entity.Type)
    """

    client = language_v1.LanguageServiceClient(client_options=gcp_client_options)

    # Available types: PLAIN_TEXT, HTML
    type_ = language_v1.Document.Type.PLAIN_TEXT

    # Optional. If not specified, the language is automatically detected.
    # For list of supported languages:
    # https://cloud.google.com/natural-language/docs/languages
    language = "en"
    document = {"content": text_content, "type_": type_, "language": language}

    # Available values: NONE, UTF8, UTF16, UTF32
    # https://cloud.google.com/natural-language/docs/reference/rest/v1/EncodingType
    encoding_type = language_v1.EncodingType.UTF32 # For python

    response = client.analyze_entities(
        request={"document": document, "encoding_type": encoding_type}
    )
    
    entitiy_info = []
    for entity in response.entities:
        _score = entity.salience
        if entity.metadata.get('wikipedia_url') is None:
            continue
        if entity.type not in accepted_entity_type_list:
            continue

        _url = entity.metadata['wikipedia_url']
        _wiki_page_title_from_url = urlparse(_url).path.rstrip('/').split('/')[-1]
        _wiki_title = unquote_plus(_wiki_page_title_from_url).replace('_', ' ')

        if _wiki_title.startswith('d:'):  # redirect to wikidata
            continue

        for mention in entity.mentions:
            _mention_text = mention.text.content 
            _mention_start_offset = mention.text.begin_offset
            _mention_end_offset = mention.text.begin_offset + len(mention.text.content)
            
            entitiy_info.append({
                'mention':_mention_text,
                'entity': _wiki_title,
                'start': _mention_start_offset,
                'end': _mention_end_offset,
                'label':None,
                'wikipedia_link':_url,
                'evidence':None,
                'gcp_entity_type': str(entity.type)
            })

    entity_info = sorted(entitiy_info, key=lambda dic: dic['start'])
    return {
        'target_text': text_content,
        'entity_info': entity_info
    }

def filter_by_length_of_text(ds):
    df = pd.DataFrame([len(text) for text in ds['text']])
    print(df.describe())
    lower_q, upper_q = df.quantile([0.25,0.75]).values
    filtered_ds = ds.filter(lambda ex: len(ex['text'])>=int(lower_q) and len(ex['text'])<=int(upper_q))
    print("filter_by_length_of_text: ", filtered_ds)
    return filtered_ds

def filtered_by_number_of_entities(ds, entity_info_dic):
    def _get_num_of_unique_entity(key):
        unique_entity = set([ei['entity'] for ei in entity_info_dic[key]])
        return len(unique_entity)

    num_of_unique_entity = [_get_num_of_unique_entity(url) for url in ds['url']]
    df = pd.DataFrame(num_of_unique_entity)
    print(df.describe())
    lower_q, upper_q = df.quantile([0.25,0.75]).values
    filtered_ds = ds.filter(lambda ex: _get_num_of_unique_entity(ex['url'])>=int(lower_q) \
                            and _get_num_of_unique_entity(ex['url'])<=int(upper_q))
    print("filtered_by_number_of_entities: ", filtered_ds)
    return filtered_ds


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_info_cache_path", type=str, default=None)
    parser.add_argument("--gcp_api_key", type=str, required=True)
    parser.add_argument("--gcp_quota_project_id", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    Path("./dump").mkdir(parents=True, exist_ok=True)

    ds = datasets.load_dataset('c4', 'realnewslike', split='validation')
    ds = ds.map(lambda ex: {'text': ex['text'].replace("\ufeff", ""), 'url': ex['url']}) # remove BOM

    # filter by length of text
    filtered_ds = filter_by_length_of_text(ds)

    # get entities
    if args.entity_info_cache_path:
        with open(args.entity_info_cache_path, 'rb') as f:
            entity_info_dic = pickle.load(f)
    else:
        _gcp_client_options = {
            "api_key": args.gcp_api_key,
            "quota_project_id": args.gcp_quota_project_id
        }
        _accepted_entity_type_list = [GcpEntity.Type.UNKNOWN, GcpEntity.Type.PERSON, GcpEntity.Type.LOCATION, \
                                        GcpEntity.Type.ORGANIZATION, GcpEntity.Type.EVENT, GcpEntity.Type.WORK_OF_ART, \
                                        GcpEntity.Type.CONSUMER_GOOD, GcpEntity.Type.OTHER]  # ALL
        entity_info_dic = {
            url: get_entities_from_text_using_gcp(
                text,
                _gcp_client_options,
                _accepted_entity_type_list
            )['entity_info'] for text, url in tqdm(zip(filtered_ds['text'], filtered_ds['url']), total=len(filtered_ds['text']))
        }

        with open('./dump/c4_realnewslike_entity_info_list.pickle', 'wb') as f:
            pickle.dump(entity_info_dic, f)

    # filter by number of entities
    filtered_ds = filtered_by_number_of_entities(filtered_ds, entity_info_dic)

    # sample target documents
    _sampled_idx = random.sample(range(len(filtered_ds)), args.num_samples)
    sampled_ds = filtered_ds.select(_sampled_idx)
    sampled_ds.save_to_disk('./dump/0_sampled_c4')
