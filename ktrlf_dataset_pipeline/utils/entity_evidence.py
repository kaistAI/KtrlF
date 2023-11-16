import requests


def crawl_wikipedia_article(_title, num_evidence_sent=20):
    try:
        _extract_url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exlimit=max&explaintext&titles={_title}"
        res = requests.get(_extract_url)
        pages = res.json()['query']['pages']
        _pageid = list(pages.keys())[0]
        content = pages[_pageid]['extract']
        evidence = list(filter(None,content.split('\n')))[:num_evidence_sent]
    except:
        evidence = []
    return

