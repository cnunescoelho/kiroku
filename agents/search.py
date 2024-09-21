# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício José Vieira Ceolin, Luiza Nacif Coelho

import logging

from .tools import tools, tavily

def get_additional_info(link):
    arxiv = tools["arxiv"]
    pubmed = tools["pub_med"]
    doc = ""
    if "https://arxiv.org/" in link:
        arxiv_entry = link.split("/")[-1].strip()
        doc = arxiv.run(arxiv_entry)
    elif "pubmed" in link:
        doc = pubmed.run(link)[0]
    if doc:
        doc = ", " + doc
    return doc

def search_query_ideas(query_ideas, cache, max_results=3, search_engine="tavily"):
    """
    Searches the web based on query ideas, and expand search in some cases.
    :param query_ideas: List of query ideas to search on the web.
    :param cache: Cached list of links so that we do not duplicate.
    :param max_results: Maximum number of results to search.
    :param search_engine: "tavily", others will be implemented later
    :return: List of searched information to be used.
    """
    content = []

    if search_engine == "tavily":
        search = tavily
    else:
        raise "Invalid search engine"

    for iter, q in enumerate(query_ideas["queries"]):
        if not q: continue
        logging.warning(f"search for query '{q}'")
        if search_engine == "tavily":
            response = search.search(q, max_results=max_results)
            for result in response["results"]:
                text = (
                    f"title: {result['title']}, "
                    f"link: {result['url']}, "
                    f"content: {result['content']}"
                )
                link = result['url']
                if link[-1] == '/': link = link[:-1]
                title = result['title']
                if link in cache or title in cache:
                    continue
                logging.warning(f"    {link}")
                try:
                    info = get_additional_info(link)
                except:
                    # an error happened. ignore
                    logging.warning(f"    {link} error")
                    info = ""
                text = text + info
                cache.add(link)
                cache.add(title)
                content.append(text)
    return content, cache
