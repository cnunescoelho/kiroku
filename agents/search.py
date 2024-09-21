# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício J V Ceolin

import re
import requests
from bs4 import BeautifulSoup
import time
import logging

from .tools import tools, tavily, google

def search_nature(url):
    """
    Searches for fields in Nature magazine.
    :param url: URL in Nature
    :return: List of fields searched.
    """

    r = requests.get(url)

    if r.status_code == 200:
        soup = BeautifulSoup(r.text, 'html.parser')

        abstract = ""
        title = ""
        authors = []
        citation = ""
        for meta in soup.find_all('meta'):
            name = meta.get("name")
            if name == "dc.description":
                abstract = meta.get("content")
            elif name == "dc.title":
                title = meta.get("content")
            elif name == "dc.creator":
                authors.append(meta.get("content"))
        for p in soup.find_all('p', class_='c-bibliographic-information__citation'):
            citation = p.get_text()
            citation = " ".join([c.strip() for c in citation.split('\n')])

        reference = (
            f"Title: '{title}', "
            f"Authors: '{authors}', "
            f"Citation: '{citation}', "
            f"Summary: '{abstract}'."
        )

        return reference
    else:
        return ""

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
    :param search_engine: "ddgs", "google", "tavily"
    :return: List of searched information to be used.
    """
    content = []

    ddgsr = tools["duckduckgo_results_json"]
    if search_engine == "ddgs":
        search = ddgsr
    elif search_engine == "tavily":
        search = tavily
    elif search_engine == "google":
        raise "Invalid search engine"
        search = google
    else:
        raise "Invalid search engine"

    for iter, q in enumerate(query_ideas["queries"]):
        if not q: continue
        logging.warning(f"search for query '{q}'")
        if search_engine == "ddgs":
            iterations = 1
            delay = 1
            while iterations < 10:
                if True:
                    response = search.run(q, max_results=max_results)
                    break
                else:
                    logging.warning(f"timeout in search engine iteration {iter}. waiting {delay} seconds now")
                    time.sleep(delay)
                    delay *= 2
                    iterations += 1
            # split search results
            for result in re.split(r'\s*\[(.+?)\](,|$)', response):
                if result and result != ',':
                    # get the link of the search result
                    link = re.search(r'link: (\S+)', result)
                    if not link:
                        logging.warning("could not extract link from search")
                        logging.warning(result)
                        import pdb; pdb.set_trace()
                        continue
                    link = link.groups()[0]
                    if link[-1] == '/': link = link[:-1]
                    if link in cache:
                        continue
                    logging.warning(f"    {link}")
                    try:
                        info = get_additional_info(link)
                    except:
                        # an error happened. ignore
                        logging.warning(f"    {link} error")
                        info = ""
                    text = result + info
                    cache.add(link)
                    content.append(text)
        elif search_engine == "tavily":
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

