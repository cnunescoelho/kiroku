# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício J V Ceolin

from nltk import sent_tokenize
import numpy as np
import re
from langchain_openai import OpenAIEmbeddings

def  get_sentences(paper):
    """
    Get the list of sentences for the paper.
    :param paper: paper without conclusions and references.
    :return: list of sentences.
    """
    search = re.search(r"## Abstract[^#]*", paper)
    if search:
        l, r = search.span()
        paper = paper[:l] + paper[r-1:]
    paragraphs = "\n".join(
        [p for p in paper.split("\n") if p and not (p[0] == '#' or p[:2] == '![')])
    sentences = [s.split('\n')[-1] for s in sent_tokenize(paragraphs)]

    return sentences

def get_references(references):
    """
    Get the references and generate an unnumbered list.
    :param references:
    :return: list of references.
    """
    ref_list = references.split('\n')
    ref_list = [('.'.join(r.split('.')[1:])).strip()
                for r in ref_list if r.strip()]
    return ref_list

def reorder_references(reference_index, references):
    new_references = []
    for key in sorted(reference_index.keys()):
        for i in range(len(reference_index[key])):
            j = len(new_references)+1
            reference_index[key][i], j = j, reference_index[key][i]
            new_references.append(references[j-1])
    return new_references

def insert_references(draft):
    """
    Insert references into sentences by computing the best match for the embeddings.
    :param draft: paper draft
    :return: new paper version.
    """
    draft = draft.strip()
    search = re.search(r"## References", draft)
    if search:
        # split paper into paper up to reference and references
        l, r = search.span()
        paper = draft[:l]
        references = draft[r:]

        # remove Conclusions if they exist as we do not want to put
        # references in conclusions.
        search = re.search(r"## Conclusions\n\n", paper)
        if search:
            l, r = search.span()
            paper_no_conclusions = paper[:r]
        else:
            paper_no_conclusions = paper

        # get list of sentences and references
        sentences = get_sentences(paper_no_conclusions)
        references = get_references(references)

        # compute the embeddings for list of sentences and list of
        # references
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        emb_sents = np.array(embeddings.embed_documents(sentences))
        emb_refs = np.array(embeddings.embed_documents(references))

        # insertion point is the argmax for each sentence. This is an
        # approximation to the problem.
        similarities = np.dot(emb_sents, emb_refs.T)
        citation_inserts = np.argmax(similarities, axis=0)

        # merge multiple citations to the correct insertion
        # point.
        citations = {}
        for i in range(len(citation_inserts)):
            s = citation_inserts[i]
            if s in citations:
                citations[s].append(i+1)
            else:
                citations[s] = [i+1]

        references = reorder_references(citations, references)

        # generate citations to references
        paper = paper.strip()
        for s in citations:
            l = paper.find(sentences[s])
            r = l + len(sentences[s])
            cit = ",".join([
                f'<a href="#{i}">{i}</a>'
                for i in citations[s]])
            paper = paper[:r-1] + f" [{cit}]" + paper[r-1:]

        # generated clickable references
        references = [f'<p id={i+1}>{i+1}. {r.strip()}</p>'
                      for i, r in enumerate(references)]

        # create new draft of the paper.
        draft = (
            paper.strip() +
            "\n\n## References\n\n" +
            "\n\n".join(references)
        )

    return draft

