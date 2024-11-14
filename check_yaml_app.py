# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício José Vieira Ceolin, Luiza Nacif Coelho

import chromalog
from fire import Fire
import glob
import logging
import re
import os
import yaml
from yaml import Loader

chromalog.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

def check_paragraphs(state):
    n_errors = 0
    sections = state.get("section_names", [])
    paragraphs = state.get("number_of_paragraphs", {})
    if len(sections) != len(paragraphs):
        logger.error(
            f"... 'number_of_paragraphs' must have same size as"
            "'section_names'")
        n_errors += 1
    if not (len(sections) and len(paragraphs)):
        return n_errors
    if isinstance(paragraphs, dict):
        for name in sections:
            if not name in paragraphs:
                logger.error(
                    f"... section '{name}' does not exist in"
                    "'number_of_paragraphs'")
                n_errors += 1
        for name in paragraphs:
            if not name in sections:
                logger.error(
                    f"... section '{name}' not present in "
                    "'section_names'")
                n_errors += 1
            if not isinstance(paragraphs[name], int):
                logger.error(
                    f"... number of paragraphs for '{name}' is not a number")
                n_errors += 1
            if paragraphs[name] < 0:
                logger.error(
                    f"... number of paragraphs for '{name}' is negative")
                n_errors += 1
        if paragraphs["References"] != 0:
            logger.error(
                f"... section 'References' must have a 0"
                " number of paragraphs")
            n_errors += 1
    elif isinstance(paragraphs, list):
        if paragraphs[-1] != 0:
            logger.error(
                f"... section 'References' must have a 0"
                " number of paragraphs")
            n_errors += 1
    else:
        logger.error(
            f"... 'number_of_paragraphs' must be of type list or dict")
        n_errors += 1

    # now, let's just report the images
    
        
    return n_errors

def check_boolean(state, field):
    try:
        logger.debug(f"... '{field}' specified")
        if state[field] not in [False, True]:
            logger.error(f"... '{field}' not boolean")
            return 1
    except:
        logger.error(f"... '{field}' not present")
        return 1

    return 0

def check_string(state, field):
    try:
        logger.debug(f"... '{field}' specified")
        if not isinstance(state[field], str):
            logger.error(f"... '{field}' not string")
            return 1
    except:
        logger.error(f"... '{field}' not present")
        return 1

    return 0

def check_list(state, field, entry_type):
    n_errors = 0
    try:
        logger.debug(f"... '{field}' specified")
        if not isinstance(state[field], list):
            logger.error(f"... '{field}' not list")
            n_errors += 1
        for entry in state[field]:
            if not isinstance(entry, entry_type):
                logger.error(f"... '{entry}' not of type {entry_type}")
                n_errors += 1
    except:
        logger.error(f"... '{field}' not present")
        n_errors += 1

    return n_errors

def check_int(state, field):
    try:
        logger.debug(f"... '{field}' specified")
        if not isinstance(state[field], int):
            logger.error(f"... '{field}' not int")
            return 1
    except:
        logger.error(f"... '{field}' not present")
        return 1

    return 0

def check_float(state, field):
    try:
        logger.debug(f"... '{field}' specified")
        if not isinstance(state[field], float):
            logger.error(f"... '{field}' not float")
            return 1
    except:
        logger.error(f"... '{field}' not present")
        return 1

    return 0

def read_initial_state(filename):
    n_errors = 0
    try:
        stream = open(filename, 'r')
    except:
        logger.error(f"... cannot open {filename}")
        n_errors += 1

    try:
        state = yaml.load(stream, Loader=Loader)
    except:
        logger.error(f"... YAML file error {filename}")
        n_errors += 1

    n_errors += check_string(state, "title")
    n_errors += check_boolean(state, "suggest_title")
    n_errors += check_boolean(state, "generate_citations")
    n_errors += check_string(state, "type_of_document")
    n_errors += check_string(state, "area_of_paper")
    n_errors += check_list(state, "section_names", str)

    # some sections are required for now
    sections = state.get("section_names", [])
    required_sections = [
        "Introduction", 
        "Conclusions", 
        "References"
    ]
    for name in required_sections:
        if name not in sections:
            logger.error(f"... section '{name}' not present")
            n_errors += 1

    n_errors += check_paragraphs(state)
    n_errors += check_string(state, "hypothesis")
    n_errors += check_string(state, "instructions")
    n_errors += check_string(state, "results")
    n_errors += check_list(state, "references", str)
    n_errors += check_int(state, "number_of_queries")
    n_errors += check_int(state, "max_revisions")
    n_errors += check_float(state, "temperature")

    # now let's find the images
    # an image should be like this: /file=images/file.type 
    working_dir = os.environ.get("WRITER_PROJECT_DIRECTORY", os.getcwd())
    images = [fn.split("/")[-1] for fn in glob.glob(working_dir + '/images/*')]
    text = state.get("hypothesis", "") + state.get("instructions", "")
    for find_entry in re.finditer(r'\/?file=[^\t\n \'\"]*', text):
        l, r = find_entry.span()
        file_entry = text[l:r]
        filename = file_entry.split('=')[-1]
        logger.debug(f"... found image {filename} in YAML")
        filename = filename.split("/")[-1]
        if filename not in images:
            logger.error(f"       did not find {filename} "
                         f"in {working_dir}/images")
        else:
            logger.debug(f"       found {filename} "
                         f"in {working_dir}/images")

    if n_errors:
        logger.error(f"... found {n_errors} errors")

if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        logger.critical("... We presently require an OPENAI_API_KEY.")
    if not os.environ.get("TAVILY_API_KEY"):
        logger.critical("... We presently require an TAVILY_API_KEY.")

    Fire(read_initial_state)
