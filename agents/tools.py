# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício J V Ceolin

from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper, GoogleSerperAPIWrapper
from langchain_experimental.utilities import PythonREPL
from langchain_community.tools import (
    DuckDuckGoSearchResults,
    DuckDuckGoSearchRun,
    WikipediaQueryRun,
    ArxivQueryRun,
    GoogleSerperRun,
)
from langchain_community.utilities import GoogleSerperAPIWrapper
from tavily import TavilyClient
import os

from .pubmed import PubMedAPIWrapper
from langchain_community.tools.pubmed.tool import PubmedQueryRun
from langchain_core.tools import Tool

tavily_api_key = os.environ.get("TAVILY_API_KEY")

if tavily_api_key:
    tavily = TavilyClient(tavily_api_key)
else:
    tavily = None
try:
    google = GoogleSerperRun(api_wrapper=GoogleSerperAPIWrapper())
except:
    google = None

pubmed = PubmedQueryRun()
pubmed.api_wrapper = PubMedAPIWrapper()
arxiv = ArxivQueryRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
python_repl = PythonREPL()
python_repl = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)
ddgs = DuckDuckGoSearchRun(max_results=2)
ddgsr = DuckDuckGoSearchResults(max_results=2)

tools_list = [wikipedia, ddgs, ddgsr, python_repl, arxiv, pubmed]

tools = {tool.name: tool for tool in tools_list}

