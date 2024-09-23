![kiroku](https://github.com/user-attachments/assets/0a254a3d-2358-40dc-91c1-ff16fa09c6f4)
 
# Introduction

Kiroku is a multi-agent system that helps you organize and write documents.  

I started writing it because during my PhD at Stanford, I had to go through a formal process to write papers and eventually the thesis, and I tried to follow these steps. 
The difference is that at that time, I was the student, and my advisor was reviewing my documents, and I got the idea: what if the writer becomes the advisor, and the 
multi-agent system becomes the student?

This flow has several advantages:
- It helps you organize the ideas in a better way because you can generate a sequence of paragraphs very quickly.
- It helps you change your communication through iterative evaluation of the message
- Recently, [<a href="#1">1</a>] suggested that LLMs help you can help complex topics by discussing with with the LLM.

![AdvisorPicture](https://github.com/user-attachments/assets/dbbed542-4d24-4af2-bf83-3d6fc5113c4f)
(c) PhDCommics (www.phdcommics.com) of the advisor and the student

The original code was obtained from a short course From Harrison Chase and Rotem Weiss [<a href="#2">2</a>], 
but I believe not even the prompts resemble any of the prompts from original prompts. However, I do recognize and
credit to them the original code that I used as a reference.

![image](https://github.com/user-attachments/assets/d5212215-4eb9-4198-a3b4-4ea0b8f7f249)

# Before You Run

To run Kiroku, you need an OPENAI_API_KEY and a TAVILY_API_KEY.

To get an OPENAI_API_KEY, you can check https://platform.openai.com/docs/quickstart .

To get a TAVILY_API_KEY, you can check the site https://app.tavily.com/sign-in, and click "Sign in".

You may want to use a tool like `direnv` to manage the environment variables `OPENAI_API_KEY` and `TAVILI_API_KEY` on a per-directory basis. 
This will help you automatically load these variables when you are working within the Kiroku project directory. 
`direnv` supports Linux, macOS, and Windows through WSL.

# Installation

Kiroku supports Python between versions 3.7 and 3.11.

### 1. Set up a virtual environment
You can use Python’s `venv` module to create an isolated environment for dependencies. This ensures a clean environment and avoids conflicts with system packages.

```shell
cd kiroku
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# The Initial Configuration

The initial configuration is specified in an YAML file with the following fields:

- `title` is a suggestion for the title or the final title to use (if `suggest_title` is false).
- `suggest_title` turns on recommendation for titles based on your original title.
- `generate_citations`: if true, it will generate citations and references.
- `type_of_document`: helps the Kiroku define whether it should use more technical terms, or whether we are trying to write children's stories.
- `area_of_paper`: together with `hypothesis`, it helps Kiroku to understand what to write.
- `section_names`: list of sections, as in the example below:
```markdown
section_names:
- Introduction
- Related Work
- Architecture of Kiroku
- Results
- Conclusions
- References
```
- `number_of_paragraphs`: instructs Kiroku to write that many paragraphs per section.
```markdown
number_of_paragraphs:
  "Introduction": 4
  "Related Work": 7
  "Architecture of Kiroku": 4
  "Results": 4
  "Conclusions": 3
  -"References": 0
```
- `hypothesis` tells Kiroku whether you want to establish something to be good or bad, and it will define the message.
- `instructions`: as you interact with the document giving instructions like "First paragraph of Introduction should
discuss the revolution that was created with the lauch of ChaGPT", you may want to add some of them to the instruction so that
in the next iteration, Kiroku will observe your recommendations. In Kiroku, `instructions` are appended into the `hypothesis` at
this time, so you will not see them. I usually put `\n\n` after each instruction to assist the underlying LLM.
- `results`: I usually put it here as I fill this later on.
- `references` are references you want Kiroku to use during its search phase for information.
- `number_of_queries` tells Kiroku how many questions it will generate to Tavily to search for information.
- `max_revisions` tells Kiroku how many times it performs reflection and document writing upon analyzing reflection results
(please note that setting this document to `1`, it means no revision).
- `temperature` is the temperature of the LLM (usually I set it to a small number).

The final YAML is given below:

```yaml
title: "Writing Masterpieces when You Become the Adivisor"
suggest_title: True
generate_citations: True
type_of_document: "research seminal paper"
area_of_paper: "AI and Computer Science"
section_names:
- Introduction
- Related Work
- Architecture
- Results
- Conclusions
- References
number_of_paragraphs:
  "Introduction": 4
  "Related Work": 7
  "Architecture": 4
  "Results": 4
  "Conclusions": 3
  "References": 0
hypothesis: "
We want to show in this paper that we turn paper writers into 'advisors'
and a multi-agent system into a 'advisee' who will observe the instructions by,
interactively turning a course draft of a paper into a publication ready
document.
"
instructions: "
For the following instructions, you should use your own words.
\n\n
The section 'Introduction', you should focus on:
\n
- In the first paragraph, you should discuss that the world has change
since the release of ChatGPT.
\n
In the section 'Architecture', you should show the picture
'/file=images/multi-agent.jpeg' to discuss we write a paper by defining a
title and hypothesis, writing topic sentences, expanding topic sentences into
paragraphs, writing the paragraphs, and finally reviewing what you have written.
"
results: "
This is an example on how you can put a results table.
<table>
  <tr>
      <td> </td>
      <td> Normal Text Rate</td>
      <td> Kiroku Rate</td>
  </tr>
  <tr>
    <td> Experiment 1</td>
    <td> 3 </td>
    <td> 9 </td>
  </tr>
  <tr>
    <td> Experiment 2</td>
    <td> 5 </td>
    <td> 10 </td>
  </tr>
</table>
"
references:
- "Harrison Chase, Rotem Weiss. AI Agents in LangGraph. https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph"
number_of_queries: 8
max_revisions: 1
temperature: 0.1
```

There is a script `check_yaml` that checks if the YAML file is consistent and it will not crash Kiroku.

I recommend putting all YAML files right now in the `kikoku/proj` direcotry. All images should be in `kiroku/proj/images`. 

Because of a limitation of Gradio, you need to specify images as `'/file=images/<your-image-file>'` such as in the example `/file=images/multi-agent.jpeg`.

# Running

I recommend running writer as:

```shell
cd {where Kiroku directory is located}
KIROKU_PROJECT_DIRECTORY=`pwd`/proj ./kiroku
```

Go to your preferred browser and open `localhost:7860`.

As for instructions, you can try `I liked title 2` or `I liked the original title`.

Whenever you give an instructions you really liked, remember to add it to the `instructions` field.

# License

Apache License 2.0 (see LICENSE.txt)

# Bugs

:-)

# References

<p id=1> 1. https://www.youtube.com/watch?v=om7VpIK90vE</p>

<p id=2> 2. Harrison Chase, Rotem Weiss. AI Agents in LangGraph. https://www.deeplearning.ai/short-courses/ai-agents-in-langgraph</p>

# Authors

Claudionor N. Coelho Jr (https://www.linkedin.com/in/claudionor-coelho-jr-b156b01/)

Fabricio Ceolin (https://br.linkedin.com/in/fabceolin)

Luiza N. Coelho (https://www.linkedin.com/in/luiza-coelho-08499112a/)



