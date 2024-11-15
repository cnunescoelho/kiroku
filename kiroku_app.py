# Copyright (c) 2024 Claudionor Coelho Jr, Fabrício José Vieira Ceolin, Luiza Nacif Coelho

from agents.states import *
from copy import deepcopy
import gradio as gr
from IPython.display import display, Image
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod
from langgraph.graph import StateGraph, START, END
import logging
import markdown
import os
import shutil
import subprocess
import re
import yaml
try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

logging.basicConfig(level=logging.WARNING)
class DocumentWriter:
    def __init__(
            self,
            suggest_title=False,
            generate_citations=True,
            model_name="openai",
            temperature=0.0):
        self.suggest_title = suggest_title
        self.generate_citations = generate_citations
        self.state = None
        self.set_thread_id(1)
        models = {
            "openai": "gpt-4o-mini",
            "openai++": "gpt-4o"
        }
        assert model_name in ["openai", "openai++"]
        model = models.get(model_name, "openai")
        # if user did not specify the beefed up models upfront, we try to
        # use cheaper models whenever possible for simpler tasks.
        if model_name in ["openai", "openai++"]:
            self.model_m = ChatOpenAI(
                model=model, temperature=temperature)
            self.model_p = ChatOpenAI(
                model=models["openai++"], temperature=temperature)
        self.state_nodes = {
            node.name : node
            for node in [
                SuggestTitle(self.model_m),
                SuggestTitleReview(self.model_m),
                InternetSearch(self.model_p),
                TopicSentenceWriter(self.model_m),
                TopicSentenceManualReview(self.model_m),
                PaperWriter(self.model_p),
                WriterManualReviewer(self.model_m),
                ReflectionReviewer(self.model_p),
                ReflectionManualReview(self.model_m),
                WriteAbstract(self.model_p),
                GenerateReferences(self.model_m),
                GenerateCitations(self.model_m),
                GenerateFigureCaptions(self.model_m),
            ] if self.mask_nodes(node.name)
        }
        self.create_graph(suggest_title)

    def mask_nodes(self, name):
        '''
        We do not process nodes if user does not want to run that phase.
        :param name: name of the node.
        :return: True if we keep nodes, False otherwise
        '''
        if (
                name in ["suggest_title", "suggest_title_review"] and
                not self.suggest_title):
            return False
        if name in ["generate_references", "generate_citations"] and not self.generate_citations:
            return False
        return True

    def create_graph(self, suggest_title):
        '''
        Builds a graph to execute the different phases of a document writing.

        :param suggest_title: If we are to suggest a better title for the paper.
        :return: Nothing.
        '''
        memory = MemorySaver()

        builder = StateGraph(AgentState)

        # Add nodes to the graph
        for name, state in self.state_nodes.items():
            builder.add_node(name, state.run)

        # Add edges to the graph
        if suggest_title:
            builder.add_conditional_edges(
                "suggest_title_review",
                self.is_title_review_complete,
                {
                    "next_phase": "internet_search",
                    "review_more": "suggest_title"
                }
            )
        builder.add_conditional_edges(
            "topic_sentence_manual_review",
            self.is_plan_review_complete,
            {
                "topic_sentence_manual_review": "topic_sentence_manual_review",
                "paper_writer": "paper_writer"
            }
        )

        builder.add_conditional_edges(
            "writer_manual_reviewer",
            self.is_generate_review_complete,
            {
                "manual_review": "writer_manual_reviewer",
                "reflection": "reflection_reviewer",
                "finalize": "write_abstract"
            }
        )
        if suggest_title:
            builder.add_edge("suggest_title", "suggest_title_review")
        builder.add_edge("internet_search", "topic_sentence_writer")
        builder.add_edge("topic_sentence_writer", "topic_sentence_manual_review")
        builder.add_edge("paper_writer", "writer_manual_reviewer")
        builder.add_edge("reflection_reviewer", "additional_reflection_instructions")
        builder.add_edge("additional_reflection_instructions", "paper_writer")
        if self.generate_citations:
            builder.add_edge("write_abstract", "generate_references")
            builder.add_edge("generate_references", "generate_citations")
            builder.add_edge("generate_citations", "generate_figure_captions")
        else:
            builder.add_edge("write_abstract", "generate_figure_captions")
        builder.add_edge("generate_figure_captions", END)

        # Starting state is either suggest_title or planner.
        if suggest_title:
            builder.set_entry_point("suggest_title")
        else:
            builder.set_entry_point("internet_search")

        self.interrupt_after = []
        self.interrupt_before = [ "suggest_title_review" ] if suggest_title else []
        self.interrupt_before.extend([
            "topic_sentence_manual_review",
            "writer_manual_reviewer",
            "additional_reflection_instructions",
        ])
        if self.generate_citations:
            self.interrupt_before.append("generate_citations")
        # Build graph
        self.graph = builder.compile(
            checkpointer=memory,
            interrupt_before=self.interrupt_before,
            interrupt_after=self.interrupt_after,
            debug=False
        )

    def is_title_review_complete(self, state: AgentState) -> str:
        '''
        Checks if title review is complete based on an END instruction.

        :param state: state of agent.
        :return: next state of agent.
        '''
        if not state["messages"]:
            return "next_phase"
        else:
            return "review_more"

    def is_plan_review_complete(self, state: AgentState, config: dict) -> str:
        '''
        Checks if plan manual review is complete based on an empty instruction.

        :param state: state of agent.
        :return: next state of agent.
        '''
        if config["configurable"]["instruction"]:
            return "topic_sentence_manual_review"
        else:
            return "paper_writer"

    def is_generate_review_complete(self, state: AgentState, config: dict) -> str:
        '''
        Checks if review of generation phase is complete based on number of revisions.

        :param state: state of agent.
        :return: next state to go.
        '''
        if config["configurable"]["instruction"]:
            return "manual_review"
        elif state["revision_number"] <= state["max_revisions"]:
            return "reflection"
        else:
            return "finalize"

    def invoke(self, state, config):
        '''
        Invokes the multi-agent system to write a paper.

        :param state: state of initial invokation.
        :return: draft
        '''
        config = { "configurable": config }
        config["configurable"]["thread_id"] = self.get_thread_id()
        response = self.graph.invoke(state, config)
        self.state = response
        draft = response.get("draft", "").strip()
        # we have to do this because the LLM sometimes decide to add
        # this to the final answer.
        if "```markdown" in draft:
            draft = "\n".join(draft.split("\n")[1:-1])
        return draft

    def stream(self, state, config):
        '''
        Invokes the multi-agent system to write a paper.

        :param state: state of initial invokation.
        :return: full state information
        '''
        config = { "configurable": config }
        config["configurable"]["thread_id"] = self.get_thread_id()
        for event in self.graph.stream(state, config, stream_mode="values"):
            pass
        draft = event["draft"]
        # we have to do this because the LLM sometimes decide to add
        # this to the final answer.
        if "```markdown" in draft:
            draft = "\n".join(draft.split("\n")[1:-1])
        return draft

    def get_state(self):
        """
        Returns the full state of the document writing process.
        :return: Generated state from invoke
        """
        config = { "configurable": { "thread_id": self.get_thread_id() }}
        return self.graph.get_state(config)

    def update_state(self, new_state):
        """
        Updates the state of langgraph.
        :param new_state:
        :return: None
        """
        config = { "configurable": { "thread_id": self.get_thread_id() }}
        self.graph.update_state(config, new_state.values)

    def get_thread_id(self):
        return str(self.thread_id)

    def set_thread_id(self, thread_id):
        self.thread_id = str(thread_id)

    def draw(self):
        display(
            Image(
                self.graph.get_graph().draw_mermaid_png(
                    draw_method=MermaidDrawMethod.API,
                )
            )
        )

class KirokuUI:
    def __init__(self, working_dir):
        self.working_dir = working_dir
        self.first = True
        self.next_state = -1
        self.references = []

    def read_initial_state(self, filename):
        '''
        Reads initial state from a YAML 'filename'.
        :param filename: YAML file containing initial paper configuration.
        :return: initial state dictionary.
        '''
        stream = open(filename, 'r')
        try:
            state = yaml.load(stream, Loader=Loader)
        except yaml.parser.ParserError:
            logging.error("Cannot load YAML file")
            return {}
        if not "sentences_per_paragraph" in state:
            state["sentences_per_paragraph"] = 4
        self.suggest_title = state.pop("suggest_title", False)
        self.generate_citations = state.pop("generate_citations", False)
        self.model_name = state.pop("model_name", "openai++")
        self.temperature = state.pop("temperature", 0.0)
        state["hypothesis"] = (
                state["hypothesis"] + "\n\n" + state.pop("instructions", "")
        )
        return state

    def step(self, instruction, state_values=None):
        """
        Performs one step of the graph invocation, stopping at the next break point.
        :param instruction: instruction to execute.
        :param state_values: initial state values or None if continuing.
        :return: draft of the paper.
        """
        config = { "instruction": instruction }
        draft = self.writer.invoke(state_values, config)
        return draft

    def update(self, instruction):
        """
        Updates state upon submitting an instruction or updating references.
        :param instruction: instruction to be executed.
        :return: new draft, atlas message and making input object non-interactive.
        """
        draft = self.step(instruction)
        state = self.writer.get_state()
        current_state = state.values["state"]
        try:
            next_state = state.next[0]
        except:
            next_state = "NONE"

        # if state is in reflection stage, draft to be shown is in the critique field.
        if (
                current_state == "reflection_reviewer" and
                next_state == "additional_reflection_instructions"
        ):
            draft = state.values["critique"]

        # if next state is going to generate citations, we populate the references
        # for the Tab References.
        if next_state == "generate_citations":
            self.references = state.values.get("references", []).split('\n')

        # if we have reached the end, we will save everything.
        if next_state == END or next_state == "NONE":
            dir = os.path.splitext(self.filename)[0]
            logging.warning(f"saving final draft in {dir}")
            self.save_as()

        self.next_state = next_state
        return (
            draft,
            self.atlas_message(next_state),
            gr.update(interactive=False)
        )

    def atlas_message(self, state):
        """
        Returns the Echo message for a given state.
        :param state: Next state of the multi-agent system.
        :return:
        """
        message = {
            "suggest_title_review":
                "Please suggest review instructions for the title.",
            "topic_sentence_manual_review":
                "Please suggest review instructions for the topic sentences.",
            "writer_manual_reviewer":
                "Please suggest review instructions for the main draft.",
            "additional_reflection_instructions":
                "Please provide additional instructions for the overall paper review.",
            "generate_citations":
                "Please look at the references tab and confirm the references."
        }

        instruction = message.get(state, "")
        if instruction or state == "generate_citations":
            if state == "generate_citations":
                return instruction
            else:
                return instruction + " Type <RETURN> when done."
        else:
            return "We have reached the end."

    def initial_step(self):
        """
        Performs initial step, in which we need to providate a staet to the graph.
        :return: draft and Echo message.
        """
        state_values = deepcopy(self.state_values)
        if self.suggest_title:
            state_values["state"] = "suggest_title"
        else:
            state_values["state"] = "topic_sentence_writer"
        # initialize a bunch of variables users should not care about.
        # in principle this could be initialized in the Pydantic object,
        # but I could not make this work there.
        state_values["references"] = state_values.get("references", [])
        state_values["draft"] = ""
        state_values["revision_number"] = 1
        state_values["messages"] = []
        state_values["review_instructions"] = []
        state_values["review_topic_sentences"] = []
        draft = self.step("", state_values)
        state = self.writer.get_state()
        current_state = state.values["state"]
        try:
            next_state = state.next[0]
        except:
            next_state = "NONE"
        return draft, self.atlas_message(next_state)

    def process_file(self, filename):
        """
        Processes file uploaded.
        :param filename: file name where to read the file.
        :return: State that was read and make input non-interactive.
        """
        pwd = os.getcwd()
        logging.warning(f"Setting working directory to {pwd}")
        self.filename = pwd + "/" + filename.split('/')[-1]
        self.state_values = self.read_initial_state(filename)
        if self.state_values:
            self.writer = DocumentWriter(
                suggest_title=self.suggest_title,
                generate_citations=self.generate_citations,
                model_name=self.model_name,
                temperature=self.temperature)
        return self.state_values, gr.update(interactive=False)

    def save_as(self):
        """
        Saves project status. We save all instructions given by the user.
        :return: message where the project was saved.
        """
        filename = self.filename
        state = self.writer.get_state()

        draft = state.values.get("draft", "")
        # need to replace file= by empty because of gradio problem in Markdown
        draft = re.sub(r'\/?file=', '', draft)
        plan = state.values.get("plan", "")
        review_topic_sentences = "\n\n".join(state.values.get("review_topic_sentences", []))
        review_instructions = "\n\n".join(state.values.get("review_instructions", []))
        content = "\n\n".join(state.values.get("content", []))

        dir = os.path.splitext(filename)[0]
        try:
            shutil.rmtree(dir)
        except:
            pass
        os.mkdir(dir)
        os.symlink(self.images, dir + "/images")
        base_filename = dir + "/" + dir.split("/")[-1]
        with open(base_filename + ".md", "w") as fp:
            fp.write(draft)
            logging.warning(f"saved file {base_filename + '.md'}")

        html = markdown.markdown(draft)
        with open(base_filename + ".html", "w") as fp:
            fp.write(html)

        try:
            # Use pandoc to convert to docx
            subprocess.run(
                [
                    "pandoc",
                    "-s", f"{base_filename + '.html'}",
                    "-f", "html",
                    "-t", "docx",
                    "-o", f"{base_filename + '.docx'}"
                ])
        except:
            logging.error("cannot find 'pandoc'")

        #with open(base_filename + ".docx", "wb") as fp:
        #    buf = html2docx(html, title=state.values.get("title", ""))
        #    fp.write(buf.getvalue())
        logging.warning(f"saved file {base_filename + '.docx'}")

        with open(base_filename + "_ts.txt", "w") as fp:
            fp.write(review_topic_sentences)
            logging.warning(f"saved file {base_filename + '_ts.txt'}")

        with open(base_filename + "_wi.txt", "w") as fp:
            fp.write(review_instructions)
            logging.warning(f"saved file {base_filename + '_wi.txt'}")

        with open(base_filename + "_plan.md", "w") as fp:
            fp.write(plan)
            logging.warning(f"saved file {base_filename + '_plan.md'}")

        with open(base_filename + "_content.txt", "w") as fp:
            fp.write(content)
            logging.warning(f"saved file {base_filename + '_content.txt'}")

        return f"Saved project {dir}"

    def update_refs(self):
        """
        Updates the reference for Gradio
        :return: list of gr.update objects.
        """
        ref_list = [gr.update() for _ in range(1000)]
        for i in range(len(self.references)):
            ref_list[i] = gr.update(
                value=True,
                visible=True,
                label=self.references[i])
        return [gr.update(
            visible=self.generate_citations and len(self.references) > 0)
        ] + ref_list

    def submit_ref_list(self, *ref_list):
        """
        Invokes step of generating citations with user reference feedback.
        :param ref_list: List of references that were unselected.
        :return: Everything returned by self.update.
        """
        ref_list = ref_list[:len(self.references)]
        state = self.writer.get_state()
        references = [self.references[i] for i in range(len(self.references)) if ref_list[i]]
        logging.warning("Keeping the following references")
        for ref in references:
            logging.warning(ref)
        state.values["references"] = '\n'.join(references)
        self.writer.update_state(state)
        return self.update("")

    def create_ui(self):
        with gr.Blocks(
                theme=gr.themes.Default(),
                fill_height=True) as self.kiroku_agent:
            with gr.Tab("Initial Instructions"):
                with gr.Row():
                    file = gr.File(file_types=[".yaml"], scale=1)
                    js = gr.JSON(scale=5)
            with gr.Tab("Document Writing"):
                out = gr.Textbox(label="Echo")
                inp = gr.Textbox(
                    placeholder="Instruction",
                    label="Rider")
                markdown = gr.Markdown("")
                doc = gr.Button("Save")
            with gr.Tab("References") as self.ref_block:
                ref_list = [
                    gr.Checkbox(
                        value=False,
                        visible=False,
                        label=False,
                        interactive=True)
                    for _ in range(1000)
                ]
                submit_ref_list = gr.Button("Submit", visible=False)

            inp.submit(
                self.update, inp, [markdown, out, inp]).then(
                lambda : gr.update(
                    value="",
                    interactive=self.next_state not in [
                        END, "generate_citations", "NONE"]), [], inp
            ).then(self.update_refs, [], [submit_ref_list] + ref_list)
            file.upload(self.process_file, file, [js, inp]).then(
                self.initial_step, [], [markdown, out]).then(
                lambda : gr.update(placeholder="", interactive=True), [], inp)
            doc.click(self.save_as, [], out)
            submit_ref_list.click(
                self.submit_ref_list,
                ref_list,
                [markdown, out, submit_ref_list])

    def launch_ui(self):
        logging.warning(f"... using KIROKU_PROJECT_DIRECTORY working directory of {self.working_dir}")
        try:
            os.chdir(self.working_dir)
        except:
            logging.warning(f"... directory {self.working_dir} does not exist")
            os.mkdir(self.working_dir)
        self.images = self.working_dir + "/images"
        logging.warning(
            f"... using directory {self.working_dir}/images to store images")
        try:
            os.mkdir(self.images)
        except:
            pass
        self.kiroku_agent.launch(server_name='localhost') #allowed_paths=[working_dir])

def run():
    working_dir = os.environ.get("KIROKU_PROJECT_DIRECTORY", os.getcwd())
    # need this to allow images to be in a different directory
    gr.set_static_paths(paths=[working_dir + '/images'])
    kiroku = KirokuUI(working_dir)
    kiroku.create_ui()
    kiroku.launch_ui()

if __name__ == "__main__":
    n_errors = 0
    if not os.environ.get("OPENAI_API_KEY"):
        logging.error("... We presently require an OPENAI_API_KEY.")
        n_errors += 1
    if not os.environ.get("TAVILY_API_KEY"):
        logging.error("... We presently require an TAVILY_API_KEY.")
        n_errors += 1
    if n_errors > 0:
        exit()

    run()
