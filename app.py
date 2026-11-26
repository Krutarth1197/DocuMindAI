import argparse, html
from pathlib import Path
import gradio as gr
from src.pipeline import initialize, documind_qa_v5
STATUS_DESCRIPTIONS = {

    "structurally_validated_answer":
        "Answer generated with structurally valid citations. "
        "Factual accuracy requires verification.",

    "rejected_low_relevance":
        "Question rejected because retrieval relevance "
        "was below the configured threshold.",

    "model_abstained":
        "The language model reported insufficient evidence.",

    "evidence_fallback":
        "Generated answer failed structural validation. "
        "Retrieved evidence is displayed instead.",

    "evidence_only":
        "Retrieved evidence displayed without LLM generation.",

    "no_context":
        "No usable context was retrieved."
}

def app_query(query, mode):

    if not query or not query.strip():

        return (
            "Please enter a research question.",
            "",
            "",
            ""
        )

    try:

        result = documind_qa_v5(

            query=query,

            top_k=3,

            max_new_tokens=96,

            evidence_only=(
                mode == "Evidence Search"
            )
        )

        answer = result["answer"]

        source_sections = []

        for source in result["sources"]:

            section = (

                f"### [{source['source_id']}] "
                f"{source['document']}\n\n"

                f"**PDF page:** {source['page']}\n\n"

                f"**Chunk ID:** {source['chunk_id']}\n\n"

                f"{html.escape(source['text'])}\n\n"

                "---"
            )

            source_sections.append(section)

        sources_text = (

            "\n\n".join(source_sections)

            if source_sections

            else "No sources displayed."
        )

        status_text = STATUS_DESCRIPTIONS.get(

            result["status"],

            result["status"]
        )

        if result["validation_issues"]:

            status_text += (

                "\n\nValidation issues: "

                + ", ".join(
                    result["validation_issues"]
                )
            )

        relevance = result["top_relevance_score"]

        relevance_display = (

            f"{relevance:.4f}"

            if relevance is not None

            else "N/A"
        )

        metrics = f"""
### Performance statistics

| Metric | Value |
|---|---|
| Pipeline version | V5 |
| Retrieval time | {result['retrieval_time']:.2f} s |
| Generation time | {result['generation_time']:.2f} s |
| Total response time | {result['total_time']:.2f} s |
| Relevance score | {relevance_display} |
| Relevance threshold | 0.0 |
| Retrieved passages | {len(result['sources'])} |
| Generated tokens | {result['generated_tokens']} |
"""

        return (

            answer,

            sources_text,

            status_text,

            metrics
        )

    except Exception as error:

        print("Application error:", repr(error))

        return (

            "An error occurred while processing the question.",

            "",

            "Check the Colab runtime for error details.",

            ""
        )

APP_EXAMPLES = [

    [
        "How does multi-head attention work in Transformers?",
        "Evidence Search"
    ],

    [
        "How does BERT select and replace tokens during masked language model training?",
        "AI Question Answering"
    ],

    [
        "Why is Sentence-BERT computationally efficient?",
        "Evidence Search"
    ],

    [
        "What is the difference between RAG-Sequence and RAG-Token?",
        "Evidence Search"
    ],

    [
        "How does DPR calculate similarity?",
        "AI Question Answering"
    ],

    [
        "What is the exact parameter count of GPT-4?",
        "AI Question Answering"
    ]
]

def build_demo():
    with gr.Blocks(
        title="DocuMind AI"
    ) as demo:
    
        gr.Markdown(
            """
    # DocuMind AI
    
    ### A Retrieval-Augmented Generation Framework for Intelligent Document Understanding and Question Answering
    
    **5 Research Papers | 74 PDF Pages | 433 Indexed Passages**
    
    Hybrid Retrieval • Cross-Encoder Reranking • Relevance Filtering • Source Citations
    
    ---
    """
        )
    
        with gr.Row():
    
            with gr.Column(scale=2):
    
                question_input = gr.Textbox(
    
                    label="Research Question",
    
                    placeholder=(
                        "Ask a question about the indexed "
                        "research papers..."
                    ),
    
                    lines=3
                )
    
                mode_input = gr.Radio(
    
                    choices=[
                        "AI Question Answering",
                        "Evidence Search"
                    ],
    
                    value="Evidence Search",
    
                    label="Operating Mode"
                )
    
                with gr.Row():
    
                    submit_button = gr.Button(
    
                        "Analyze Question",
    
                        variant="primary"
                    )
    
                    clear_button = gr.Button(
                        "Clear"
                    )
    
            with gr.Column(scale=1):
    
                gr.Markdown(
                    """
    ### Indexed research papers
    
    1. Attention Is All You Need
    2. BERT
    3. Sentence-BERT
    4. Retrieval-Augmented Generation
    5. Dense Passage Retrieval
    
    ### Operating modes
    
    **AI Question Answering:** Generates an answer using retrieved evidence.
    
    **Evidence Search:** Returns supporting passages without language model generation.
    
    AI generation may take several minutes on CPU.
    """
                )
    
        gr.Markdown("---")
    
        answer_output = gr.Textbox(
    
            label="Research Answer",
    
            lines=8,
    
            interactive=False
        )
    
        status_output = gr.Textbox(
    
            label="System Status",
    
            lines=3,
    
            interactive=False
        )
    
        with gr.Tabs():
    
            with gr.Tab("Supporting Sources"):
    
                sources_output = gr.Markdown()
    
            with gr.Tab("Performance Statistics"):
    
                metrics_output = gr.Markdown()
    
            with gr.Tab("Research Evaluation"):
    
                gr.Markdown(
                    """
    ## Baseline experimental results
    
    ### Retrieval benchmark
    
    25 manually annotated questions.
    
    | Retrieval method | Hit@5 |
    |---|---:|
    | BM25 | 76% |
    | Dense Semantic | 40% |
    | Hybrid RRF | 72% |
    | Hybrid + Cross-Encoder | 80% |
    
    ### Relevance gate evaluation
    
    - Evaluation questions: 35
    - Accuracy: 100%
    - F1 score: 1.000
    
    ### Baseline generation evaluation
    
    10 answerable questions and 5 unanswerable questions.
    
    - Factual correctness rubric score: 75%
    - Completeness rubric score: 50%
    - Citation support rubric score: 25%
    - Valid citation presence: 60%
    - Correct negative abstention: 5/5
    
    These are baseline experimental results, not independently
    validated performance measurements of Pipeline V5.
    
    Manual answer assessment was LLM-assisted and remains
    subject to independent human verification.
    """
                )
    
        gr.Examples(
    
            examples=APP_EXAMPLES,
    
            inputs=[
                question_input,
                mode_input
            ],
    
            cache_examples=False
        )
    
        gr.Markdown(
            """
    ---
    
    ### Research prototype — Version 5
    
    **Technologies:** Python, PyMuPDF4LLM, Sentence Transformers,
    FAISS, BM25, Reciprocal Rank Fusion, Cross-Encoder,
    Qwen 2.5 and Gradio.
    
    **Disclaimer:** This system is an experimental research
    prototype. Generated answers may contain factual errors.
    Valid citation IDs do not guarantee factual support.
    Always verify important claims against the original papers.
    """
        )
    
        submit_button.click(
    
            fn=app_query,
    
            inputs=[
                question_input,
                mode_input
            ],
    
            outputs=[
                answer_output,
                sources_output,
                status_output,
                metrics_output
            ]
        )
    
        question_input.submit(
    
            fn=app_query,
    
            inputs=[
                question_input,
                mode_input
            ],
    
            outputs=[
                answer_output,
                sources_output,
                status_output,
                metrics_output
            ]
        )
    
        clear_button.click(
    
            fn=lambda: ("", "", "", "", ""),
    
            outputs=[
                question_input,
                answer_output,
                sources_output,
                status_output,
                metrics_output
            ]
        )
    
    print("DocuMind AI interface constructed!")
    return demo

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path("."))
    parser.add_argument("--share",action="store_true")
    args=parser.parse_args()
    initialize(args.root)
    build_demo().queue(default_concurrency_limit=1).launch(share=args.share)
