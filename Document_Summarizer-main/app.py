    

import gradio as gr
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load summarizer model (LaMini-Flan-T5)
summarizer_tokenizer = AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
summarizer_model = AutoModelForSeq2SeqLM.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")

# Load translators
translator_hi = pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
translator_te = pipeline("translation", model="Helsinki-NLP/opus-mt-en-mul")

# Extract text from PDF
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Summarize based on doc type
def summarize_text(text, doc_type):
    prompt = f"Summarize this {doc_type} document clearly:\n{text}\nSummary:"
    inputs = summarizer_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = summarizer_model.generate(**inputs, max_length=300, num_beams=4, early_stopping=True)
    return summarizer_tokenizer.decode(outputs[0], skip_special_tokens=True)

# Translate summary
def translate_summary(summary, lang):
    if lang == "hindi":
        return translator_hi(summary)[0]["translation_text"]
    elif lang == "telugu":
        return translator_te(summary)[0]["translation_text"]
    else:
        return summary  # English or unsupported

# Main processing logic
def process(file, lang, doc_type):
    text = extract_text_from_pdf(file)
    if not text.strip():
        return "Error: PDF has no extractable text."
    
    summary = summarize_text(text, doc_type)
    return translate_summary(summary, lang)

# Gradio UI
with gr.Blocks() as app:
    gr.Markdown("## Multilingual AI Document Summarizer")
    gr.Markdown("Upload a document and get summaries in multiple languages using mT5.")

    file_input = gr.File(label="Upload PDF")

    with gr.Row():
        language_input = gr.Dropdown(
            label="Select Language",
            choices=["english", "hindi", "telugu"],
            value="english"
        )
        type_input = gr.Dropdown(
            label="Select Document Type",
            choices=["legal", "medical", "general"],
            value="general"
        )

    output = gr.Textbox(label="Summary Output", lines=10)

    with gr.Row():
        clear = gr.Button("Clear")
        submit = gr.Button("Submit")

    submit.click(fn=process, inputs=[file_input, language_input, type_input], outputs=output)
    clear.click(lambda: "", inputs=[], outputs=output)

app.launch()
