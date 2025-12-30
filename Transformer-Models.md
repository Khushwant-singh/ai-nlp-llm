
# Transformer Models → Hugging Face Mapping (Cheat Sheet)

----------

## 1️⃣ Encoder-Only Transformers

**Role:** Text understanding (no generation)

**Input → Output**

`Text → Label / Vector` 

**Hugging Face Models**

-   `bert-base-uncased`
    
-   `roberta-base`
    
-   `microsoft/deberta-v3-base`
    

**Typical Pipelines**

-   `text-classification`
    
-   `token-classification`
    
-   `feature-extraction`
    

**Use When**

-   Intent classification
    
-   Sentiment analysis
    
-   Semantic similarity
    
-   Lightweight embeddings
    

**One-Line Summary**

> Best at **reading and understanding text**, not writing it.

----------

## 2️⃣ Decoder-Only Transformers

**Role:** Text generation & reasoning

**Input → Output**

`Prompt → Generated  text` 

**Hugging Face Models**

-   `meta-llama/Llama-2-7b-chat-hf`
    
-   `mistralai/Mistral-7B-Instruct-v0.2`
    
-   `tiiuae/falcon-7b-instruct`
    

**Typical Pipelines**

-   `text-generation`
    

**Use When**

-   Chatbots
    
-   Question answering
    
-   Reasoning tasks
    
-   RAG answer generation
    

**One-Line Summary**

> Best at **generating text and reasoning step by step**.

----------

## 3️⃣ Encoder–Decoder Transformers

**Role:** Transform one text into another

**Input → Output**

`Source  text → Target text` 

**Hugging Face Models**

-   `facebook/bart-large-cnn`
    
-   `t5-base`
    
-   `google/pegasus-cnn_dailymail`
    

**Typical Pipelines**

-   `summarization`
    
-   `translation`
    
-   `text2text-generation`
    

**Use When**

-   Summarization
    
-   Translation
    
-   Paraphrasing
    
-   Classic QA tasks
    

**One-Line Summary**

> Best for **structured text-to-text transformations**.

----------

## 4️⃣ Text-to-Text Transformers

**Role:** One model for many NLP tasks

**Input → Output**

`Text → Text` 

**Hugging Face Models**

-   `google/flan-t5-base`
    
-   `google/flan-t5-large`
    

**Typical Pipelines**

-   `text2text-generation`
    

**Use When**

-   Multi-task NLP
    
-   Classification as text labels
    
-   Instruction-based tasks
    

**One-Line Summary**

> Every task is framed as **text in → text out**.

----------

## 5️⃣ Embedding-Only Transformers

**Role:** Semantic vectors (core of RAG)

**Input → Output**

`Text → Embedding vector` 

**Hugging Face Models**

-   `sentence-transformers/all-MiniLM-L6-v2`
    
-   `sentence-transformers/all-mpnet-base-v2`
    
-   `BAAI/bge-base-en-v1.5`
    

**Typical Pipelines**

-   `feature-extraction`
    
-   SentenceTransformers API
    

**Use When**

-   Vector databases
    
-   Semantic search
    
-   RAG retrieval
    
-   Recommendation systems
    

**One-Line Summary**

> Converts meaning into **searchable vectors**.

----------

## 6️⃣ Multimodal Transformers

**Role:** Text + image understanding

**Input → Output**

`Text + Image → Text` 

**Hugging Face Models**

-   `openai/clip-vit-base-patch32`
    
-   `Salesforce/blip-image-captioning-base`
    
-   `llava-hf/llava-1.5-7b-hf`
    

**Typical Pipelines**

-   `image-to-text`
    
-   `zero-shot-image-classification`
    
-   `visual-question-answering`
    

**Use When**

-   Image captioning
    
-   Document understanding
    
-   Visual Q&A
    
-   Multimodal assistants
    

**One-Line Summary**

> Transformers that **see and understand text together**.

----------

## 7️⃣ Instruction-Tuned Transformers

**Role:** Follow human instructions

**Input → Output**

`Instruction → Helpful response` 

**Hugging Face Models**

-   `mistralai/Mistral-7B-Instruct-v0.2`
    
-   `meta-llama/Llama-2-7b-chat-hf`
    
-   `google/flan-t5-xl`
    

**Typical Pipelines**

-   `text-generation`
    
-   `text2text-generation`
    

**Use When**

-   Chat assistants
    
-   Tool-using agents
    
-   RAG answer generation
    

**One-Line Summary**

> Trained to **do what you ask**, not just predict text.

----------

## 8️⃣ Retrieval-Augmented Generation (RAG Systems)

**Role:** Answer using private knowledge

**Input → Output**

`Question + Retrieved context → Answer` 

**Common Hugging Face Combinations**

-   `sentence-transformers/all-MiniLM-L6-v2` (retrieval)
    
-   `meta-llama/Llama-2-7b-chat-hf` (generation)
    
-   `mistralai/Mistral-7B-Instruct-v0.2` (generation)
    

**Vector Stores**

-   FAISS
    
-   Chroma
    
-   Weaviate
    

**Use When**

-   Private knowledge bases
    
-   Enterprise chatbots
    
-   Documentation Q&A
    

**One-Line Summary**

> **LLM + embeddings + vector DB = grounded answers**

----------

## 🧠 One-Glance Mental Map

Goal

Hugging Face Model Type

Understand text

Encoder-only

Generate text

Decoder-only

Transform text

Encoder–Decoder

Multi-task NLP

Text-to-Text

Search meaning

Embedding models

Text + image

Multimodal

Follow instructions

Instruction-tuned

Use private data

RAG system

----------

## 🔑 Final Takeaway (Save This)

> **Modern AI apps don’t use one Transformer — they use a system.**  
> Embeddings retrieve knowledge, LLMs reason over it, and RAG connects them safely.
