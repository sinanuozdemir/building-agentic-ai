#!/usr/bin/env python3
"""
Prompt templates for various agent types
"""

from langchain_core.prompts import ChatPromptTemplate

BOOK_INFO = '''\
Quick Start Guide to Large Language Models: Strategies and Best Practices for ChatGPT, Embeddings, Fine-Tuning, and Multimodal AI (Addison-Wesley Data & Analytics Series) 2nd Edition
by Sinan Ozdemir (Author)
4.6 out of 5 stars

The Practical, Step-by-Step Guide to Using LLMs at Scale in Projects and Products


Large Language Models (LLMs) like Llama 3, Claude 3, and the GPT family are demonstrating breathtaking capabilities, but their size and complexity have deterred many practitioners from applying them. In Quick Start Guide to Large Language Models, Second Edition, pioneering data scientist and AI entrepreneur Sinan Ozdemir clears away those obstacles and provides a guide to working with, integrating, and deploying LLMs to solve practical problems.

Ozdemir brings together all you need to get started, even if you have no direct experience with LLMs: step-by-step instructions, best practices, real-world case studies, and hands-on exercises. Along the way, he shares insights into LLMs' inner workings to help you optimize model choice, data formats, prompting, fine-tuning, performance, and much more. The resources on the companion website include sample datasets and up-to-date code for working with open- and closed-source LLMs such as those from OpenAI (GPT-4 and GPT-3.5), Google (BERT, T5, and Gemini), X (Grok), Anthropic (the Claude family), Cohere (the Command family), and Meta (BART and the LLaMA family).

Learn key concepts: pre-training, transfer learning, fine-tuning, attention, embeddings, tokenization, and more
Use APIs and Python to fine-tune and customize LLMs for your requirements
Build a complete neural/semantic information retrieval system and attach to conversational LLMs for building retrieval-augmented generation (RAG) chatbots and AI Agents
Master advanced prompt engineering techniques like output structuring, chain-of-thought prompting, and semantic few-shot prompting
Customize LLM embeddings to build a complete recommendation engine from scratch with user data that outperforms out-of-the-box embeddings from OpenAI
Construct and fine-tune multimodal Transformer architectures from scratch using open-source LLMs and large visual datasets
Align LLMs using Reinforcement Learning from Human and AI Feedback (RLHF/RLAIF) to build conversational agents from open models like Llama 3 and FLAN-T5
Deploy prompts and custom fine-tuned LLMs to the cloud with scalability and evaluation pipelines in mind
Diagnose and optimize LLMs for speed, memory, and performance with quantization, probing, benchmarking, and evaluation frameworks
"A refreshing and inspiring resource. Jam-packed with practical guidance and clear explanations that leave you smarter about this incredible new field."
--Pete Huang, author of The Neuron'''

# TODO the {tools} is not being replaced with the actual tools.
def get_sdr_system_prompt(extra_instructions: str = '') -> str:
    """Get the prompt template for the SDR agent with required input keys."""
    template = """\
You are an expert Sales Development Representative (SDR) assistant with access to the following tools:

{{tools}}

--------------------------------

Be casual and friendly. Your role is to help sales professionals with:
- Prospect research and identification
- Outreach email generation and optimization
- Campaign tracking and analysis
- Follow-up scheduling and management
- Company intelligence gathering

--------------------------------

The item you are attempting to sell/qualify leads for is:
<<BOOK_INFO>>

--------------------------------

<<extra_instructions>>""".replace('<<extra_instructions>>', extra_instructions).replace('<<BOOK_INFO>>', BOOK_INFO).strip()

    return template
