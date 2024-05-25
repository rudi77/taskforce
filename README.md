# Taskforce: LLM-Based Autonomous Agent Framework

Welcome to Taskforce! This framework leverages large language models (LLMs) to create autonomous agents capable of performing complex tasks. Our framework integrates planning, memory, and various tools to enable sophisticated processing and decision-making workflows.

## Introduction

Large Language Models (LLMs) have revolutionized the field of natural language processing by enabling machines to understand and generate human-like text. By integrating LLMs into an autonomous agent system, we can create intelligent agents that can read, understand, and extract information from documents autonomously.

## Framework Overview

Taskforce is designed with a modular architecture, allowing for flexibility and scalability. The core components of the system include:

- **Agent**: The central coordinator that manages the workflow.
- **Planner**: Responsible for planning the sequence of actions required to achieve the agent's goal.
- **Memory**: Manages both short-term and long-term memory to store intermediate and historical data.
- **Tools**: Various auxiliary tools for specific tasks (e.g., OCR, layout detection).
- **LLM**: Provides natural language understanding and processing capabilities.

### Agent Responsibilities

The Agent is the heart of the system, orchestrating the interactions between other components. It receives documents, processes them using the LLM, utilizes tools, stores relevant information in memory, and generates the final output.

### Planner

The Planner component determines the sequence of actions needed to process a document. It interacts with the LLM to generate a series of sub-tasks based on the user's prompt and the provided instructions.

### Memory

The Memory component is split into short-term and long-term memory. Short-term memory stores intermediate results, while long-term memory archives important information for future reference.

### Tools

The framework includes various tools that the Agent can use to perform specific tasks. These tools are essential for tasks like OCR, layout detection, and web searching.

### LLM

The LLM component handles the interaction with the large language model, enabling the Agent to process natural language text effectively.


## Contribution

## Future Development
Taskforce is at the beginning of its journey and will evolve over time. We plan to add more features, and enhance the capabilities of the framework. Stay tuned for updates and new releases!
