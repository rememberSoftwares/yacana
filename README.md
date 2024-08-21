# Yacana
Tasks driven agentic framework that gives you the building blocks to create AI workflows for local LLMs.  

## Functionalities
* Built for getting the best results with local opensource LLMs ;  
* Multi Agent workflows ;  
* Tool calling for all models, even dumb ones ;
* Multi turn chat between Agents

## Why choose Yacana over other frameworks ?

Yacana was built to make local LLMs great ! It won't work with chatGPT, and we don't intend to make it. An important part of the future lies in local opensource models. Many companies cannot afford paied subscriptions to proprietary models or won't for privacy issues. Also agentic framework tend to cost way more than simple chat, generating cloud bills that could ruin a business. But why pay for AI when you can have it for only a fraction of the cost hosted on a single GPU machine or even on the device itself ? This is where Yacana can help you harness the power of LLMs whatever the size and cachieve production grade results.

Currently the best AI frameworks in their category are CrewAI and LangGraph.  
* CrewAI is very simple to use but is terrible when dealing with local models. This is because most of the heavy lifting is left to the LLM and if chatGPT or any other proprietary frontier models can deal with complex prompting, local LLMs cannot. Can we really expect to have the same performance from 8 Billion parameters model than proprietary model that have over 1000 Billion parameters ? it's impossible. That's why CrewAI doesn't do well with local LLMs and also. This means that it cannot call tools ! And calling tool is the most useful functionality a framework should offer.  

**So, why Yacana ?** We loved how easy creating agents in CrewAI was. So when building Yacana it was clear to us that it needed the same level of simplicity. Learning to program with Yacana is very easy and only takes a few minutes. Also, the tool calling feature was of paramount importance. That's why we worked on making Yacana's tool calling ability so good that even tiny LLMs like Phi can call tools with almost 100% chance of success.

* LangGraph is one of the best tool of 2024 to work with local models. It allows to create complex workflows that guide the LLM along a specific route to achieve a specific result. The problem with langGraph is its learning curve that is far above many developers and the spotty documentation doesn't help. It does allow tool calling but only with models that support function calling natively. Learning LangGraph has a one in three chance to give you a burnout. Also, the graph definition, beside being a great conceptual idea, is poorly implemented (IMHO) and the fact that they introduced a web UI to control the framework seems as an admission of failure (even though the UI looks great). But as a developer I don't want to rely on yet another tool.

**So, why Yacana ?** The LangGraph nodes system that guides LLMs in a graph is great and is the reason why it succeeds with local LLMs. Knowing this we build Yacana with the exact same concept. But instead of having a complex graph system Yacana provides a classic programming interface in Python that can be leverage by any developers regardless of their level of expertise. No burn outs here. So, Yacana provides its own way to chain Tasks and achieve a computable result in the end that can be used in classic programming products. This makes Yacana a production ready tool that can be easily integrated with any software.

**In summary...** Yacana took the simplicity of CrewAI, and it's capability to have multi-agent chats but following the LangGraph principle of guidance using a dedicated Task system that guaranties a computable result at the end of the workflow. Also, Yacana is the only Framework proposing a tool calling system with a high success rate.  
Also, it's free and opensource (MIT) forever. It is made by the community, for the community.  

**What's the catch ?** Well nothing really. But for transparency it is important that you understand that this version has only been tested with Ollama as inference server. We know that you guys out there are also using other inference technologies like VLLM, etc. Yacana has not been tested with this type of tech... Yet ! The next update will focus on making sure that performance when connected to something other than Ollama doesn't degrade in any way. Stay tuned. This will be addressed in the next few weeks. Also note that we are not saying that it's not compatible with other inference servers. We're saying that we don't know yet. ^^

## Quickstart

### Installation

As mentionned previously, the current version of Yacana has only been tested on Ollama. This will be sorted out in the next few weeks.  
To work with Yacana you'll need an Ollama server. If your computer is performant enough you can use it to run LLMs directly on your machine. Ollama is a great inference server and the most simple to install. Click [here](https://ollama.com/download) to get the latest release.
Ollama is:
* Compatible with all operating systems Windows/Max/Linux ;  
* Gets installed using one command ;  
* Has a great CLi that even a 4 year old can use to download models ;  
* Tons of tutorial out there if you ran in any type of trouble ;

ℹ️ You can connect Yacana to a remote Ollama instance. Read forward.  

---

When Ollama is installed you can browse the list of available LLMs on the Ollama website

### Imports

### Creating your first Agent

### Creating your first Task

### Managing Agents history

### Assigning a tool to a Task

### Assigning multiple Tools

### Chat between two Agents

### Chat between many Agents
