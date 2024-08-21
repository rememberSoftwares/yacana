# Yacana
Task driven LLM multi-agents framework that gives you the building blocks to create AI workflows for local LLMs.

## Functionnalities
* Built for getting the best results with local opensource LLMs ;  
* Multi Agent workflows ;  
* Tool calling for all models, even dumb ones ;
* Multi turn chat between Agents

## Why choose Yacana over other frameworks ?

Yacana was built to make local LLMs great ! It won't work with chatGPT and we don't intend to make it.  
Currently the best AI frameworks in their category are CrewAI and LangGraph.  
* CrewAI is very simple to use but is terrible when dealing with local models. This is because most of the heavy lifting is left to the LLM and if chatGPT or any other proprietary frontier models can deal with complexe prompting, local LLMs cannot. Can we really expect to have the same performance from 8 Billion parameters model than proprietary model that have over 1000 Bilion parameters ? it's impossible. That's why CrewAI doesn't do well with local LLMs and also. This means that it cannot call tools ! And calling tool is the most useful funtionnality a framework should offer.  
**So why Yacana ?** We loved how easy creating agents in CrewAI was. So when building Yacana it was clear to us that it needed the same level of simplicity. Learning to program with Yacana is very easy and only takes a few minutes. Also the tool calling feature was of paramount importance. That's why we worked on making Yacana's tool calling ability so good that event tiny LLMs like Phi can call tools with almost 100% chance of success.
* LangGraph is one of the best tool of 2024 to work with local models. It allows to create complex worflows that guide the LLM allong a specific route to achieve a specific result. The problem with langGraph is its learning curve that is far above many developers and the spotty documentation doesn't help. It does allow tool calling but only with models that support function calling natively. Learning LangGraph has a one in three chance to give you a burnout. Also the graph definition, beside being a great conceptual idea, is poorly implemented (IMHO) and the fact that they introduced a web UI to control the framework seems as an admission of failure (even though the UI looks great). But as a developper I don't want to rely on yet another tool. 



## Quickstart

### Installation

### Imports

### Creating your first Agent

### Creating your first Task

### Managing Agents history

### Assigning a tool to a Task

### Assigning multiple Tools

### Chat between two Agents

### Chat between many Agents
