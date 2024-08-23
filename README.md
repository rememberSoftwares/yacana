# Yacana
Tasks driven agentic framework that gives you the building blocks to create AI workflows for local LLMs.  

# Functionalities
* Built for getting the best results with local opensource LLMs ;  
* Multi Agent workflows ;  
* Tool calling for all models, even dumb ones ;
* Multi turn chat between Agents ;
* Routing ;

...and many more !

# Why choose Yacana over other frameworks ?

Yacana was built to make local LLMs great ! It won't work with chatGPT, and we don't intend to make it. An important part of the future lies in local opensource models. Many companies cannot afford paied subscriptions to proprietary models or won't for privacy issues. Also agentic framework tend to cost way more than simple chat, generating cloud bills that could ruin a business. But why pay for AI when you can have it for only a fraction of the cost hosted on a single GPU machine or even on the device itself ? This is where Yacana can help you harness the power of LLMs whatever the size and cachieve production grade results.

> What other good framework are out there ?  
* **CrewAI** is very simple to use but is terrible when dealing with local models. This is because most of the heavy lifting is left to the LLM and if chatGPT or any other proprietary frontier models can deal with complex prompting, local LLMs cannot. Can we really expect to have the same performance from 8 Billion parameters model than proprietary model that have over 1000 Billion parameters ? it's impossible. That's why CrewAI doesn't do well with local LLMs and also. This means that it cannot call tools ! And calling tool is the most useful functionality a framework should offer.  

➡️**Why would Yacana be better suited ?** We loved how easy creating agents in CrewAI was. So when building Yacana it was clear to us that it needed the same level of simplicity. Learning to program with Yacana is very easy and only takes a few minutes. Also, the tool calling feature was of paramount importance. That's why we worked on making Yacana's tool calling ability so good that even tiny LLMs like Phi can call tools with almost 100% chance of success.

* **LangGraph** is one of the best tool of 2024 to work with local models. It allows to create complex workflows that guide the LLM along a specific route to achieve a specific result. The problem with langGraph is its learning curve that is far above many developers and the spotty documentation doesn't help. It does allow tool calling but only with models that support function calling natively. Learning LangGraph has a one in three chance to give you a burnout. Also, the graph definition, beside being a great conceptual idea, is poorly implemented (IMHO) and the fact that they introduced a web UI to control the framework seems as an admission of failure (even though the UI looks great). But as a developer I don't want to rely on yet another tool.

➡️**Why would Yacana be better suited ?** The LangGraph nodes system that guides LLMs in a graph is great and is the reason why it succeeds with local LLMs. Knowing this we build Yacana with the exact same concept. But instead of having a complex graph system Yacana provides a classic programming interface in Python that can be leverage by any developers regardless of their level of expertise. No burn outs here. So, Yacana provides its own way to chain Tasks and achieve a computable result in the end that can be used in classic programming products. This makes Yacana a production ready tool that can be easily integrated with any software.

**In summary...** Yacana took the simplicity of CrewAI, and it's capability to have multi-agent chats but following the LangGraph principle of guidance using a dedicated Task system that guaranties a computable result at the end of the workflow. Also, Yacana is the only Framework proposing a tool calling system with a high success rate.  
Also, it's free and opensource (MIT) forever. It is made by the community, for the community.  

**What's the catch ?** Well nothing really. But for transparency it is important that you understand that this version has only been tested with Ollama as inference server. We know that you guys out there are also using other inference technologies like VLLM, etc. Yacana has not been tested with this type of tech... Yet ! The next update will focus on making sure that performance when connected to something other than Ollama doesn't degrade in any way. Stay tuned. This will be addressed in the next few weeks. Also note that we are not saying that it's not compatible with other inference servers. We're saying that we don't know yet. ^^

# Quickstart

## I. Installation

### Installing Ollama

![image](https://github.com/user-attachments/assets/f3c45d0e-efca-4853-8237-3e56d90e1747)

As mentionned previously, the current version of Yacana has only been tested on Ollama. This will be sorted out in the next few weeks.  
To work with Yacana you'll need an Ollama server. If your computer is performant enough you can use it to run LLMs directly on your machine. Ollama is a great inference server and the most simple to install. Click [here](https://ollama.com/download) to get the latest release.
Ollama is:
* Compatible with all operating systems Windows/Max/Linux ;  
* Gets installed using one command ;  
* Has a great CLi that even a 4 year old can use to download models ;  
* Tons of tutorial out there if you ran in any type of trouble ;

ℹ️ You can connect Yacana to a remote Ollama instance. Read forward.  

### Choosing an LLM model

After Ollama is installed you can browse the list of available LLMs on the [Ollama website](https://ollama.com/library) and download any model you want (or you computer can deal with).  
For reference if you don't know what LLM model to chose (we've all been there) here is a list of models you can try out on consumer hardware:  
| Computer power   |      LLM models name to try      |  LLM quality |
|----------|:-------------:|:------:|
| **Out of this world** *(GC RTX 4090 / 64 GB RAM)* | 'llama3.1:70b' or 'mixtral:8x22b' | Excellent reasoning and instruction following. |
| **Epic** *(GC RTX 4090 / 32 GB RAM)* | 'llama3.1:8b' or 'dolphin-mixtral:8x7b' or 'dolphin-mixtral:8x7b-v2.5-q6_K' | Good reasoning and instruction following. (q6_K model should be less consumming than the default Mixtral if you have any issues) |
| **Gamer** *(GC GTX 1080TI / 16 GB RAM)* |    'llama3.1:8b' or 'mistral:7b'   | Llama still works but is slower. Expect limited reasonning and no more than 2 complex instruction at a time  |
| **Potato** | 'phi:2.7b' or 'phi3:3.8b' or 'tinyllama:1.1b' | Almost no reasoning, incapable of following more than 1 instruction at a time, english bound only ; Dumb as a stone |

---

If you have access to a GPU VM with A LOT of RAM than you should try the state of the art 'llama3.1:405b' model.

### Running the model

When you have chosen your model it is time to use Ollama CLI to pull it on your computer. Simply do `ollama pull <model_name>`. The download will start immediatly. When finished you can test the model locally by doing `ollama run <model_name>` which will start a conversation with the LLM.

### Installing Yacana

```
pip3 install yacana
```

## II. Imports

When using other frameworks 'import hell' quickly appears. To prevent this bothersome problem we propose that you import all of Yacana's modules and when finished developping let the IDE remove the unsued imports. Unused imports generaly appear grayed. Thus we recommand that you prepend these imports in all your files and clean them later. This way the IDE will have auto competion available and will help you develop 10 times faster.

## III. Creating your first Agent

### Initializing the Agent
Now that you have an Ollama server running and Yacana installed let's create are first agent !  

Create a python file with this content:
```python
agent1 = Agent("AI assistant", "llama3.1:8b", system_prompt="You are a helpful AI assistant", endpoint="http://127.0.0.1:11434")
```
The Agent(...) class takes 2 mandatory parameters:
1. **The agent name** : Chose something short and in relation to what the agent global focus
2. **A model name** : The Ollama model that this Agent will use. You may have multiple Agents running different models. Some models are better suitted for some specific jobs so it can be interesting to mix LLM models.

The Agent(...) class has many optionnal parameters that we will discover in this tutorial. Here we can see 2 of them:  
1. **The system prompt** : Helps defining the personnality of the Agent.
2. **The endpoint** : The URL of your Ollama instance. It points by default to your localhost and on the Ollama default port. If you are using Ollama on your computer you can remove this optionnal parameter and the default value will be used.

### Testing Yacana and Ollama interraction

This framework is not meant for basic roleplay. However for people starting their journey in the realm of AI and for debug purposes we added a simple chat system. Add this line to test it :
```python
agent1.simple_chat()
```
When running this python file you should enter a chat with the LLM. The Agent keeps track of the history so that it can answer using past information.
```shell
$ python3 simple_chat_demo.py
```
Output:
```
Type 'quit' then enter to exit.
> hey
Hey! It's nice to meet you. Is there something I can help you with, or would you like to chat about something in particular? I'm here to assist you with any questions or topics you'd like to discuss.
>
```

Let's change the **system prompt** and have some fun !
```python
agent1 = Agent("Pirate", "llama3:8b", system_prompt="You are a pirate", endpoint="http://127.0.0.1:11434")
```
Output:
```
Type 'quit' then enter to exit.
> hey
Arrrr, shiver me timbers! What be bringin' ye to these fair waters? Are ye lookin' fer a swashbucklin' adventure or just passin' through?
> Searching for the tresor of red beard any idea where it's hidden ?
Red Beard's treasure, ye say? (puffs on pipe) Well, I be knowin' a thing or two about that scurvy dog and his loot. But, I'll only be tellin' ye if ye be willin' to share yer own booty... of information! (winks)
```

### Complete section code 

```python
# @todo Imports

agent1 = Agent("Pirate", "llama3:8b", system_prompt="You are a pirate", endpoint="http://127.0.0.1:11434")
agent1.simple_chat()
```

⚠️From now on we will not set the *endpoint* attribute anymore for clarity and will resort to the defaults. If your LLM is not served by Ollama or isn't on your localhost you should continue setting this value.

## IV. Creating your first Task

### Introduction example

The whole concept of the framework lies here. If gou get this following section then you have mastered 80% of Yacana's building principle. Like in LangGraph where you create nodes that you link together, Yacana has a Task() class wich takes as arguments a task to solve. There are no hardcoded links between the Tasks so it's easy to refactor and move things arround. The important concept to grasp here is that through these Task() classes you will give instructions to the LLM in a way that the result must be a computable result. So instructions must be clear and the prompt to use must reflect that. It's Task, it's a job, it's something that needs solving but wrote like it is given as an order ! Let's see some examples :

```python
# First, let's make a basic AI agent
agent1 = Agent("AI assistant", "llama3:8b", system_prompt="You are a helpful AI assistant")

# Now we create a task and assign the agent1 to the task
task1 = Task(f"Solve the equation 2 + 2 and output the result", agent1)

# So that something actually happens you must call the .solve() method on your task
task1.solve()
```

What's happening above ?  
* First, we instanciated an Agent with the `llama3:8b` model. You might need to update that depending on what LLM you downloaded from Ollama ;
* Second, we instanciated a Task ;
* Third, we asked that the Task be solved ;

ℹ️ For easing the learning curve the default logging level is INFO. It will show what is going on in Yacana. Note that not ALL intermediary prompts are shown.

Output of executing should look like this:
```
INFO: [PROMPT]: Solve the equation 2 + 2 and output the result

INFO: [AI_RESPONSE]: The answer to the equation 2 + 2 is... (drumroll please)... 4!

So, the result of solving the equation 2 + 2 is indeed 4.
```

If your terminal is working normaly you should see task's prompts in green and starting with the '[PROMPT]' string. The LLM's answer should appear purple and start with the [AI_RESPONSE] string.  

### About Task parameters

The Task class takes 2 mandatory parameters:
* The prompt : It is the task to be solved. Use imperative language, be precise, ask for step by step thinking for complexe Tasks and expected outputs if needed.
* The Agent : The agent that will be assigned to this task. The agent will be in charge of solving the task.

ℹ️ There are many other parameters that can be given to a Task. We will see some of them in the following sections of this tutorial. But you can checkout the Task class documentation @todo URL

### Do you see how disruptiv this is compared to other Frameworks ?

In the above code snippet we assigned the agent to the Task. So it's the Task that leads the direction that the AI takes. In most other frameworks it's the opposite, where you assign some work to an existing agent. This reversed way allows to have fined grained control on each resolution step as the LLM only follow bread crumb (the Tasks). The patern will become even more obvious as we get to the Tool section of this tutorial. As you'll see the Tools are also assigned at the Task level and not to the Agent directly.  

TO compare with LangGraph, it's true that we canno't make a call graph as we don't bind the Task together explicitly. However, Yacana's way gives more flexibility and allows a hierarchical programing way or ordonancing the Task and keeping control of the flow. It also allows creating new Task dynamically if the need arised. You shall rely on your programming skill and good OOP to have a clean code and a good Task ordering.

### Getting the result of a Task

Even though we get logs on the standard output of the terminal, we still need to extract the answer of the LLM that solved that Task to actualy do something with it.  
Getting the string message is quite easy as the .solve() methods returns a Message() class.  
Maybe you are thinking "ho noo another class to deal with". Well let me tell you that it's always better to have an OOP class than somme semi-random python dictionnary where you'll forget what keys it contain. Also the Message class is very straightforward. It exposes a `content` attribute. Modify the curent code like this:
```python
# So that something actually happens you must call the .solve() method on your task
my_message: Message = task1.solve()

# Printing the LLM's response
print(f"The AI response to our task is : {my_message.content}")
```
There you go ! Give it a try.

ℹ️ Note that we used duck typing, wich is postfixing all variables declaration with their type `my_message: Message`. Yacana's source code is entirely duck typed so that your IDE always know what type it's dealing with and proposes you with the best methods and arguments. We recommand that you do the same as  it's the industry's best strandard.

---

Dont like having 100 lines of code for something simple ? Then chain them all in one line !
```
# First, let's make a basic AI agent
agent1 = Agent("AI assistant", "llama3:8b", system_prompt="You are a helpful AI assistant")

# Creating the task, solving it, extracting the result, printint the result
print(f"The AI response to our task is : {Task(f'Solve the equation 2 + 2 and output the result', agent1).solve().content}")
```
🤔 If I were you I would do the message extraction on one line and the print on a new one. Let's not one-line things that much 😅.


### Chaining Tasks

Chaining Tasks is nothing more than just calling a second Task with the same Agent. Agents keep track of the History of what they did (aka the Tasks they solved). So just call a second Task and assign the same Agent. For instance let's multiply by 2 the result of initial Task. Append this to our current script:

```
task2_result: str = Task(f'Multiply by 2 our previous result', agent1).solve().content
print(f"The AI response to our task is : {task2_result}")
```
You should get:
```
The AI response to our task is : If we multiply the previous result of 4 by 2, we get:

8
```

ℹ️ Without tools this only relies on the LLM ability to do the maths and is dependant to it's training.

---

See ? The assigned Agent remembered that it solve the task1 previously and used this information to solve the second task.  
You can chain as many Task as you need. Also, you should create other Agents that don't have the knowledge of previous Task and make them do things based on the output of your first agent. You can build anything now !    

### Routing

Other frameworks have the tedency to make abastraction for everything. Even thing that dont need any. That's why I'll show you how to do routing whith only what we have seen earlier. Yacana doesn't provide routing abstraction because there is no need to do so.  

But what is routing ? Well, having LLMs solving a Task and then chaining many other in sequence is good but to be efficient you have to create conditionnal workflows. In particular when using local LLMs that don't have the power to solve all Tasks with only one prompt. You have to create an AI worflow in advance that wil go foward step by step and converge to some expected result. AI allows you to deal with some level of unknown but expecting that you can have a master brain (like in crewAI) that distributes Tasks to agents and achieve an expected result is IMPOSSIBLE with local LLMs. They are too dumb ! Therefor they need you to help them allong their path. This is why LangGraph works well with local LLMs and Yacana does to. You create a workflow and when conditions are met you switch from one branch to another that treats more specific cases, etc.

---

The most common routing mechanic is "yes" / "no". Depending on the result your program can do different things next. Let's see an example:  

```

```

###

## V. Managing Agents history

## VI. Assigning a tool to a Task

## VII. Assigning multiple Tools

## VIII. Chat between two Agents

## IX. Chat between many Agents
