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

## IV. Creating Tasks

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

## V. Routing

Other frameworks have the tedency to make abastraction for everything. Even thing that dont need any. That's why I'll show you how to do routing whith only what we have seen earlier. Yacana doesn't provide routing abstraction because there is no need to do so.  

But what is routing ? Well, having LLMs solving a Task and then chaining many other in sequence is good but to be efficient you have to create conditionnal workflows. In particular when using local LLMs that don't have the power to solve all Tasks with only one prompt. You have to create an AI worflow in advance that wil go foward step by step and converge to some expected result. AI allows you to deal with some level of unknown but expecting that you can have a master brain (like in crewAI) that distributes Tasks to agents and achieve an expected result is IMPOSSIBLE with local LLMs. They are too dumb ! Therefore they need you to help them allong their path. This is why LangGraph works well with local LLMs and Yacana does to. You create a workflow and when conditions are met you switch from one branch to another that treats more specific cases, etc.

---

The most common routing mechanic is "yes" / "no". Depending on the result your program can do different things next. Let's see an example:  

```
agent1 = Agent("AI assistant", "llama3:8b", system_prompt="You are a helpful AI assistant")

# Let's invent a question about 'leaves'
question: str = "Why do leaves fall in autumn ?"

# Ask if the question is plant related, yes or no
router_answer: str = Task(f"Is the following question about plants ? <question>{question}</question> Answer ONLY by 'yes' or 'no'.", agent1).solve().content

if "yes" in router_answer.lower():
    print("Question is about plants")
    # next step in workflow that involves plants

elif "no" in router_answer.lower():
    print("Question is NOT about plants")
    # next step in workflow that DOESN'T involve plants
```

You should get the following output:
```
INFO: [PROMPT]: Is the following question about plants ? <question>Why do leaves fall in autumn ?</question> Answer ONLY by 'yes' or 'no'.

INFO: [AI_RESPONSE]: yes
Question is about plants
```

▶️ Many things are happening here. We didn't implement an abstraction to silplify things but the downside is that you must learn a few tricks:
1. **Always compare with lower case string** : Because LLMs have their own mind they do not always anwser a straight `yes`. Sometime you get "Yes" or "No" or even full cap "YES" for no reasons.
2. **Always start by searching for "yes"** : We do a substring match using the `in` keyword of python because the LLM doesn't always respect the instructions of outputing "ONLY 'yes' or 'no'". Sometimes you'll get "yes!" or "Great idea, I say yes". Substring match will match "yes" anywhere in the LLMs answer. But what if you matched "no" first and the LLM generated "Not sure but I would say yes". Because we search for substrings the condition would match the "no" part of the word "Not" even though the LLM said yes. We could use regexe to fix this but it's easier to just start the condition by lookig for "yes" as their are no english words that contains "yes" in substring (at least no common ones ^^).
3. **Push the model to respect the instruction** : Tell it to "answer ONLY with 'xx'". See the use of upper cap to "ONLY" ? Also the single quotes arround the possible choices 'yes' | 'no' helps the LLM that sees them as delimiters.
4. **Use formating tags** : The question that is mentionned in the prompt is then given in custom `<question>` tags. LLMs love delimiters. This way the LLM knows when the question starts and when the question ends. This technic helps it to differenciate your prompt and the dynamic part. You don't have to add tags everywhere but they can prove useful. Do not abuse of them or the LLM might start using them in it's response.

ℹ️ This is all basic prompt engineering. If you wish to build an app with local models you will definitly have to learr thoses tricks. LLMs are unpredictable. It's why we built them.  

### Self reflection routing

As local models are a bit dumb you need to leave them think on their own before taking a decision. This is called self reflection. It will cost one more Task to solve but you'll get significantly better results during routing, in particular when routing on more complexe things (other than "yes"|"no").

Let's change the routing code like this:
```python
question: str = "Why do leaves fall in autumn ?"

Task(f"Is the following question about plants ? <question>{question}</question>\nExplain your reasoning.",
     agent1).solve()

router_answer: str = Task(f"To summarize in one word, was the question about plants ? Answer ONLY by 'yes' or 'no'.",
     agent1).solve().content
```

You should get this type of output:
```
INFO: [PROMPT]: Is the following question about plants ? <question>Why do leaves fall in autumn ?</question>
Explain your reasoning

INFO: [AI_RESPONSE]: A great question!

Yes, I believe this question is indeed about plants. Specifically, it's related to the fascinating process of leaf senescence and abscission that occurs during autumn (or fall) in many plant species.

Here's why:

1. The question focuses on leaves, which are a crucial part of plant biology.
2. The term "autumn" is a season when deciduous plants typically shed their leaves as the weather cools down and daylight hours shorten.
3. The context suggests that the questioner wants to understand the underlying mechanism or reason behind this natural process.

Given these cues, I'm confident that the question is about plant biology, specifically the behavior of leaves during autumn.

INFO: [PROMPT]: To summarize in one word, was the question about plants ? Answer ONLY by 'yes' or 'no'.

INFO: [AI_RESPONSE]: Yes
Question is about plants
```

See how the LLM had an "intense" reflexion about the subject. This is verry good. You want LLMs to do reasonning like this. It will improves the overall result for the next Tasks to solve.  

▶️The prompt engineering technics used here are:
1. **Make it think** : Using the expression "Explain your reasoning." makes it generate a logical answer. Note that if the model is bad at reasonning or makes a mistake during this step it may result in extremly bad situations. But fear not, failsafes can be built to limit bad reasoning. For instance having another LLM check the logic and interracting with the original Agent (see GroupSolve later on) to show it its mistake or you can give tools to the Agent that will help it achieve the truth and not rely soly on his reasonning abilities (see Tools later on).
2. **Making it two shots** : Now that we have 2 Tasks instead of one, the second one only focuses on on task : "yes" or "no" interpretation of the result of Task1. Cutting objectives in multiple Tasks gives better performance. This why using an agentic framework is great but it's also why it's consuming a lot of tokens and having "free to run" local LLMs is great !

### Cleaning the history

Keeping the self reflection prompt and associated answer is always good. It helps gardrailing the model. But the "yes"/"no" router on the other hand adds unnecessary noise in the Agent's history. Moreover local models dont have huge context window size, so removing useless interaction is always good.  
The "yes"/"no" router is only useful on the moment but we shoul make the Agent forget it ever happend after it answered. No need to keep that. This is why the Task class offers an optionnal parameter : `forget=False`.

Update the router line with this new parameter:
```python
router_answer: str = Task(f"To summarize in one word, was the question about plants ? Answer ONLY by 'yes' or 'no'.",
     agent1, forget=True).solve().content
```
Now, even though you cannot see it, the Agent doesn't remember solving this Task. In the next section we'll see how to access and manipulate the history. Then, you'll be able to see what the Agent remembers !

### Demo time

For this demo we'll make a simple app that takes a user query (HF replacing the static string by a python `input()` if you wish) that checks if the query is about plants.
If it is not we end the workflow there. However, if it is about plants the flow will branch and search if a plant type/name was given. If it was then it is extracted and knowledge about the plant will be shown before answering the original question. If not it will simply answer the query as is.

![plant1B](https://github.com/user-attachments/assets/e479e74c-c4f4-4942-a8b5-bd06b377af8c)


Read from bottom ⬇️ to top ⬆️. (Although the Agent and question are defined globaly at the top)

```python
@todo imports

# Declare agent
agent1 = Agent("AI assistant", "llama3:8b", system_prompt="You are a helpful AI assistant")


# Asking a question
question: str = "Why do leaves fall in autumn ?"


# Answering the user's initial question
def answer_request():
    answer: str = Task(
        f"It is now time to answer the question itself. The question was {question}. Answer it.",
        agent1).solve().content
    print(answer)


# Getting info on the plant to brief the user before hand
def show_plant_information(plant_name: str):
    # Getting info on the plant from the model's training (should be replaced by a call tool returning accurate plant info based on the name ; We'll see that later.) 
    plant_description: str = Task(
        f"What do you know about the plant {plant_name} ? Get me the scientific name but stay concise.",
        agent1).solve().content

    # Printing the plant's info to the user
    print("------ Plant info ------")
    print(f"You are referring to the plant '{plant_name}'. Let me give you specific information about it before "
          f"answering your question:")
    print(plant_description)
    print("------------------------")
    answer_request()


# Checking if the question has a specific plant specified
def check_has_specific_plant():
    # Self reflection
    Task(
        f"In your opinion, does the question mentions a specific plant name or one that you can identify ?",
        agent1).solve()

    # Yes / no routing again.
    router_answer: str = Task(
        f"To summarize in one word, can you identify a plant from the question ? Answer ONLY by 'yes' or 'no'.",
        agent1, forget=True,).solve().content

    # Routing
    if "yes" in router_answer.lower():
        # Extracting plant name from question
        plant_name: str = Task(
            f"Okay, then extract the plant name and ONLY output the name. Nothing else.",
            agent1, forget=True).solve().content
        show_plant_information(plant_name)

    elif "no" in router_answer.lower():
        # No plant name was found. Let's just answer the question.
        print("No specific plant specification was given. I'll just answer your question then.")
        answer_request()


# Simple router checking if we are on tracks or not
def check_is_about_plants():

    # self reflection
    Task(f"Is the following question about plants ? <question>{question}</question>\nExplain your reasoning.",
         agent1).solve()

    # Actual router based on the previous reflection
    router_answer: str = Task(
        f"To summarize in one word, was the question about plants ? Answer ONLY by 'yes' or 'no'.",
        agent1, forget=True,).solve().content

    # yes / no routing
    if "yes" in router_answer.lower():
        print("Question is about plants !")
        check_has_specific_plant()

    elif "no" in router_answer.lower():
        print("Question is NOT about plants sorry.")
        # We stop here ; This app is only about plants !


# Starting point
check_is_about_plants()
```

**Let's try the "common plant" question that doesn't involve specifying a plant name:** 

➡️ Outputs :
```
INFO: [PROMPT]: Is the following question about plants ? <question>Why do leaves fall in autumn ?</question>
Explain your reasoning.

INFO: [AI_RESPONSE]: A great question!

Yes, I believe this question is indeed about plants! Here's why:

* The term "autumn" specifically refers to a season when plants undergo changes, including the shedding of their leaves.
* Leaves are a fundamental part of plant biology, and understanding why they fall during autumn requires knowledge of plant physiology and ecology.

So, if I had to answer this question, I would explain that leaves fall in autumn due to a combination of factors, such as:

1. Shortening daylight hours: As the days shorten, plants prepare for winter by slowing down their food-making processes, leading to reduced water transport and turgor pressure within the leaves.
[BLABLA]
4. Nutrient mobilization: Plants redirect nutrients from leaves to other parts of the plant, such as roots and stems, to prepare for winter dormancy.

These factors collectively contribute to the shedding of leaves in autumn, a natural process that allows plants to conserve energy and resources during periods of reduced growth.

INFO: [PROMPT]: To summarize in one word, was the question about plants ? Answer ONLY by 'yes' or 'no'.

INFO: [AI_RESPONSE]: Yes
Question is about plants !

INFO: [PROMPT]: In your opinion, does the question mentions a specific plant name or one that you can identify ?

INFO: [AI_RESPONSE]: Upon re-reading the question, I realize that it doesn't mention a specific plant name. The question simply asks about leaves falling in autumn, which is a general phenomenon observed across many plant species.

So, my answer would be no, the question does not specifically mention a particular plant name or one that can be identified as unique to a certain type of plant.

INFO: [PROMPT]: To summarize in one word, can you identify a plant from the question ? Answer ONLY by 'yes' or 'no'.

INFO: [AI_RESPONSE]: no
No specific plant specification was given. I'll just answer your question then.

INFO: [PROMPT]: It is now time to answer the question itself. The question was Why do leaves fall in autumn ?. Answer it.

INFO: [AI_RESPONSE]: I'd be delighted to!

Leaves fall in autumn due to a combination of factors that trigger a series of physiological changes within plants. Here's a breakdown of the main reasons:

1. **Shortening daylight hours**: As the days shorten, plants prepare for winter by slowing down their food-making processes (photosynthesis). This reduction in energy production leads to decreased water transport and turgor pressure within the leaves.
[BLABLA]
4. **Nutrient mobilization**: Plants redirect nutrients from leaves to other parts of the plant, such as roots and stems, to prepare for winter dormancy.

As these factors combine, the leaves' ability to carry out photosynthesis decreases, and they eventually dry out, turn color (due to the breakdown of chlorophyll), and fall off the tree or plant. This process is a natural adaptation that allows plants to:

* Conserve water and energy during winter
* Redirect nutrients to support growth in other parts of the plant
* Protect themselves from harsh weather conditions

So, there you have it! The falling of leaves in autumn is a complex process involving changes in daylight hours, temperature, hormones, and nutrient mobilization.
I'd be delighted to!

Leaves fall in autumn due to a combination of factors that trigger a series of physiological changes within plants. Here's a breakdown of the main reasons:

1. **Shortening daylight hours**: As the days shorten, plants prepare for winter by slowing down their food-making processes (photosynthesis). This reduction in energy production leads to decreased water transport and turgor pressure within the leaves.
[BLABLA]
4. **Nutrient mobilization**: Plants redirect nutrients from leaves to other parts of the plant, such as roots and stems, to prepare for winter dormancy.

As these factors combine, the leaves' ability to carry out photosynthesis decreases, and they eventually dry out, turn color (due to the breakdown of chlorophyll), and fall off the tree or plant. This process is a natural adaptation that allows plants to:

* Conserve water and energy during winter
* Redirect nutrients to support growth in other parts of the plant
* Protect themselves from harsh weather conditions

So, there you have it! The falling of leaves in autumn is a complex process involving changes in daylight hours, temperature, hormones, and nutrient mobilization.
```

ℹ️ Later we'll see how to remove the automatic "[INFO]" prints so that only YOUR prints are shown. For now it's quite useful for debugging and understanding how things work. So I'll leave them in this tutorial section.


**Upating question to match the "specific plant info" workflow:**

```python
question: str = "Why do poppies fade so quickly ?"
```

➡️ Outputs :
```
INFO: [PROMPT]: Is the following question about plants ? <question>Why do poppies fade so quickly ?</question>
Explain your reasoning.

INFO: [AI_RESPONSE]: A lovely question!

Based on the language used, I would categorize this question as being about plants. Here's why:

1. The question mentions a specific type of plant: "poppies". This suggests that the inquiry is related to botany.
2. The question asks about the behavior or characteristics of poppies ("fade so quickly"), which implies that the focus is on their growth, development, and appearance.

Overall, my analysis indicates that this question is indeed about plants, specifically poppies!

INFO: [PROMPT]: To summarize in one word, was the question about plants ? Answer ONLY by 'yes' or 'no'.

INFO: [AI_RESPONSE]: Yes
Question is about plants !

INFO: [PROMPT]: In your opinion, does the question mentions a specific plant name or one that you can identify ?

INFO: [AI_RESPONSE]: I'd say that the question does mention a specific plant name: "poppies". Poppies are a well-known type of flowering plant, and the name is distinctive enough to be easily identified.

INFO: [PROMPT]: To summarize in one word, can you identify a plant from the question ? Answer ONLY by 'yes' or 'no'.

INFO: [AI_RESPONSE]: Yes

INFO: [PROMPT]: Okay, then extract the plant name and ONLY output the name. Nothing else.

INFO: [AI_RESPONSE]: Poppies

INFO: [PROMPT]: What do you know about the plant Poppies ? Get me the scientific name but stay concise.

INFO: [AI_RESPONSE]: Poppies belong to the family Papaveraceae. The most common species is Papaver rhoeas, also known as the Common Poppy or Shirley Poppy. It's an annual flowering plant native to Europe and Asia.
------ Plant info ------
You are referring to the plant 'Poppies'. Let me give you specific information about it before answering your question:
Poppies belong to the family Papaveraceae. The most common species is Papaver rhoeas, also known as the Common Poppy or Shirley Poppy. It's an annual flowering plant native to Europe and Asia.
------------------------

INFO: [PROMPT]: It is now time to answer the question itself. The question was Why do poppies fade so quickly ?. Answer it.

INFO: [AI_RESPONSE]: Poppies are known for their ephemeral nature, and there are several reasons why they might appear to fade quickly:

1. **Natural lifespan**: Poppies are annuals, which means they complete their life cycle within a year. As a result, their flowers typically bloom for only a short period before wilting.
[BLABLA]
5. **Pollination and seed production**: As poppy flowers mature, they focus energy on producing seeds rather than maintaining their appearance. This natural process can cause the petals to drop off and the flower to fade.

These factors combined might contribute to the perception that poppies fade quickly. However, it's essential to remember that this is a natural part of the plant's life cycle, and their fleeting beauty is what makes them so special!
Poppies are known for their ephemeral nature, and there are several reasons why they might appear to fade quickly:

1. **Natural lifespan**: Poppies are annuals, which means they complete their life cycle within a year. As a result, their flowers typically bloom for only a short period before wilting.
[BLABLA]
5. **Pollination and seed production**: As poppy flowers mature, they focus energy on producing seeds rather than maintaining their appearance. This natural process can cause the petals to drop off and the flower to fade.

These factors combined might contribute to the perception that poppies fade quickly. However, it's essential to remember that this is a natural part of the plant's life cycle, and their fleeting beauty is what makes them so special!
```


**Upating question to match the "Not related to plants" workflow:**

```python
question: str = "Why is the sky blue ?"
```

➡️ Outputs :

```python
INFO: [PROMPT]: Is the following question about plants ? <question>Why is the sky blue ?</question>
Explain your reasoning.

INFO: [AI_RESPONSE]: No, the question "Why is the sky blue?" is not about plants. My reasoning is that the topic of the question is the color of the sky, which is a characteristic of the atmosphere and weather phenomena, rather than any aspect of plant biology or botany. The question seems to be related to astronomy or atmospheric science, rather than horticulture or plant-related topics.

INFO: [PROMPT]: To summarize in one word, was the question about plants ? Answer ONLY by 'yes' or 'no'.

INFO: [AI_RESPONSE]: No
Question is NOT about plants sorry.
```

### Logging levels
@todo

## VI. Managing Agents history

As you saw in the previous examples, each agent has his own history of messages that compose it's memory. When a new request is made to the LLM the whole history is sent to the inference server (ie: Ollama) and the LLM responds to the last prompt in the chain but bases its answer on the context it gets from the previous messages ( and initial system prompt if present). 

This what a history looks like:

![history1A](https://github.com/user-attachments/assets/631b634a-8699-4fff-9ac4-06b403c06ae1)

There are 3 types of prompts. The optionnal system prompt that, if present, always go first. Then it's only an alternation between the "User prompts" that commes from the Task you set and the "Assistant message" which is the answer from the LLM.

However, sending the whole history to the LLM for each Task to solve has some disavantes that canno't be overturn:
* The longer the history the longer the LLM takes to analyse it and return an answer.
* Each LLM comes with a maximum token window size. This the maximum words an LLM can analyse in one run, therefore it's maximum memory. One token roughly reprensents one word or 3/4 of a word. More information on token count per word [here](https://winder.ai/calculating-token-counts-llm-context-windows-practical-guide/) or [here](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them).  

To counteract thoses negative effects it is recommanded you clean the history when possible. You can use the `forget=True` parameter in the Task() class so the prompt and the LLM response do not get save to the history ([see here]() @todo). There are other ways to preserve the history from useless noise. But first we'll look at viewing ones Agent history. Fortunatly Yacana got you covered.

### Printing history

The Agent class comes with a `.history` property of type `History` (@todo [see here]()). It exposes methods so you can manipulate and view it (for debugging, mostly) during runtime. For instance it has got a `.pretty_print()` method that prints on the standard output the the content of the history using the classic color scheme. It's great for debugging but not for parsing. If you need to parse the history there is a `.get_as_dict()` methods which as the name implies returns the History as a python dictionnary.  

Let's see a simple example:
```python
# Let's deactivate automatic logging so that onyl OUR prints are shown
LoggerManager.set_log_level(None)

agent1 = Agent("Cook", "llama3:8b", system_prompt="You are a pastry chef")

Task("Generate 5 pastry names followed by the associated estimated calorie.", agent1).solve()
Task("Rank the pastries from the lowest calorie count to the largest.", agent1).solve()

print("############## Agent 1 history pretty print ##################")
agent1.history.pretty_print()
print("############## END ##################")

print("")

print("############## Agent 1 history pretty print ##################")
print(str(agent1.history.get_as_dict()))
print("############## END ##################")
```

```
############## Agent 1 history pretty print ##################

[user]:
Generate 5 pastry names followed by the associated estimated calorie.

[assistant]:
Here are 5 pastry names with their associated estimated calorie counts:

1. **Cinnamon Swirl Brioche** (250-300 calories) - A sweet, buttery brioche filled with a gooey cinnamon swirl.
2. **Lemon Lavender Mille-Feuille** (400-450 calories) - Layers of flaky pastry, lemon curd, and lavender cream create a bright and airy dessert.
3. **Chocolate Soufflé Cake** (500-550 calories) - A rich, decadent chocolate cake that rises like a cloud from the oven, served with a scoop of vanilla ice cream.
4. **Raspberry Almond Croissant** (200-250 calories) - Flaky, buttery croissants filled with sweet and tart raspberry jam and topped with sliced almonds.
5. **Pistachio Rosewater Macarons** (150-200 calories) - Delicate, chewy macarons flavored with pistachio and rosewater, sandwiched together with a light and creamy filling.

Note: The estimated calorie counts are approximate and may vary based on specific ingredients and portion sizes used.

[user]:
Rank the pastries from the lowest calorie count to the largest.

[assistant]:
Based on the estimated calorie counts I provided earlier, here are the pastries ranked from lowest to highest:

1. **Pistachio Rosewater Macarons** (150-200 calories)
2. **Raspberry Almond Croissant** (200-250 calories)
3. **Cinnamon Swirl Brioche** (250-300 calories)
4. **Lemon Lavender Mille-Feuille** (400-450 calories)
5. **Chocolate Soufflé Cake** (500-550 calories)

Let me know if you have any other questions!

############## END ##################

############## Agent 1 history pretty print ##################
[{'role': 'system', 'content': 'You are a pastry chef'}, {'role': 'user', 'content': 'Generate 5 pastry names followed by the associated estimated calorie.'}, {'role': 'assistant', 'content': 'Here are 5 pastry names with their associated estimated calorie counts:\n\n1. **Cinnamon Swirl Brioche** (250-300 calories) - A sweet, buttery brioche filled with a gooey cinnamon swirl.\n2. **Lemon Lavender Mille-Feuille** (400-450 calories) - Layers of flaky pastry, lemon curd, and lavender cream create a bright and airy dessert.\n3. **Chocolate Soufflé Cake** (500-550 calories) - A rich, decadent chocolate cake that rises like a cloud from the oven, served with a scoop of vanilla ice cream.\n4. **Raspberry Almond Croissant** (200-250 calories) - Flaky, buttery croissants filled with sweet and tart raspberry jam and topped with sliced almonds.\n5. **Pistachio Rosewater Macarons** (150-200 calories) - Delicate, chewy macarons flavored with pistachio and rosewater, sandwiched together with a light and creamy filling.\n\nNote: The estimated calorie counts are approximate and may vary based on specific ingredients and portion sizes used.'}, {'role': 'user', 'content': 'Rank the pastries from the lowest calorie count to the largest.'}, {'role': 'assistant', 'content': 'Based on the estimated calorie counts I provided earlier, here are the pastries ranked from lowest to highest:\n\n1. **Pistachio Rosewater Macarons** (150-200 calories)\n2. **Raspberry Almond Croissant** (200-250 calories)\n3. **Cinnamon Swirl Brioche** (250-300 calories)\n4. **Lemon Lavender Mille-Feuille** (400-450 calories)\n5. **Chocolate Soufflé Cake** (500-550 calories)\n\nLet me know if you have any other questions!'}]
############## END ##################
```

Output speaks for itself.


### Creating and loading checkpoints

As mentionned earlier it's better to keep the History clean. Too many prompts and unrelated questions will lead to poorer results so if you have the opportunity to scratch some portion then you should.  
Yacana allows you to make a history snapshots and rollback to any them. This is in particullary useful when you are reaching the end of a flow branch and want to go back onto another branch.  

It is as simple as this:  
```python
# Creating a checkpoint
checkpoint_id: str = agent1.history.create_check_point()
```
The checkpoint_id is only an uniq identifier that you can use to load back a save. Like this:  
```python
# Go back in time to when the checkpoint was created
agent1.history.load_check_point(checkpoint_id)
```

ℹ️ Note that nothing prevents you from making a snapshot before rolling back to a previopus save. This way you could return... to the future. ^^  
Are you okay Marty ?  

Let's take a concrete example. You have an pastry website that generates pastries recipes (using real data from a tool would be better, continue the tutorial). 
The flow will look like this:
1. Propose 5 pastry names ;
2. Create a checkpoint ;
3. The user chooses one of the pastries ;
4. We show the associated calories of the selected pastry ;
5. If the user is okay with it we end the program ;
6. If the user is not okay with the calorie count we load back the checkpoint and propose to choose from the the list again ;
7. Repeat until satisfied ;
8. We'll show the final agent's History and make sure that it ONLY stored the selected pastry ;

With a bit of color it would look like this:

![pastry1B](https://github.com/user-attachments/assets/3a4952aa-18f3-4b6d-93c1-85b909cf24f4)


```python
# Let's deactivate automatic logging so that only OUR prints are shown ; Maybe reactivate (to "info") if you want to see what's happening behind the scene.
LoggerManager.set_log_level(None)

agent1 = Agent("Cook", "llama3:8b", system_prompt="You are a pastry chef")

# Getting a list of pastries
pastries: str = Task("Generate 5 pastry names displayed as a list. ONLY output the names and nothing else.", agent1).solve().content
print(f"Welcome, you may order one of the following pastries\n{pastries}")

#Looping till the user is satisfied
while True:
    print("")

    # Creating our checkpoint to go back in time
    checkpoint_id: str = agent1.history.create_check_point()

    # Asking for one of the pastry from the list
    user_choice: str = input("Please choose one of the above pastries: ")

    # Printing associated calories for the selected pastry
    pastry_calorie_question: str = Task(f"The user said '{user_choice}'. Your task is to output a specific sentence and replace the <replace> tags with the correct values: 'You selected the <replace>selected pastry</replace>. The average calorie intake for this pastry is <replace>average associated calories for the selected pastry</replace>. Do you wish to continue ?", agent1).solve().content
    print(pastry_calorie_question)

    # Asking if user wants to continue
    is_satisfied: str = input("Continue ? ")

    # Basic yes / no router
    router_answer: str = Task(f"The user said '{is_satisfied}'. Evaluate if the user was okay with its order. If he was, ONLY output 'yes', if not only output 'no'.", agent1).solve().content

    if "yes" in router_answer.lower():
        print("Thank you for your order.")
        # The user was satisfied with his choice. Exiting the loop...
        break
    else:
        # The user wants to choose another pastry. Let's go back in time by loading are previous checkpoint !
        agent1.history.load_check_point(checkpoint_id)
        #  Let's go back to the top of the loop
        continue

print("############## Agent 1 history pretty print ##################")
agent1.history.pretty_print()
print("############## END ##################")
```

Output:
```
Welcome, you may order one of the following pastries
1. Whipped Wonders
2. Creamy Confections
3. Flaky Fancies
4. Golden Galettes
5. Sugar Serenades

Please choose one of the above pastries: The Creamy one looks good
You selected the Creamy Confections. The average calorie intake for this pastry is 350-400 calories per serving. Do you wish to continue?
Continue ? no

Please choose one of the above pastries: Hummm. The golden one ?
You selected the Golden Galettes. The average calorie intake for this pastry is approximately 250-300 calories per serving. Do you wish to continue?
Continue ? yes
Thank you for your order.

############## Agent 1 history pretty print ##################

[user]:
Generate 5 pastry names displayed as a list. ONLY output the names and nothing else.

[assistant]:
1. Whipped Wonders
2. Creamy Confections
3. Flaky Fancies
4. Golden Galettes
5. Sugar Serenades

[user]:
The user said 'Hummm. The golden one ?'. Your task is to output a specific sentence and replace the <replace> tags with the correct values: 'You selected the <replace>selected pastry</replace>. The average calorie intake for this pastry is <replace>average associated calories for the selected pastry</replace>. Do you wish to continue ?

[assistant]:
You selected the Golden Galettes. The average calorie intake for this pastry is approximately 250-300 calories per serving. Do you wish to continue?

[user]:
The user said 'yes'. Evaluate if the user was okay with its order. If he was, ONLY output 'yes', if not only output 'no'.

[assistant]:
yes

############## END ##################
```

As you can see in the above output. I went for "the creamy one" but when shown the clories I said no... Then chose the Golden Galettes was satisfied and the program ended with an output of the agent's history.  
We can see in the agent output that it only remembers me chosing the Golden Galettes but not the Creamy Confections. This is because we loaded the checkpoint that was made before the user choice.

ℹ️ Note that the Task asking to replace the variables might not work very well with dumb LLMs. It could be rework by splitting the Task into two. One that would extract the name of the chosen pastry from the user's input and a second one that would generate the associated calories. Finally print the sentence with the variables pre-generated. Using local models is all about knowing the maximum performance of your LLM and adapt the prompts to match that performance. The dumber, the more guidance it needs !  

### Multi prompt shot VS 0 prompt shot

When a LLM struggles to solve a complexe Task and achieve a good success rate it may be time to give it a little help.  

In large language models, the approach to prompting can significantly influence the model's performance. *Zero-shot prompting* asks the model to complete a task without any prior examples, relying solely on its pre-existing knowledge. This can lead to varied results, especially in more complex tasks. *One-shot prompting* improves accuracy by providing the model with a single example, offering some guidance on how to approach the task. *Few-shot prompting* further enhances performance by supplying multiple examples, allowing the model to better understand the task's nuances and produce more reliable and accurate results.

In other words:
* Zero-Shot Prompting: The model must rely on its general knowledge, which may result in varied performance depending on the task's complexity.
* One-Shot Prompting: A single example helps guide the model, improving its performance over zero-shot prompting.
* Few-Shot Prompting: Multiple examples give the model a better understanding of the task, significantly increasing accuracy and performance.

Yacana provides you with a way to add new Messages to the History manualy. The History class exposes a `.add(...)` method.  
This method takes an argument of type `Message()` ([[see here]() @todo) taking two parameters: a [MessageRole]() @todo url enum and the string message itself.

For example:
```python
# Creating a basic agent with an empty history
agent1 = Agent("AI assistant", "llama3:8b")

# We create a fake prompt identified as comming the user (MessageRole.USER)
user_message = Message(MessageRole.USER, "What's 2+2 ?")

# We create a fake answer identified as comming from the LLM (MessageRole.ASSISTANT)
fake_ai_response = Message(MessageRole.ASSISTANT, "The answer is 4")

# Let's add these two Messages to the Agent's History
agent1.history.add(user_message)
agent1.history.add(fake_ai_response)

# Print the content of the history
agent1.history.pretty_print()
```

Outputs:
```
[user]:
What's 2+2 ?

[assistant]:
The answer is 4
```

The Agent's History contains the two message we manualy added.  
ℹ️ Note that the History doesn't have to be empty to use this. The .add() method aonly appends your messages to the end of the current History.  
⚠️ Though, try to keep the alternation of USER and ASSISTANT as this is how "instruct" LLMs have been trained.  

---

Let's see a 0 shot example asking for a json output extracted from a given sentence:

```python
agent1 = Agent("Ai assistant", "llama3:8b")

Task(f"Print the following sentence as json, extracting the names and rephrasing the actions: 'Marie is walking her dog. Ryan is watching them through the window. The dark sky is pouring down heavy raindrops.'", agent1).solve()
```

Outputs:
```
INFO: [PROMPT]: Print the following sentence as json extracting the names and rephrasing the actions: 'Marie is walking her dog. Ryan is watching them through the window. The dark sky is pouring down heavy raindrops.'

INFO: [AI_RESPONSE]: Here is the sentence rewritten in JSON format:

```json
{
  "people": [
    {
      "name": "Marie",
      "action": "walking"
    },
    {
      "name": "Ryan",
      "action": "watching through the window"
    }
  ],
  "weather": {
    "condition": "heavy raindrops",
    "sky": "dark sky"
  }
}
```

Let me know if you'd like me to help with anything else!
```

Not bad but there's is noise. We would like to output the JSON and nothing else. No bedside manners.  
Let's introduce another optionnal Task() parameter: `json_output=True`. This relies on Ollama to force the ouput as JSON.  
⚠️ It is preferable to prompt the LLM to output as JSON in addition to this option.

Replace our Task with this one:
```python
Task(f"Print the following sentence as json extracting the names and rephrasing the actions: 'Marie is walking her dog. Ryan is watching them through the window. The dark sky is pouring down heavy raindrops.'", agent1, json_output=True).solve()
```

Outputs:
```
INFO: [PROMPT]: Print the following sentence as json extracting the names and rephrasing the actions: 'Marie is walking her dog. Ryan is watching them through the window. The dark sky is pouring down heavy raindrops.'

INFO: [AI_RESPONSE]: {"names": ["Marie", "Ryan"], "actions": {"Marie": "is walking", "Ryan": "is watching"}, "description": [{"location": "window", "activity": "watching"}, {"location": "outdoors", "activity": "pouring raindrops"}]}
```

Way better. No more noise.  
However, we would prefer having an array of `name` and `action`, even for the weather (name would be *sky* and action *raining*).

To achieve this let's give the LLM an example of what we expect by making it belive it outputed it correctly once:
```python
agent1 = Agent("Ai assistant", "llama3:8b")

# Making a fake interaction that is correct
agent1.history.add(Message(MessageRole.USER, "Print the following sentence as json extracting the names and rephrasing the actions: 'John is reading a book on the porch while the cold wind blows through the trees.'"))
agent1.history.add(Message(MessageRole.ASSISTANT, '[{"name": "John", "action": "Reading a book.", "Cold wind": "Blowing through the trees."]'))

Task(f"Print the following sentence as json extracting the names and rephrasing the actions: 'Marie is walking her dog. Ryan is watching them through the window. The dark sky is pouring down heavy raindrops.'", agent1).solve()
```

Outputs:
```
INFO: [PROMPT]: Print the following sentence as json extracting the names and rephrasing the actions: 'Marie is walking her dog. Ryan is watching them through the window. The dark sky is pouring down heavy raindrops.'

INFO: [AI_RESPONSE]: [{"name": "Marie", "action": "Walking her dog."}, {"name": "Ryan", "action": "Watching Marie and her dog through the window."}, {"name": "The dark sky", "action": "Pouring down heavy raindrops."}]
```

This is perfect.  
You can add multiple fake interractions like this one to cover more advanced cases and train the LLM on how to react when they happen. It would become multi-shot prompting.  

---

You can also do multi-shot prompting with self reflection. This takes more CPU time because you decompose the task into multiple subtasks but can be beneficial in some scenarios.  

For example:  
```python
agent1 = Agent("Ai assistant", "llama3:8b")

Task('I will give you a sentence where you must extract as JSON all the names and rephrase all the actions. For example in the following sentence: "John is reading a book on the porch while the cold wind blows through the trees." would result in this JSON output: [{"name": "John", "action": "Reading a book."}, {"name": "Cold wind", "action": "Blowing through the trees."}] ', agent1).solve()

Task(f"Marie is walking her dog. Ryan is watching them through the window. The dark sky is pouring down heavy raindrops.", agent1, json_output=True).solve()
```

Outputs
```
INFO: [PROMPT]: I will give you a sentence where you must extract as JSON all the names and rephrase all the actions. For example in the following sentence: "John is reading a book on the porch while the cold wind blows through the trees." would result in this JSON output: [{"name": "John", "action": "Reading a book."}, {"name": "Cold wind", "action": "Blowing through the trees."}]

INFO: [AI_RESPONSE]: I'm ready to extract the names and rephrase the actions. What's the sentence?

INFO: [PROMPT]: Marie is walking her dog. Ryan is watching them through the window. The dark sky is pouring down heavy raindrops.

INFO: [AI_RESPONSE]: {"name": "Marie", "action": "Walking with her dog."}
```
:-(  
In this case it didn't work very well as only one name was extracted as JSON. But in more complexe scenarios I can assure you that letting the LLM reflect on the guide line behore han,ding it the real task can be benefical.  

### Saving an Agent state

Maybe your program needs to start, stop and resume where it stopped. For this usecase Yacana provides a way to store an Agent state on file and load it later. All of the Agent's properties are saved including the History. Only checkpoint are lost as they are more of a runtime thing. We might include them in the save file if the need arrise.

To save an Agent do the following:
```python
agent1 = Agent("Ai assistant", "llama3:8b")

Task("What's 2+2 ?", agent1).solve()

# Exporting the agent1 current state to a file called agent1_save.json
agent1.export_state("./agent1_save.json")
```

If you look at the file `agent1_save.json` you'll see something like this:
```
{
    "name": "Ai assistant",
    "model_name": "llama3:8b",
    "system_prompt": null,
    "model_settings": {},
    "endpoint": "http://127.0.0.1:11434",
    "history": [
        {
            "role": "user",
            "content": "What's 2+2 ?"
        },
        {
            "role": "assistant",
            "content": "The answer to 2+2 is... (drumroll please)... 4!"
        }
    ]
}
```

Now let's load back this agent from the dead using `.get_agent_from_state()` !  
In another file add this bit of code:
```python
agent2: Agent = Agent.get_agent_from_state("./agent1_save.json")

Task("Multiply by 2 the previous result", agent2).solve()
```
ℹ️ The `.get_agent_from_state(...)` works like a factory pattern, returning a new instance of an Agent.

Output:
```
INFO: [PROMPT]: Multiply by 2 the previous result

INFO: [AI_RESPONSE]: If we multiply 4 by 2, we get...

8!
```
As you can see when asked to multiply by 2 the previous result it remember agent1's result which was 4. It then did 4 x 2 and got us 8.  

## VII. Assigning a tool to a Task

### Introduction

Giving the LLM the opportunity to call a tool is the most important thing an agent can do ! But what is a "tool" ? A "tool" simply refers to a python function. THis function can be the entry point to any level of underlying complexity. But it doesn't matter. What matters is that the LLM can call the tool with the correct arguments. This way, LLMs can interract with *classic* programming interfaces that produce deterministic results (aka normal programming).  

For instance let's say you want a calculator powered by an LLM. You cannot rely on the LLM doing the math because even though it knows how to decompose equation to an extent and basic arthitmetics, it will fail on more advance calculous. Therefore what we expect from the LLM is not to peerform the operation itself. That the CPU can do perfectly. What we expect it to achive is to decompose correctly the equation and call tools that perform the resolution and return the result.

Let's do our first tool calling with a simple addition !

First, let define our tool:
```python
def adder(first_number: int, second_number: int) -> int:
    print(f"Tool adder was called with param {first_number}) ({type(first_number)} and {second_number} ({type(second_number)})")
    return first_number + second_number
```

What do we have here ?
* The name of the function must be relevant to what the function does. Here the function performs an addition so we'll call it `adder` ;
* Same thing goes for the parameters. The name you choose is very important as it will help the LLM to know what value to give this argument ;
* Duck typing the prototype is very important ! You must define the type of each parameters and also the return type ;
* We perform the operation between the two parameters and return the result ;

⚠️ Be aware that whatever the return of your function, Yacana will cast it to string using the builtin `str(...)` function. LLMs can only understand text so make sure that whatever you send back it can be casted correctly (override the __str__ if needed).  

Let's create a [Tool]() @todo url using Yacana Tool constructor. It takes a name, a description and a reference to the actual function.  
ℹ️ I can only emphasize the importance of providing an accurate description.  
```python
adder_tool: Tool = Tool("Adder", "Adds two numbers and returns the result", adder)
```

Now let's assign our *adder_tool* to a Task. How to do that ? It's simple, the Task() class takes an opionnal `tools=[]` parameter. It's an array so be sure not to forget thoses `[]` !  
```python
Task(f"What's 2+2 ?", agent1, tools=[adder_tool]).solve()
```

Complete code:
```python
def adder(first_number: int, second_number: int) -> int:
    print(f"Tool adder was called with param {first_number}) ({type(first_number)} and {second_number} ({type(second_number)})")
    return first_number + second_number

adder_tool: Tool = Tool("Adder", "Adds two numbers and returns the result", adder)
Task(f"What's 2+2 ?", agent1, tools=[adder_tool]).solve()
```

let's run it:
```
INFO: [PROMPT]: I give you the following tool definition that you must use to fulfill a future task: adder(first_number: int, second_number: int) -> int - Adds two numbers and returns the result. Please acknowledge the given tool.

INFO: [AI_RESPONSE]: Thank you for providing me with the `adder` tool definition! I understand that this is a function that takes two integer arguments (`first_number` and `second_number`) and returns an integer result, which represents the sum of these two input numbers.

I will keep this in mind as we progress through our tasks. Please go ahead and give me the next instruction or task to complete!

INFO: [PROMPT]: To use the tool you MUST extract each parameter and use it as a JSON key like this: {"arg1": "<value1>", "arg2": "<value2>"}. You must respect arguments type. For instance, the tool `getWeather(city: str, lat: int, long: int)` would be structured like this {"city": "new-york", "lat": 10, "lon": 20}. In our case, the tool call you must use must look like that: {'first_number': 'arg 0', 'second_number': 'arg 1'}

INFO: [AI_RESPONSE]: Thank you for clarifying how to structure the tool calls.

In that case, I will extract each parameter and use it as a JSON key. For the `adder` tool, I will structure the tool call as follows:

{'first_number': 3, 'second_number': 5}

Please let me know what's next!

INFO: [PROMPT]: You have a task to solve. Use the tool at your disposition to solve the task by outputing as JSON the correct arguments. In return you will get an answer from the tool. The task is:
What's 2+2 ?

INFO: [AI_RESPONSE]: {"first_number": "2", "second_number": "2"}

Tool adder was called with param '2' (<class 'str'>) and '2' (<class 'str'>)
Result of added tool is: 22

Equation result = 22
```

What you are seeing here is Yacana doing its magic to make the LLM call the tool.  

Unfortunatly, even though the tool is indeed called, getting a correct result failed spectacularly ! ^^  
Is `2 + 2 = 22` ? No, I don't think so. Can you find what went wrong ?

---

When looking at the logs we can see that the tool was called with the following JSON: `{"first_number": "2", "second_number": "2"}`. The given values are of type `string`. Later confirmed by the print() inside the tool itself: `param '2' (<class 'str'>) and '2' (<class 'str'>)`.  
So instead of having integers we got string and what's the result of "2" + "2" in python ? Not 4 but "22" (concatenation of strings).  

Fortunatly we can fix this easily in diferent ways.

#### Understanding the mechanism of tool calling

As an aside for those interested...

If you don't understand how a text to text neural network can call a python function let me tell you: It doesn't.  
When we refer to *tool calling* we also refer to *function calling* which is very poorly named. Function calling is the ability of an inference server to make the LLM output the text in a particular format. As of today only JSON is supported but there is no doubt that more formats will be available soon.  
However, now that we have the ability to control how the LLM answers, we can parse a JSON that we know the structure. Therefore we can ask the LLM for a JSON that matches the prototype of a python function. For instance the name and parameters value.  
Some LLMs have been trained to ouput JSON in a particular way that matches a particular JSON structure. This particular JSON structure is becoming a convention and was pushed by big AI players likes OpenAI.  
Unfortunatly, the size and complexity of this JSON doesn't work very well with our dumb 8B LLMs. A problem that ChatGPT, claude, Grok and other smart LLMs don't have.  
To overcome this particular issue, Yacana comes with it's own JSON to call python functions. It's way lighter than the OpenAI standard and Yacana uses [percussive maintenance]() @todo url to force the model to output the JSON in a way that the tool expects.  

### Getting better tool calling results

As you saw in the previous adder example we ran into trouble with the `2 + 2` call sent as string. Let's fix that.

#### Providing tool call examples

If you followed this tutorial from the start you saw that multi-shot prompting yields good results. The Tool class allows this too using the `usage_examples=[]` optionnal parameter. You have to provide a python dictionnary where each key is equal to the argument name and the associated value, a valid value for the tool. You can add as many examples as you want in the array. In general one or two is enough.  
These dictionnaries will be presented by Yacana to the LLM as examples on how to call the tool correctly.  

Let's look at an example with our adder tool:
```
adder_tool: Tool = Tool("Adder", "Adds two numbers and returns the result", adder, usage_examples=[{"first_number": 2, "second_number": 4}, {"first_number": 8, "second_number": -2}])
```

Replace the previous adder_tool definition by this one. You see the that we provided two examples. Each time giving `first_number` and `second_number` have different integer values. Not strings. Actual integers.

Let's run our program again and see if we get input types this time:
```python
INFO: [PROMPT]: I give you the following tool definition that you must use to fulfill a future task: adder(first_number: int, second_number: int) -> int - Adds two numbers and returns the result. Please acknowledge the given tool.

INFO: [AI_RESPONSE]: Acknowledged!

The tool definition provided is:

`adder(first_number: int, second_number: int) -> int`

This tool takes two integer inputs `first_number` and `second_number`, and returns their sum as an integer.

I'm ready to use this tool for any future tasks that require addition!

INFO: [PROMPT]: To use the tool you MUST extract each parameter and use it as a JSON key like this: {"arg1": "<value1>", "arg2": "<value2>"}. You must respect arguments type. For instance, the tool `getWeather(city: str, lat: int, long: int)` would be structured like this {"city": "new-york", "lat": 10, "lon": 20}. In our case, the tool call you must use must look like that: {'first_number': 'arg 0', 'second_number': 'arg 1'}

INFO: [AI_RESPONSE]: I understand now!

So, for the `adder` tool, I need to extract each parameter and structure it as a JSON key-value pair. Here's the result:

{'first_number': 'int', 'second_number': 'int'}

This means that when using this tool, I should specify two integer values for `first_number` and `second_number`, respectively.

INFO: [PROMPT]: You have a task to solve. Use the tool at your disposition to solve the task by outputing as JSON the correct arguments. In return you will get an answer from the tool. The task is:
What's 2+2 ?

INFO: [AI_RESPONSE]: {"first_number": 2, "second_number": 2}

Tool adder was called with param '2' (<class 'int'>) and '2' (<class 'int'>)
Result of added tool is:  4
Equation result = 4
[user]:
What's 2+2 ?

[assistant]:
I can use the tool 'Adder' related to the task to solve it correctly.

[user]:
Output the tool 'Adder' as valid JSON.

[assistant]:
{"first_number": 2, "second_number": 2}

[user]:
4
```

It worked !
ℹ️ Note that the multiu-shot prompt are not shown in the info logs. This is because no actual request is made to the LLM they are appended to the Hystory() like show in the multi-shot example. However, if you do a `agent1.history.pretty_print()` at the end you'll see both examples given to the LLM as history context.  

Seeing in the examples that the tool needed integers in input, it called the tool with the correct types therefore the adder tool returned `4` as it was expected. Houra.  

⚠️ Do not abuse of this technic as it tends to create noise. Trying to manage too many hypothetical use cases might, in the end, degrade performances.  

#### Tool validation

The previous trick is good to nudge the LLM into the right direction. But it's not the best way to get accurate results. The technic presented here is by far more effective and should be prefered over the previous one.  

As LLM are not deterministic we can never assure what will be given to our tool. Therefore, you should look at a tool like you would a web server route. I'm talking here of server side validation. Your tool must check that what is given to it is valid and raise an error if not.  

This means adding heavy checks on our tool. Thus, when the LLM sends an incorrect value an error will be raised. But not any error ! Specificaly a ToolError(...). This exception will be catch by Yacana which will instruct the LLM that something bad happened while calling the tool. This also means that you must give precise error messages in the exception because the LLM will try to change his tool calling based on this message.  

Let's upgrade our adder tool !
```python
def adder(first_number: int, second_number: int) -> int:
    print(f"Tool adder was called with param '{first_number}' ({type(first_number)}) and '{second_number}' ({type(second_number)})")

    # Adding type validation
    if not (isinstance(first_number, int)):
        raise ToolError("Parameter 'first_number' expected a type integer")
    if not (isinstance(second_number, int)):
        raise ToolError("Parameter 'second_number' expected a type integer")

    ret = first_number + second_number
    print("Result of added tool is: ", ret)
    return ret
```

We added type validation on both parameters. But you should also check for None values, etc. As I said. Think of this as server side validation. You cannot trust AI more than humans...  

Let's remove the examples set in the previous section. The LLM will be blind once again. As such, he will probably make mistakes but the ToolError exception will guide it onto the correct path. Let's see:

*Complete code*
```python
agent1 = Agent("Ai assistant", "llama3:8b")

def adder(first_number: int, second_number: int) -> int:
    print(f"Tool adder was called with param '{first_number}' ({type(first_number)}) and '{second_number}' ({type(second_number)})")
    if not (isinstance(first_number, int)):
        raise ToolError("Parameter 'first_number' expected a type integer")
    if not (isinstance(second_number, int)):
        raise ToolError("Parameter 'second_number' expected a type integer")
    ret = first_number + second_number
    print("Result of added tool is: ", ret)
    return ret

# No more examples
adder_tool: Tool = Tool("Adder", "Adds two numbers and returns the result", adder)

result: str = Task(f"What's 2+2 ?", agent1, tools=[adder_tool]).solve().content

print(f"Equation result = {result}")
```

Output:
```
INFO: [PROMPT]: I give you the following tool definition that you must use to fulfill a future task: adder(first_number: int, second_number: int) -> int - Adds two numbers and returns the result. Please acknowledge the given tool.

INFO: [AI_RESPONSE]: Acknowledged!

I have taken note of the `adder` tool definition:

`adder(first_number: int, second_number: int) -> int`

This tool takes two integer arguments, `first_number` and `second_number`, and returns their sum as an integer.

I'm ready to use this tool when needed.

INFO: [PROMPT]: To use the tool you MUST extract each parameter and use it as a JSON key like this: {"arg1": "<value1>", "arg2": "<value2>"}. You must respect arguments type. For instance, the tool `getWeather(city: str, lat: int, long: int)` would be structured like this {"city": "new-york", "lat": 10, "lon": 20}. In our case, the tool call you must use must look like that: {'first_number': 'arg 0', 'second_number': 'arg 1'}

INFO: [AI_RESPONSE]: Thank you for the clarification.

For the `adder` tool, I will extract each parameter and use it as a JSON key. Here is the result:

{"first__number": "arg 0", "second__number": "arg 1"}

I will make sure to respect the argument types (in this case, both are integers) when using this structure in the future.

Thank you for the guidance!

INFO: [PROMPT]: You have a task to solve. Use the tool at your disposition to solve the task by outputing as JSON the correct arguments. In return you will get an answer from the tool. The task is:
What's 2+2 ?

INFO: [AI_RESPONSE]: {"first__number": 2, "second__number": 2}

WARNING: Yacana failed to call tool 'Adder' correctly based on the LLM output


INFO: [PROMPT]: The tool returned an error: `adder() got an unexpected keyword argument 'first__number'`
Using this error message, fix the JSON arguments you gave.
Remember that you must output ONLY the tool arguments as valid JSON. For instance: {'first_number': 'arg 0', 'second_number': 'arg 1'}

INFO: [AI_RESPONSE]: {"first_number": "arg 0", "second_number": "arg 1"}
Tool adder was called with param 'arg 0' (<class 'str'>) and 'arg 1' (<class 'str'>)

WARNING: Tool 'Adder' raised an error


INFO: [PROMPT]: The tool returned an error: `Parameter 'first_number' expected a type integer`
Using this error message, fix the JSON arguments you gave.


INFO: [AI_RESPONSE]: {"first_number": 2, "second_number": 2}
Tool adder was called with param '2' (<class 'int'>) and '2' (<class 'int'>)
Result of added tool is:  4
Equation result = 4
```

It worked !

2 errors happened here:
* "WARNING: Yacana failed to call tool 'Adder' correctly based on the LLM output"
* "WARNING: Tool 'Adder' raised an error"

**Error 1**: Regarding the first one if you look closely the output you can see a strange malformation in the JSON : `{"first__number": "arg 0", "second__number": "arg 1"}`. The first parameter was called with two underscores for some reasons (LLMs...). Fortunatly Yacana banged on the LLM's head and it was fixed in the next iteration.  
**Error 2**: Concerning the second error, it was definitely the tool that raised the exception: `The tool returned an error: `Parameter 'first_number' expected a type integer`. This is only logical as the LLM sent castrophic values to the tool: `{'first_number': 'arg 0', 'second_number': 'arg 1'}`. When the ToolError was sent the error message was given to the LLM and a third iteration started. This time all was correct: `{"first_number": 2, "second_number": 2}` and we got our result which is 4.

#### Maximum tool errors

You should combine both methods described above. But adding heavy tool validation with great error messages on what went wrong is always what yields the best results and is also the safest option.  

So what happens if the LLM 


## VIII. Assigning multiple Tools

## IX. Chat between two Agents

## X. Chat between many Agents
