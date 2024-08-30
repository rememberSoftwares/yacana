# Yacana
Tasks-driven agentic framework that gives you the building blocks to create AI workflows for local LLMs.  

# Functionalities
* Built for getting the best results with local opensource LLMs ;  
* Multi-Agent workflows ;  
* Tool calling for all models, even dumb ones ;
* Multi-turn chat between Agents ;
* Routing ;

...and many more!

# Why choose Yacana over other frameworks?

Yacana was built to make local LLMs great! It won't work with chatGPT, and we don't intend to make it. An important part of the future lies in local open source models. Many companies cannot afford paid subscriptions to proprietary models or won't for privacy issues. Also, agentic frameworks tend to cost way more than simple chat, generating cloud bills that could ruin a business. But why pay for AI when you can have it for only a fraction of the cost hosted on a single GPU machine or the device itself? This is where Yacana can help you harness the power of LLMs whatever the size and achieve production-grade results.

> What other good frameworks are out there?  
* **CrewAI** is very simple to use but is terrible when dealing with local models. This is because most of the heavy lifting is left to the LLM and if chatGPT or any other proprietary frontier models can deal with complex prompting, local LLMs cannot. Can we expect the same performance from 8 Billion parameters proprietary models with over 1000 Billion parameters? it's impossible. That's why CrewAI doesn't do well with local LLMs and also. This means that it cannot call tools! And calling tool is the most useful functionality a framework should offer.  

➡️**Why would Yacana be better suited ?** We loved how easy creating agents in CrewAI was. So when building Yacana it was clear to us that it needed the same level of simplicity. Learning to program with Yacana is very easy and only takes a few minutes. Also, the tool-calling feature was of paramount importance. That's why we worked on making Yacana's tool-calling ability so good that even tiny LLMs like Phi can call tools with almost a 100% chance of success.

* **LangGraph** is one of the best tools of 2024 to work with local models. It allows to creation of complex workflows that guide the LLM along a specific route to achieve a specific result. The problem with langGraph is its learning curve that is far above many developers and the spotty documentation doesn't help. It does allow tool calling but only with models that support function calling natively. Learning LangGraph has a one in three chance to give you a burnout. Also, the graph definition, besides being a great conceptual idea, is poorly implemented (IMHO) and the fact that they introduced a web UI to control the framework seems like an admission of failure (even though the UI looks great). But as a developer, I don't want to rely on yet another tool.

➡️**Why would Yacana be better suited?** The LangGraph nodes system that guides LLMs in a graph is great and is the reason why it succeeds with local LLMs. Knowing this we build Yacana with the same concept. But instead of having a complex graph system Yacana provides a classic programming interface in Python that can be leveraged by any developers regardless of their level of expertise. No burnouts here. So, Yacana provides its own way to chain Tasks and achieve a computable result in the end that can be used in classic programming products. This makes Yacana a production-ready tool that can be easily integrated with any software.

**In summary...** Yacana took the simplicity of CrewAI, and its capability to have multi-agent chats but followed the LangGraph principle of guidance using a dedicated Task system that guarantees a computable result at the end of the workflow. Also, Yacana is the only Framework proposing a tool-calling system with a high success rate.  
Also, it's free and open source (MIT) forever. It is made by the community, for the community.  

**What's the catch ?** Well nothing really. But for transparency, you must understand that this version has only been tested with Ollama as an inference server. We know that you guys out there are also using other inference technologies like VLLM, etc. Yacana has not been tested with this type of tech... Yet! The next update will focus on making sure that performance when connected to something other than Ollama doesn't degrade in any way. Stay tuned. This will be addressed in the next few weeks. Also, note that we are not saying that it's not compatible with other inference servers. We're saying that we don't know yet. ^^

# Quickstart

## I. Installation

### Installing Ollama

![image](https://github.com/user-attachments/assets/f3c45d0e-efca-4853-8237-3e56d90e1747)

As mentioned previously, the current version of Yacana has only been tested on Ollama. This will be sorted out in the next few weeks.  
To work with Yacana you'll need an Ollama server. If your computer is performant enough you can use it to run LLMs directly on your machine. Ollama is a great inference server and the most simple to install. Click [here](https://ollama.com/download) to get the latest release.
Ollama is:
* Compatible with all operating systems Windows/Max/Linux ;  
* Gets installed using one command ;  
* Has a great CLi that even a 4-year-old can use to download models ;  
* Tons of tutorials out there if you run into any type of trouble ;

ℹ️ You can connect Yacana to a remote Ollama instance. Read forward.  

### Choosing an LLM model

After Ollama is installed you can browse the list of available LLMs on the [Ollama website](https://ollama.com/library) and download any model you want (or your computer can deal with).  
For reference, if you don't know what LLM model to choose (we've all been there) here is a list of models you can try out on consumer hardware:  
| Computer power   |      LLM models name to try      |  LLM quality |
|----------|:-------------:|:------:|
| **Out of this world** *(GC RTX 4090 / 64 GB RAM)* | 'llama3.1:70b' or 'mixtral:8x22b' | Excellent reasoning and instruction following. |
| **Epic** *(GC RTX 4090 / 32 GB RAM)* | 'llama3.1:8b' or 'dolphin-mixtral:8x7b' or 'dolphin-mixtral:8x7b-v2.5-q6_K' | Good reasoning and instruction following. (q6_K model should be less consuming than the default Mixtral if you have any issues) |
| **Gamer** *(GC GTX 1080TI / 16 GB RAM)* |    'llama3.1:8b' or 'mistral:7b'   | Llama still works but is slower. Expect limited reasoning and no more than 2 complex instructions at a time  |
| **Potato** | 'phi:2.7b' or 'phi3:3.8b' or 'tinyllama:1.1b' | Almost no reasoning, incapable of following more than 1 instruction at a time, English bound only ; Dumb as a stone |

---

If you have access to a GPU VM with A LOT of RAM then you should try the state of the art 'llama3.1:405b' model.

### Running the model

When you have chosen your model it is time to use Ollama CLI to pull it on your computer. Simply do `ollama pull <model_name>`. The download will start immediately. When finished you can test the model locally by doing `ollama run <model_name>` which will start a conversation with the LLM.

### Installing Yacana

```
pip3 install yacana
```

## II. Imports

When using other frameworks 'import hell' quickly appears. To prevent this bothersome problem we propose that you import all of Yacana's modules and when finished developing let the IDE remove the unused imports. Unused imports generally appear grayed. Thus we recommend that you prepend these imports in all your files and clean them later. This way the IDE will have auto-completion available and will help you develop 10 times faster.

## III. Creating your first Agent

### Initializing the Agent
Now that you have an Ollama server running and Yacana installed let's create our first agent!  

Create a Python file with this content:
```python
agent1 = Agent("AI assistant", "llama3.1:8b", system_prompt="You are a helpful AI assistant", endpoint="http://127.0.0.1:11434")
```
The Agent(...) class takes 2 mandatory parameters:
1. **The agent name**: Choose something short about the agent's global focus
2. **A model name**: The Ollama model that this Agent will use. You may have multiple Agents running different models. Some models are better suited for some specific jobs so it can be interesting to mix LLM models.

The Agent(...) class has many optional parameters that we will discover in this tutorial. Here we can see 2 of them:  
1. **The system prompt**: Helps define the personality of the Agent.
2. **The endpoint**: The URL of your Ollama instance. It points by default to your localhost and on the Ollama default port. If you are using Ollama on your computer you can remove this optional parameter and the default value will be used.

### Testing Yacana and Ollama's interaction

This framework is not meant for basic roleplay. However, for people starting their journey in the realm of AI and for debugging purposes, we added a simple chat system. Add this line to test it :
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

Let's change the **system prompt** and have some fun!
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

The whole concept of the framework lies here. If you understand this following section then you have mastered 80% of Yacana's building principle. Like in LangGraph, where you create nodes that you link together, Yacana has a Task() class which takes as arguments a task to solve. There are no hardcoded links between the Tasks so it's easy to refactor and move things around. The important concept to grasp here is that through these Task() classes you will give instructions to the LLM in a way that the result must be a computable result. So instructions must be clear and the prompt to use must reflect that. It's a Task, it's a job, it's something that needs solving but written like it is given as an order! Let's see some examples :

```python
# First, let's make a basic AI agent
agent1 = Agent("AI assistant", "llama3:8b", system_prompt="You are a helpful AI assistant")

# Now we create a task and assign the agent1 to the task
task1 = Task(f"Solve the equation 2 + 2 and output the result", agent1)

# So that something actually happens you must call the .solve() method on your task
task1.solve()
```

What's happening above?  
* First, we instantiated an Agent with the `llama3:8b` model. You might need to update that depending on what LLM you downloaded from Ollama ;
* Second, we instantiated a Task ;
* Third, we asked that the Task be solved ;

ℹ️ For easing the learning curve the default logging level is INFO. It will show what is going on in Yacana. Note that not ALL intermediary prompts are shown.

The output of execution should look like this:
```
INFO: [PROMPT]: Solve the equation 2 + 2 and output the result

INFO: [AI_RESPONSE]: The answer to the equation 2 + 2 is... (drumroll please)... 4!

So, the result of solving the equation 2 + 2 is indeed 4.
```

If your terminal is working normally you should see the task's prompts in green and starting with the '[PROMPT]' string. The LLM's answer should appear purple and start with the [AI_RESPONSE] string.  

### About Task parameters

The Task class takes 2 mandatory parameters:
* The prompt: It is the task to be solved. Use imperative language, be precise, and ask for step-by-step thinking for complex Tasks and expected outputs if needed.
* The Agent: The agent that will be assigned to this task. The agent will be in charge of solving the task.

ℹ️ Many other parameters can be given to a Task. We will see some of them in the following sections of this tutorial. But you can check out the Task class documentation @todo URL

### Do you see how disruptive this is compared to other frameworks?

In the above code snippet, we assigned the agent to the Task. So it's the Task that leads the direction that the AI takes. In most other frameworks it's the opposite, where you assign some work to an existing agent. This reversed way allows to have fined grained control on each resolution step as the LLM only follows bread crumb (the Tasks). The pattern will become even more obvious as we get to the Tool section of this tutorial. As you'll see the Tools are also assigned at the Task level and not to the Agent directly.  

To compare with LangGraph, we indeed cannot make a call graph as we don't bind the Task together explicitly. However, Yacana's way gives more flexibility and allows a hierarchical programming way or scheduling the Task and keeping control of the flow. It also allows creating new Task dynamically if the need arises. You shall rely on your programming skills and good OOP to have a clean code and good Task ordering.

### Getting the result of a Task

Even though we get logs on the standard output of the terminal, we still need to extract the answer of the LLM that solved that Task to actually do something with it.  
Getting the string message is quite easy as the .solve() method returns a Message() class.  
Maybe you are thinking "Ho noo another class to deal with". Well, let me tell you that it's always better to have an OOP class than some semi-random Python dictionary where you'll forget what keys it contains. Also, the Message class is very straightforward. It exposes a `content` attribute. Modify the current code like this:
```python
# So that something actually happens you must call the .solve() method on your task
my_message: Message = task1.solve()

# Printing the LLM's response
print(f"The AI response to our task is : {my_message.content}")
```
There you go! Give it a try.

ℹ️ Note that we used duck typing, which is postfixing all variables declaration with their type `my_message: Message`. Yacana's source code is entirely duck-typed so that your IDE always knows what type it's dealing with and proposes you with the best methods and arguments. We recommend that you do the same as  it's the industry's best standards.

---

Don't like having 100 lines of code for something simple? Then chain them all in one line!
```
# First, let's make a basic AI agent
agent1 = Agent("AI assistant", "llama3:8b", system_prompt="You are a helpful AI assistant")

# Creating the task, solving it, extracting the result, printint the result
print(f"The AI response to our task is: {Task(f'Solve the equation 2 + 2 and output the result', agent1).solve().content}")
```
🤔 If I were you I would do the message extraction on one line and the print on a new one. Let's not one-line things that much 😅.


### Chaining Tasks

Chaining Tasks is nothing more than just calling a second Task with the same Agent. Agents keep track of the History of what they did (aka the Tasks they solved). So just call a second Task and assign the same Agent. For instance, let's multiply by 2 the result of the initial Task. Append this to our current script:

```
task2_result: str = Task(f'Multiply by 2 our previous result', agent1).solve().content
print(f"The AI response to our task is : {task2_result}")
```
You should get:
```
The AI response to our task is: If we multiply the previous result of 4 by 2, we get:

8
```

ℹ️ Without tools this only relies on the LLM's ability to do the maths and is dependent on its training.

---

See? The assigned Agent remembered that it had solved the task1 previously and used this information to solve the second task.  
You can chain as many Tasks as you need. Also, you should create other Agents that don't have the knowledge of previous Task and make them do things based on the output of your first agent. You can build anything now!    

## V. Routing

Other frameworks have the tendency to make abstractions for everything. Even things that don't need any. That's why I'll show you how to do routing with only what we have seen earlier. Yacana doesn't provide routing abstraction because there is no need to do so.  

But what is routing? Well, having LLMs solve a Task and then chaining many others in a sequence is good but to be efficient you have to create conditional workflows. In particular when using local LLMs that don't have the power to solve all Tasks with only one prompt. You have to create an AI workflow in advance that will go forward step by step and converge to some expected result. AI allows you to deal with some level of unknown but expecting that you can have a master brain (like in crewAI) that distributes Tasks to agents and achieves an expected result is IMPOSSIBLE with local LLMs. They are too dumb! Therefore they need you to help them along their path. This is why LangGraph works well with local LLMs and Yacana does too. You create a workflow and when conditions are met you switch from one branch to another that treats more specific cases, etc.

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
    # next step in the workflow that involves plants

elif "no" in router_answer.lower():
    print("Question is NOT about plants")
    # Next step in the workflow that DOESN'T involve plants
```

You should get the following output:
```
INFO: [PROMPT]: Is the following question about plants? <question>Why do leaves fall in autumn ?</question> Answer ONLY by 'yes' or 'no'.

INFO: [AI_RESPONSE]: yes
Question is about plants
```

▶️ Many things are happening here. We didn't implement an abstraction to simplify things but the downside is that you must learn a few tricks:
1. **Always compare with lower case string**: Because LLMs have their own mind they do not always answer a straight `yes`. Sometimes you get "Yes" or "No" or even full cap "YES" for no reason.
2. **Always start by searching for "yes"**: We do a substring match using the `in` keyword of Python because the LLM doesn't always respect the instructions of outputting "ONLY 'yes' or 'no'". Sometimes you'll get "yes!" or "Great idea, I say yes". Substring match will match "yes" anywhere in the LLM answer. But what if you matched "no" first and the LLM generated "Not sure but I would say yes". Because we search for substrings the condition would match the "no" part of the word "Not" even though the LLM said yes. We could use regexe to fix this but it's easier to just start the condition by looking for "yes" as there are no English words that contains "yes" in the substring (at least no common ones ^^).
3. **Push the model to respect the instruction**: Tell it to "answer ONLY with 'xx'". See the use of upper cap to "ONLY"? Also, the single quotes around the possible choices 'yes' | 'no' helps the LLM that sees them as delimiters.
4. **Use formatting tags**: The question that is mentioned in the prompt is then given in custom `<question>` tags. LLMs love delimiters. This way the LLM knows when the question starts and when the question ends. This technique helps to differentiate your prompt and the dynamic part. You don't have to add tags everywhere but they can prove useful. Do not abuse them or the LLM might start using them in its response.

ℹ️ This is all basic prompt engineering. If you wish to build an app with local models you will definitely have to learn those tricks. LLMs are unpredictable. It's why we built them.  

### Self-reflection routing

As local models are a bit dumb you need to leave them think on their own before making a decision. This is called self-reflection. It will cost one more Task to solve but you'll get significantly better results during routing, in particular when routing on more complex things (other than "yes"|"no").

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

See how the LLM had an "intense" reflection on the subject. This is very good. You want LLMs to do reasoning like this. It will improve the overall result for the next Tasks to solve.  

▶️The prompt engineering techniques used here are:
1. **Make it think**: Using the expression "Explain your reasoning." makes it generate a logical answer. Note that if the model is bad at reasoning or makes a mistake during this step it may result in extremely bad situations. But fear not, failsafe can be built to limit bad reasoning. For instance, having another LLM check the logic and interacting with the original Agent (see GroupSolve later on) to show it its mistake or you can give tools to the Agent that will help it achieve the truth and not rely soly on his reasoning abilities (see Tools later on).
2. **Making it two shots**: Now that we have 2 Tasks instead of one, the second one only focuses on one task: "yes" or "no" interpretation of the result of Task1. Cutting objectives in multiple Tasks gives better performance. This why using an agentic framework is great but it's also why it's consuming a lot of tokens and having "free to run" local LLMs is great!

### Cleaning the history

Keeping the self-reflection prompt and the associated answer is always good. It helps gardrailing the model. But the "yes"/"no" router on the other hand adds unnecessary noise in the Agent's history. Moreover, local models don't have huge context window sizes, so removing useless interactions is always good.  
The "yes"/"no" router is only useful on the moment but we should make the Agent forget it ever happened after it answered. No need to keep that. This is why the Task class offers an optional parameter: `forget=False`.

Update the router line with this new parameter:
```python
router_answer: str = Task(f"To summarize in one word, was the question about plants ? Answer ONLY by 'yes' or 'no'.",
     agent1, forget=True).solve().content
```
Now, even though you cannot see it, the Agent doesn't remember solving this Task. In the next section, we'll see how to access and manipulate the history. Then, you'll be able to see what the Agent remembers!

### Demo time

For this demo, we'll make a simple app that takes a user query (HF replacing the static string by a Python `input()` if you wish) that checks if the query is about plants.
If it is not we end the workflow there. However, if it is about plants the flow will branch and search if a plant type/name was given. If it was then it is extracted and knowledge about the plant will be shown before answering the original question. If not it will simply answer the query as is.

![plant1B](https://github.com/user-attachments/assets/e479e74c-c4f4-4942-a8b5-bd06b377af8c)


Read from bottom ⬇️ to top ⬆️. (Although the Agent and question are defined globally at the top)

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


# Getting info on the plant to brief the user beforehand
def show_plant_information(plant_name: str):
    # Getting info on the plant from the model's training (should be replaced by a call tool returning accurate plant info based on the name; We'll see that later.) 
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
    # Self-reflection
    Task(
        f"In your opinion, does the question mention a specific plant name or one that you can identify ?",
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

    # self-reflection
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
        # We stop here; This app is only about plants!


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


**Updating question to match the "Not related to plants" workflow:**

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

As entering the AI landscape can get a bit hairy we decided to leave the INFO log level by default. This allows to log to the standard output all the request made to the LLM.  
Note that not everything of Yacana's internal magic appears in these logs. We don't show everything because there are many time-traveling things going around the history of an Agent and printing a log at the time it is generated wouldn't make sense for you.  
However, we try to log a maximum of information to help you understand what is happening internally and allow you to tweak your prompts accordingly.  

Nonetheless, you are the master of what is logged and what isn't. You cannot let Yacana logs activated when working with an app in production.  
There are 5 levels of logs:  
1. "DEBUG"
2. "INFO" <= Default
3. "WARNING"
4. "ERROR"
5. "CRITICAL"
6. None <= No logs

To configure the log simply add this line at the start of your program:
```python
LoggerManager.set_log_level("INFO")
```

ℹ️ Note that Yacana utilizes the python logging package. This means that setting de level to "DEBUG" will print other libraries' logs on the debug level too.

If you need a library to stop spamming could do:
```python
LoggerManager.set_library_log_level("httpx", "WARNING")
```
The above example sets the logging level of the network httpx library to warning, thus reducing the log spamming.  

### Configuring the LLM internal settings

LLMs have settings that can be tweaked through Ollama but also many inference servers. For instance, lowering the `temperature` setting makes the model less creative in its responses. On the contrary, raising this setting will make the LLM more chatty and creative.  
Yacana provides you with a class that exposes all the possible LLM properties. Look at them [here]() @todo url and if you need a good explanation of each of them I would recommend the [excellent video](https://youtu.be/QfFRNF5AhME?si=lpSYUq2WoidYiqzP) Matt Williams did on this subject.  

These settings are set at the Agent level so that you can have the same model used by two separate agents and have them behave differently.  

We use the [ModelSettings]() @todo url class to configure the settings we need. 

For example, let's lower the temperature of an Agent to 0.4:
```
ms = ModelSettings(temperature=0.4)

agent1 = Agent("Ai assistant", "llama3:8b", model_settings=ms)
```
If you're wondering what are the default values of these when not set. Well, Ollama sets the default for you. They can also be overridden in the Model config file (looks like a dockerfile but for LLM) and finally, you can set them through Yacana during runtime.  

A good way to show how this can have a real impact on the output, is by setting the `num_predict` parameter. This one allows control of how many tokens should be generated by the LLM. Let's make the same Task twice but with different `num_predict` values:
```python
# Setting temperature and max token to 100
ms = ModelSettings(temperature=0.4, num_predict=100)

agent1 = Agent("Ai assistant", "llama3:8b", model_settings=ms)
Task("Why is the sky blue ?", agent1).solve()

print("-------------------")

# Settings max token to 15
ms = ModelSettings(num_predict=15)

agent2 = Agent("Ai assistant", "llama3:8b", model_settings=ms)
Task("Why is the sky blue ?", agent2).solve()
```

Output:
```python
INFO: [PROMPT]: Why is the sky blue ?

INFO: [AI_RESPONSE]: The sky appears blue because of a phenomenon called Rayleigh scattering, named after the British physicist Lord Rayleigh. Here's what happens:

1. **Sunlight**: When sunlight enters Earth's atmosphere, it contains all the colors of the visible spectrum (red, orange, yellow, green, blue, indigo, and violet).
2. **Molecules**: The atmosphere is made up of tiny molecules of gases like nitrogen (N2) and oxygen (O2). These molecules are much smaller than

-------------------

INFO: [PROMPT]: Why is the sky blue ?

INFO: [AI_RESPONSE]: The sky appears blue because of a phenomenon called Rayleigh scattering, named after
```
As you can see above the two agents didn't output the same number of tokens.

## VI. Managing Agents history

As you saw in the previous examples, each agent has his own history of messages that compose its memory. When a new request is made to the LLM the whole history is sent to the inference server (ie: Ollama) and the LLM responds to the last prompt in the chain but bases its answer on the context it gets from the previous messages ( and the initial system prompt if present). 

This is what a history looks like:

![history1A](https://github.com/user-attachments/assets/631b634a-8699-4fff-9ac4-06b403c06ae1)

There are 3 types of prompts. The optional system prompt that, if present, always go first. Then it's only an alternation between the "User prompts" that come from the Task you set and the "Assistant message" which is the answer from the LLM.

However, sending the whole history to the LLM for each Task to solve has some disadvantages that can not be overturned:
* The longer the history the longer the LLM takes to analyze it and return an answer.
* Each LLM comes with a maximum token window size. This is the maximum number of words an LLM can analyze in one run, therefore it's maximum memory. One token roughly represents one word or 3/4 of a word. More information on token count per word [here](https://winder.ai/calculating-token-counts-llm-context-windows-practical-guide/) or [here](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them).  

To counteract these negative effects it is recommended you clean the history when possible. You can use the `forget=True` parameter in the Task() class so the prompt and the LLM response do not get saved to the history ([see here]() @todo). There are other ways to preserve the history from useless noise. But first, we'll look at viewing one's Agent history. Fortunately, Yacana got you covered.

### Printing History

The Agent class comes with a `.history` property of type `History` (@todo [see here]()). It exposes methods so you can manipulate and view it (for debugging, mostly) during runtime. For instance, it has a `.pretty_print()` method that prints on the standard output the content of the history using the classic color scheme. It's great for debugging but not for parsing. If you need to parse the history there is a `.get_as_dict()` method which as the name implies returns the History as a Python dictionary.  

Let's see a simple example:
```python
# Let's deactivate automatic logging so that only OUR prints are shown
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

As mentioned earlier it's better to keep the History clean. Too many prompts and unrelated questions will lead to poorer results so if you have the opportunity to scratch some portion then you should.  
Yacana allows you to make a history snapshot and rollback to any of them. This is in particullary useful when you are reaching the end of a flow branch and want to go back onto another branch.  

It is as simple as this:  
```python
# Creating a checkpoint
checkpoint_id: str = agent1.history.create_check_point()
```
The checkpoint_id is only a unique identifier that you can use to load back a save. Like this:  
```python
# Go back in time to when the checkpoint was created
agent1.history.load_check_point(checkpoint_id)
```

ℹ️ Note that nothing prevents you from making a snapshot before rolling back to a previous save. This way you could return... to the future. ^^  
Are you okay Marty?  

Let's take a concrete example. You have a pastry website that generates pastry recipes (using real data from a tool would be better, continue the tutorial). 
The flow will look like this:
1. Propose 5 pastry names ;
2. Create a checkpoint ;
3. The user chooses one of the pastries ;
4. We show the associated calories of the selected pastry ;
5. If the user is okay with it we end the program ;
6. If the user is not okay with the calorie count we load back the checkpoint and propose to choose from the the list again ;
7. Repeat until satisfied ;
8. We'll show the final agent's History and make sure that it ONLY stored the selected pastry ;

With a bit of color, it would look like this:

![pastry1B](https://github.com/user-attachments/assets/3a4952aa-18f3-4b6d-93c1-85b909cf24f4)


```python
# Let's deactivate automatic logging so that only OUR prints are shown; Maybe reactivate (to "info") if you want to see what's happening behind the scenes.
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

    # Asking for one of the pastries from the list
    user_choice: str = input("Please choose one of the above pastries: ")

    # Printing associated calories for the selected pastry
    pastry_calorie_question: str = Task(f"The user said '{user_choice}'. Your task is to output a specific sentence and replace the <replace> tags with the correct values: 'You selected the <replace>selected pastry</replace>. The average calorie intake for this pastry is <replace>average associated calories for the selected pastry</replace>. Do you wish to continue ?", agent1).solve().content
    print(pastry_calorie_question)

    # Asking if the user wants to continue
    is_satisfied: str = input("Continue ? ")

    # Basic yes / no router
    router_answer: str = Task(f"The user said '{is_satisfied}'. Evaluate if the user was okay with its order. If he was, ONLY output 'yes', if not only output 'no'.", agent1).solve().content

    if "yes" in router_answer.lower():
        print("Thank you for your order.")
        # The user was satisfied with his choice. Exiting the loop...
        break
    else:
        # The user wants to choose another pastry. Let's go back in time by loading are previous checkpoint!
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

Please choose one of the above pastries: Hummm. The golden one?
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
The user said 'yes'. Evaluate if the user was okay with the order. If he was, ONLY output 'yes', if not only output 'no'.

[assistant]:
yes

############## END ##################
```

As you can see in the above output. I went for "the creamy one" but when shown the calories I said no... Then chose the Golden Galettes was satisfied and the program ended with an output of the agent's history.  
We can see in the agent output that it only remembers me choosing the Golden Galettes but not the Creamy Confections. This is because we loaded the checkpoint that was made before the user choice.

ℹ️ Note that the Task asking to replace the variables might not work very well with dumb LLMs. It could be reworked by splitting the Task into two. One that would extract the name of the chosen pastry from the user's input and a second one that would generate the associated calories. Finally, print the sentence with the variables pre-generated. Using local models is all about knowing the maximum performance of your LLM and adapting the prompts to match that performance. The dumber, the more guidance it needs!  

### Multi prompt shot VS 0 prompt shot

When a LLM struggles to solve a complex Task and achieve a good success rate it may be time to give it a little help.  

In large language models, the approach to prompting can significantly influence the model's performance. *Zero-shot prompting* asks the model to complete a task without any prior examples, relying solely on its pre-existing knowledge. This can lead to varied results, especially in more complex tasks. *One-shot prompting* improves accuracy by providing the model with a single example, offering some guidance on how to approach the task. *Few-shot prompting* further enhances performance by supplying multiple examples, allowing the model to better understand the task's nuances and produce more reliable and accurate results.

In other words:
* Zero-Shot Prompting: The model must rely on its general knowledge, which may result in varied performance depending on the task's complexity.
* One-Shot Prompting: A single example helps guide the model, improving its performance over zero-shot prompting.
* Few-Shot Prompting: Multiple examples give the model a better understanding of the task, significantly increasing accuracy and performance.

Yacana provides you with a way to add new Messages to the History manually. The History class exposes a `.add(...)` method.  
This method takes an argument of type `Message()` ([[see here]() @todo) taking two parameters: a [MessageRole]() @todo url enum and the string message itself.

For example:
```python
# Creating a basic agent with an empty history
agent1 = Agent("AI assistant", "llama3:8b")

# We create a fake prompt identified as coming the user (MessageRole.USER)
user_message = Message(MessageRole.USER, "What's 2+2 ?")

# We create a fake answer identified as coming from the LLM (MessageRole.ASSISTANT)
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

The Agent's History contains the two messages we manualy added.  
ℹ️ Note that the History doesn't have to be empty to use this. The .add() method only appends your messages to the end of the current History.  
⚠️ Though, try to keep the alternation of USER and ASSISTANT as this is how "instruct" LLMs have been trained.  

---

Let's see a 0 shot example asking for a json output extracted from a given sentence:

```python
agent1 = Agent("Ai assistant", "llama3:8b")

Task(f"Print the following sentence as JSON, extracting the names and rephrasing the actions: 'Marie is walking her dog. Ryan is watching them through the window. The dark sky is pouring down heavy raindrops.'", agent1).solve()
```

Outputs:
```
INFO: [PROMPT]: Print the following sentence as JSON extracting the names and rephrasing the actions: 'Marie is walking her dog. Ryan is watching them through the window. The dark sky is pouring down heavy raindrops.'

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
Let's introduce another optional Task() parameter: `json_output=True`. This relies on Ollama to force the output as JSON.  
⚠️ It is preferable to prompt the LLM to output as JSON in addition to this option.

Replace our Task with this one:
```python
Task(f"Print the following sentence as JSON extracting the names and rephrasing the actions: 'Marie is walking her dog. Ryan is watching them through the window. The dark sky is pouring down heavy raindrops.'", agent1, json_output=True).solve()
```

Outputs:
```
INFO: [PROMPT]: Print the following sentence as JSON extracting the names and rephrasing the actions: 'Marie is walking her dog. Ryan is watching them through the window. The dark sky is pouring down heavy raindrops.'

INFO: [AI_RESPONSE]: {"names": ["Marie", "Ryan"], "actions": {"Marie": "is walking", "Ryan": "is watching"}, "description": [{"location": "window", "activity": "watching"}, {"location": "outdoors", "activity": "pouring raindrops"}]}
```

Way better. No more noise.  
However, we would prefer having an array of `name` and `action`, even for the weather (the name would be *sky* and action *raining*).

To achieve this let's give the LLM an example of what we expect by making it believe it outputted it correctly once:
```python
agent1 = Agent("Ai assistant", "llama3:8b")

# Making a fake interaction that is correct
agent1.history.add(Message(MessageRole.USER, "Print the following sentence as json extracting the names and rephrasing the actions: 'John is reading a book on the porch while the cold wind blows through the trees.'"))
agent1.history.add(Message(MessageRole.ASSISTANT, '[{"name": "John", "action": "Reading a book.", "Cold wind": "Blowing through the trees."]'))

Task(f"Print the following sentence as json extracting the names and rephrasing the actions: 'Marie is walking her dog. Ryan is watching them through the window. The dark sky is pouring down heavy raindrops.'", agent1).solve()
```

Outputs:
```
INFO: [PROMPT]: Print the following sentence as JSON extracting the names and rephrasing the actions: 'Marie is walking her dog. Ryan is watching them through the window. The dark sky is pouring down heavy raindrops.'

INFO: [AI_RESPONSE]: [{"name": "Marie", "action": "Walking her dog."}, {"name": "Ryan", "action": "Watching Marie and her dog through the window."}, {"name": "The dark sky", "action": "Pouring down heavy raindrops."}]
```

This is perfect.  
You can add multiple fake interactions like this one to cover more advanced cases and train the LLM on how to react when they happen. It would become multi-shot prompting.  

---

You can also do multi-shot prompting with self-reflection. This takes more CPU time because you decompose the task into multiple subtasks but can be beneficial in some scenarios.  

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
In this case, it didn't work very well as only one name was extracted as JSON. But in more complex scenarios I can assure you that letting the LLM reflect on the guideline beforehand, can be very beneficial to solving the task.  

### Saving an Agent state

Maybe your program needs to start, stop, and resume where it stopped. For this usecase Yacana provides a way to store an Agent state on file and load it later. All of the Agent's properties are saved including the History. Only checkpoints are lost as they are more of a runtime thing. We might include them in the save file if the need arises.

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

Now let's load back this agent from the dead using `.get_agent_from_state()`!  
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
As you can see when asked to multiply by 2 the previous result, it remembered agent1's result which was 4. It then did 4 x 2 and got us 8.  

## VII. Assigning a tool to a Task

### Introduction

Allowing the LLM to call a tool is the most important thing an agent can do! But what is a "tool"? A "tool" simply refers to a Python function. This function can be the entry point to any level of underlying complexity. But it doesn't matter. What matters is that the LLM can call the tool with parameters that match the function. This way, LLMs can interact with *classic* programming interfaces that produce deterministic results (aka normal programming).  

For instance, let's say you want a calculator powered by an LLM. You cannot rely on the LLM doing the math because even though it knows how to decompose equations to an extent and basic arithmetics, it will fail on more advanced calculous. Therefore we do not expect the LLM to perform the operation itself. We already have the CPU to do this task perfectly. On the other hand, we expect the LLM to decompose correctly the equation and call tools for each arithmetic operation needed to get the result.  

#### In what way is Yacana different than other frameworks?

Other frameworks assign their tools to the agent during its initialization. This creates a hard link between the tools and the agents. In our opinion, this implementation tends to confuse the agent because it's getting access to many tools that may not be relevant to the immediate task it is given. In Yacana tools are only available at the Task level. Thus no noise is generated before having to solve a particular task. The tool is made available to the LLM only when it's needed and not before. Also, the Agent doesn't keep the memory of having used the tool so it won't be tempted to use it elsewhere, where it would not have been appropriate.  

#### Understanding the underlying mechanism of tool calling in LLMs

As an aside for those interested...

If you don't understand how a text-to-text neural network can call a Python function let me tell you: It doesn't.  
When we refer to *tool calling* we also refer to *function calling* which is very poorly named. Function calling is the ability of an inference server to make the LLM output the text in a particular format. As of today, only JSON is supported but there is no doubt that more formats will be available soon.  
However, now that we can control how the LLM answers, we can parse a JSON that we know the structure. Therefore we can ask the LLM for a JSON that matches the prototype of a Python function. For instance the name and parameter value.  
Some LLMs have been trained to output JSON in a particular way that matches a particular JSON structure. This particular JSON structure is becoming a convention and was pushed by big AI players like OpenAI.  
Unfortunately, the size and complexity of this JSON doesn't work very well with our dumb 8B LLMs. This a problem that ChatGPT, Claude, Grok and other smart LLMs don't have.  
To overcome this particular issue, Yacana comes with its own JSON to call Python functions. It's way lighter than the OpenAI standard and Yacana uses [percussive maintenance]() @todo url to force the model to output the JSON in a way that the tool expects.  

#### How to write a prompt for a tool?

The title spoils one of the most important things about tool calling in Yacana.  
**The prompt is to guide the LLM on how to use the tool and not what to do with the tool result!**  
It is of the utmost importance that you understand this concept. This implies that you will have to create a second Task to deal with the output of the first one.  

This is a step-by-step of the internal mechanism:  
1. The Task is asked to be solved ;
2. There is a Tool assigned, so the Agent must use it ;
3. Yacana provides examples of how the tool is used ;
4. Giving the initial Task prompt to the Agent ;
5. The agent decides what values to send the Tool based on the prompt and previous History ;
6. The Tool is called with the previously mentioned parameters ;
7. The tool returns a value ;
8. The tool's value is the final output of the Task

As you can see, nothing was made with the tool result itself. This means that the tool return value must carry all the necessary information so the next Task can work with it.

Example of a **bad** prompt with a Tool that gets the current weather of a city:  
* `Task(f"I give you the following city: '{city}'. Get the current weather there and output 'So much sun !' if there is some sun else say 'So  much rain !'", some_agent, tools=[get_weather_tool]).solve()` 
=> You'll never get the output 'sunny' or 'rainy'. The result of this Task will be the result of the Tool output. The prompt only served the Agent to know what city to give to the tool.  

ℹ️ In some next update we might add a "post tool prompt" so that you get a way to control the output of the task once the tool has answered.  

To fix the above scenario you have two options:  

1. Split the Task in two:  
* `Task(f"I give you the following city: '{city}'. Get the current weather there.", some_agent, tools=[get_weather_tool]).solve()`  
* `Task(f"Output 'So much sun !' if there is some sun else say 'So  much rain !'", some_agent).solve()`  
Splitting the tasks into two allows the second Task to work on the output of the first one (the tool output)  

2. Make the tool do the work:  
* Let's write a pseudo code get_weather tool
```python
def get_weather(city: str) -> str:
    some_json = curl weather.com?city=$city
    if some_json["sun_level_percent"] > 50:
      return "So much sun !"
    else:
      return "So much rain !"
```

This tool would return either "So much sun !" or "So much rain !" based on the output of some fake weather API. **However**, the original prompt it's still bad! The section "*output 'sunny' if there is some sun else say 'rainy'*" of the initial *bad* prompt is still useless as it's the tool that will output this string. Not the LLM.  
You should rewrite the prompt like this: "I give you the following city: '{city}'. Get the current weather there."  


---

The complete example for those who wish to test IRL. Though, we didn't even show a real tool example so this section should be optional!  

Bad prompt:  
```python
agent1 = Agent("AI assistant", "llama3:8b")

def get_weather(city: str) -> str:
    return "Sunny"

get_weather_tool = Tool("get_weather", "Returns the weather for a given city.", get_weather)

Task(f"I give you the following city: LA. Get the current weather there and output 'So much sun !' if there is some sun else say 'So  much rain !'", agent1, tools=[get_weather_tool]).solve()

print("--history--")
agent1.history.pretty_print()
```

Good prompt:  
```python
agent1 = Agent("AI assistant", "llama3:8b")

def get_weather(city: str) -> str:
    return "Sunny"


get_weather_tool = Tool("get_weather", "Returns the weather for a given city.", get_weather)

Task(f"I give you the following city: LA. Get the current weather there.", agent1, tools=[get_weather_tool]).solve()
Task(f"Output 'So much sun !' if there is some sun else say 'So  much rain !'", agent1).solve()

print("--history--")
agent1.history.pretty_print()
```



### Calling a tool

Let's do our first tool calling with a simple addition!

First, let's define our tool:
```python
def adder(first_number: int, second_number: int) -> int:
    print(f"Tool adder was called with param {first_number}) ({type(first_number)} and {second_number} ({type(second_number)})")
    return first_number + second_number
```

What do we have here?
* The name of the function must be relevant to what the function does. Here the function performs an addition so we'll call it `adder` ;
* The same thing goes for the parameters. The name you choose is very important as it will help the LLM to know what value to give this argument ;
* Duck typing the prototype is very important! You must define the type of each parameter and also the return type ;
* We perform the operation between the two parameters and return the result ;

⚠️ Be aware that whatever the return of your function, Yacana will cast it to string using the built-in `str(...)` function. LLMs can only understand the text so make sure that whatever you send back can be cast correctly (override the __str__ if needed).  

Let's create a [Tool]() @todo url using the Yacana Tool constructor. It takes a name, a description, and a reference to the actual function.  
ℹ️ I can only emphasize the importance of providing an accurate description.  
```python
adder_tool: Tool = Tool("Adder", "Adds two numbers and returns the result", adder)
```

Now let's assign our *adder_tool* to a Task. How to do that? It's simple, the Task() class takes an optional `tools=[]` parameter. It's an array so be sure not to forget those `[]`!  
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

INFO: [PROMPT]: To use the tool you MUST extract each parameter and use it as a JSON key like this: {"arg1": "<value1>", "arg2": "<value2>"}. You must respect the arguments type. For instance, the tool `getWeather(city: str, lat: int, long: int)` would be structured like this {"city": "new-york", "lat": 10, "lon": 20}. In our case, the tool call you must use must look like that: {'first_number': 'arg 0', 'second_number': 'arg 1'}

INFO: [AI_RESPONSE]: Thank you for clarifying how to structure the tool calls.

In that case, I will extract each parameter and use it as a JSON key. For the `adder` tool, I will structure the tool call as follows:

{'first_number': 3, 'second_number': 5}

Please let me know what's next!

INFO: [PROMPT]: You have a task to solve. Use the tool at your disposition to solve the task by outputting as JSON the correct arguments. In return you will get an answer from the tool. The task is:
What's 2+2 ?

INFO: [AI_RESPONSE]: {"first_number": "2", "second_number": "2"}

Tool adder was called with param '2' (<class 'str'>) and '2' (<class 'str'>)
Result of added tool is: 22

Equation result = 22
```

What you are seeing here is Yacana doing its magic to make the LLM call the tool.  

Unfortunatly, even though the tool is indeed called, getting a correct result failed spectacularly! ^^  
Is `2 + 2 = 22`? No, I don't think so. Can you find out what went wrong?

---

When looking at the logs we can see that the tool was called with the following JSON: `{"first_number": "2", "second_number": "2"}`. The given values are of type `string`. Later confirmed by the print() inside the tool itself: `param '2' (<class 'str'>) and '2' (<class 'str'>)`.  
So instead of having integers, we got string and what's the result of "2" + "2" in Python? Not 4 but "22" (concatenation of strings).  

Fortunately, we can fix this easily in different ways.

### Getting better tool-calling results

As you saw in the previous adder example we ran into trouble with the `2 + 2` call sent as a string. Let's fix that.

#### Providing tool call examples

If you followed this tutorial from the start you saw that multi-shot prompting yields good results. The Tool class allows this too, using the `usage_examples=[]` optional parameter. You have to provide a Python dictionary where each key is equal to the argument name and the associated value, a valid value for the tool. You can add as many examples as you want in the array. In general one or two is enough.  
These dictionaries will be presented by Yacana to the LLM as examples of how to call the tool correctly.  

Let's look at an example with our adder tool:
```
adder_tool: Tool = Tool("Adder", "Adds two numbers and returns the result", adder, usage_examples=[{"first_number": 2, "second_number": 4}, {"first_number": 8, "second_number": -2}])
```

Replace the previous adder_tool definition with this one. You see the that we provided two examples. Each time giving `first_number` and `second_number` have different integer values. No strings. Actual integers.

Let's run our program again and see if we get input types this time:
```python
INFO: [PROMPT]: I give you the following tool definition that you must use to fulfill a future task: adder(first_number: int, second_number: int) -> int - Adds two numbers and returns the result. Please acknowledge the given tool.

INFO: [AI_RESPONSE]: Acknowledged!

The tool definition provided is:

`adder(first_number: int, second_number: int) -> int`

This tool takes two integer inputs `first_number` and `second_number`, and returns their sum as an integer.

I'm ready to use this tool for any future tasks that require addition!

INFO: [PROMPT]: To use the tool you MUST extract each parameter and use it as a JSON key like this: {"arg1": "<value1>", "arg2": "<value2>"}. You must respect the arguments type. For instance, the tool `getWeather(city: str, lat: int, long: int)` would be structured like this {"city": "new-york", "lat": 10, "lon": 20}. In our case, the tool call you must use must look like that: {'first_number': 'arg 0', 'second_number': 'arg 1'}

INFO: [AI_RESPONSE]: I understand now!

So, for the `adder` tool, I need to extract each parameter and structure it as a JSON key-value pair. Here's the result:

{'first_number': 'int', 'second_number': 'int'}

This means that when using this tool, I should specify two integer values for `first_number` and `second_number`, respectively.

INFO: [PROMPT]: You have a task to solve. Use the tool at your disposition to solve the task by outputting as JSON the correct arguments. In return, you will get an answer from the tool. The task is:
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

It worked!
ℹ️ Note that the multi-shot prompts are not shown in the info logs. This is because no actual request is made to the LLM they are appended to the History() like shown in the multi-shot example. However, if you do a `agent1.history.pretty_print()` at the end you'll see both examples given to the LLM as history context.  

Seeing in the examples that the tool needed integers in input, it called the tool with the correct types therefore the adder tool returned `4` as it was expected. Houra.  

⚠️ Do not abuse this technic as it tends to create noise. Trying to manage too many hypothetical use cases might, in the end, degrade performances.  

#### Adding validation inside the Tool

The previous trick is good to nudge the LLM in the right direction. But it's not the best way to get accurate results. The technique presented here is by far more effective and should be preferred over the previous one.  

As LLM are not deterministic we can never assure what will be given to our tool. Therefore, you should look at a tool like you would a web server route. I'm talking here of server-side validation. Your tool must check that what is given to it is valid and raise an error if not.  

This means adding heavy checks on our tool. Thus, when the LLM sends an incorrect value an error will be raised. But not any error! Specifically a ToolError(...). This exception will be caught by Yacana which will instruct the LLM that something bad happened while calling the tool. This also means that you must give precise error messages in the exception because the LLM will try to change his tool calling based on this message.  

Let's upgrade our adder tool!
```python
    # Adding type validation
    if not (isinstance(first_number, int)):
        raise ToolError("Parameter 'first_number' expected a type integer")
    if not (isinstance(second_number, int)):
        raise ToolError("Parameter 'second_number' expected a type integer")
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

INFO: [PROMPT]: To use the tool you MUST extract each parameter and use it as a JSON key like this: {"arg1": "<value1>", "arg2": "<value2>"}. You must respect argument type. For instance, the tool `getWeather(city: str, lat: int, long: int)` would be structured like this {"city": "new-york", "lat": 10, "lon": 20}. In our case, the tool call you must use must look like that: {'first_number': 'arg 0', 'second_number': 'arg 1'}

INFO: [AI_RESPONSE]: Thank you for the clarification.

For the `adder` tool, I will extract each parameter and use it as a JSON key. Here is the result:

{"first__number": "arg 0", "second__number": "arg 1"}

I will make sure to respect the argument types (in this case, both are integers) when using this structure in the future.

Thank you for the guidance!

INFO: [PROMPT]: You have a task to solve. Use the tool at your disposition to solve the task by outputting as JSON the correct arguments. In return, you will get an answer from the tool. The task is:
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

It worked!

2 errors happened here:
* "WARNING: Yacana failed to call tool 'Adder' correctly based on the LLM output"
* "WARNING: Tool 'Adder' raised an error"

- **Error 1**: Regarding the first one if you look closely at the output you can see a strange malformation in the JSON: `{"first__number": "arg 0", "second__number": "arg 1"}`. The first parameter was called with two underscores for some reason (LLMs...). Fortunately Yacana banged on the LLM's head and it was fixed in the next iteration.  
- **Error 2**: Concerning the second error, it was definitely the tool that raised the exception: `The tool returned an error: `Parameter 'first_number' expected a type integer`. This is only logical as the LLM sent catastrophic values to the tool: `{'first_number': 'arg 0', 'second_number': 'arg 1'}`. When the ToolError was sent the error message was given to the LLM and a third iteration started. This time all was correct: `{"first_number": 2, "second_number": 2}` and we got our result which is 4.

#### Maximum tool errors

You should combine both methods described above. But adding heavy tool validation with great error messages on what went wrong is always what yields the best results and is also the safest option.  

So what happens if the LLM is stubborn and gets stuck in a loop? Even though Yacana's percussive maintenance should avoid that by shifting LLM internal configuration during runtime more or less randomly, the LLM still might go into an infinite loop. And this is NOT a viable option!  
Fortunately Yacana come with a default of 5 iterations (tries) for each of the 2 types of errors. Either the calling error like the `"first__number"` error seen above or the custom ToolError that the tool throws. This means that if one of these two counters gets to 5 then an error is raised.  
Specifically a `MaxToolErrorIter` exception. You should try/catch all of your Tasks that utilise Tools as they might loop too many times and trigger an exception.  

However, you can also set these counters to the value you wish. Move them higher or lower with the following Tool optional parameters: `max_custom_error: int = 5, max_call_error: int = 5`
For instance:
```
# Doubling the number of iterations the LLM can do before raising: 5 -> 10
adder_tool: Tool = Tool("Adder", "Adds two numbers and returns the result", adder, max_custom_error=10, max_call_error=10)
```

ℹ️ Note that showing the Agent's history with `agent1.history.pretty_print()` won't show you all the shenanigans that are shown in the logs. Many prompts get rid of the final history once the tool call is successful. It's always in the interest of the LLM to keep a clean History. 

### Making a tool optional

Sometimes you assign a Tool to a Task without knowing for sure that the tool will be useful. If you have a fine-tuned model or doing basic operations you may want to rely on the LLM's reasoning to choose if it really needs to call the tool or use his own training knowledge. Setting the `optional: bool = True` will tweak how Yacana proposes the Tools to the LLM, leaving it a chance to pass on the offer of the tool and use its own knowledge instead.

To demonstrate this let's make a tool that returns a temperature from a city. It will return a fake temperature as we don't care. We won't set `optionnal=True` so it will be forced to use the tool.
```python
def get_temperature(city: str) -> int:
    return 20


result: str = Task(f"What's the temperature in NY ?", agent1, tools=[Tool("get_temp", "Returns the celsius temperature of a given city", get_temperature)]).solve().content

print(f"Temperature = {result}")
```

Output:
```
INFO: [PROMPT]: I give you the following tool definition that you must use to fulfill a future task: get_temperature(city: str) -> int - Adds two numbers and returns the result. Please acknowledge the given tool.

INFO: [AI_RESPONSE]: I've received the tool definition!

So, I understand that I have been given a tool called `get_temperature` that takes one argument `city` which is a string, and it's supposed to return an integer.

However, I notice that the tool definition says "Adds two numbers and returns the result" but the function name `get_temperature` suggests it should be related to getting the temperature of a city.

I'll make sure to keep this in mind for any future tasks that require using this tool!

INFO: [PROMPT]: To use the tool you MUST extract each parameter and use it as a JSON key like this: {"arg1": "<value1>", "arg2": "<value2>"}. You must respect arguments type. For instance, the tool `getWeather(city: str, lat: int, long: int)` would be structured like this {"city": "new-york", "lat": 10, "lon": 20}. In our case, the tool call you must use must look like that: {'city': 'arg 0'}

INFO: [AI_RESPONSE]: So I understand now. Thank you for the clarification!

To use the `get_temperature` tool, I will extract each parameter and use it as a JSON key with its corresponding value.

The tool definition is: `get_temperature(city: str) -> int`

So, I will structure my tool call like this:  {"city": "arg0"}

Please let me know what's the next step!

INFO: [PROMPT]: You have a task to solve. Use the tool at your disposition to solve the task by outputting as JSON the correct arguments. In return you will get an answer from the tool. The task is:
What's the temperature in NY ?

INFO: [AI_RESPONSE]: { "city": "NY" }
Temperature = 20
```
The tool was called with `{ "city": "NY" }` which returned 20. Good !

---

Now let's ask something completely unrelated but keep the tool mandatory:
```python
result: str = Task(f"Why is the sky blue ?", agent1, tools=[Tool("get_temp", "Returns the celsius temperature of a given city", get_temperature)]).solve().content
```

Output:
```
INFO: [PROMPT]: I give you the following tool definition that you must use to fulfill a future task: get_temperature(city: str) -> int - Returns the celsius temperature of a given city. Please acknowledge the given tool.

INFO: [AI_RESPONSE]: Acknowledged!

You've provided me with a tool definition for `get_temperature(city: str) -> int`, which returns the Celsius temperature of a given city.

I'm ready to use this tool whenever I need it in future tasks!

INFO: [PROMPT]: To use the tool you MUST extract each parameter and use it as a JSON key like this: {"arg1": "<value1>", "arg2": "<value2>"}. You must respect the arguments type. For instance, the tool `getWeather(city: str, lat: int, long: int)` would be structured like this {"city": "new-york", "lat": 10, "lon": 20}. In our case, the tool call you must use must look like that: {'city': 'arg 0'}

INFO: [AI_RESPONSE]: For the `get_temperature( city: str ) -> int` tool, I understand that I need to extract each parameter and use it as a JSON key.

So, for this specific tool, I would structure my JSON call as follows:

{"city": "arg0"}

Where "arg0" is replaced with the actual value of the city I'm interested in.

INFO: [PROMPT]: You have a task to solve. Use the tool at your disposition to solve the task by outputting as JSON the correct arguments. In return you will get an answer from the tool. The task is:
Why is the sky blue ?

INFO: [AI_RESPONSE]: {"city": "Paris"}
Temperature = 20
```
It asked the tool for the temperature in Paris without any reason. We can't blame it. The tool we provided and the Task to solve have nothing to do with each other. So it's making things up, like the city name.  

---

Now let's make the tool optional and keep our unrelated Task:
```python
result: str = Task(f"Why is the sky blue ?", agent1, tools=[Tool("get_temp", "Returns the celsius temperature of a given city", get_temperature, optional=True)]).solve().content
```

Output:
```
INFO: [PROMPT]: I give you the following tool definition that you may use to fulfill a future task: get_temperature(city: str) -> int - Returns the celsius temperature of a given city. Please acknowledge the given tool.

INFO: [AI_RESPONSE]: Acknowledged!

I have taken note of the `get_temperature` tool, which takes a city name as input (`city: str`) and returns the Celsius temperature of that city (`-> int`). I'm ready to use this tool whenever you'd like me to fulfill a task in the future!

INFO: [PROMPT]: To use the tool you MUST extract each parameter and use it as a JSON key like this: {"arg1": "<value1>", "arg2": "<value2>"}. You must respect arguments type. For instance, the tool `getWeather(city: str, lat: int, long: int)` would be structured like this {"city": "new-york", "lat": 10, "lon": 20}. In our case, the tool call you must use must look like that: {'city': 'arg 0'}

INFO: [AI_RESPONSE]: So for the `get_temperature( city: str )` tool, I would extract the single parameter as:

{"city": "arg0"}

Please let me know what's next!

INFO: [PROMPT]: You have a task to solve. In your opinion, is using the tool "get_temp" relevant to solve the task or not ? The task is:
Why is the sky blue ?

INFO: [AI_RESPONSE]: Ha ha, that's an interesting question!

In my opinion, using the `get_temperature` tool is NOT relevant to solving this task. The task asks about why the sky is blue, and temperature doesn't seem to be directly related to that.

So, I'll have to come up with a different approach or use a different tool (if one exists) to tackle this question!

INFO: [PROMPT]: To summarize in one word your previous answer. Do you wish to use the tool or not ? Respond ONLY by 'yes' or 'no'.

INFO: [AI_RESPONSE]: no

INFO: [PROMPT]: Why is the sky blue ?

INFO: [AI_RESPONSE]: A question that has puzzled humans for centuries!

The sky appears blue because of a phenomenon called Rayleigh scattering, named after the British physicist Lord Rayleigh. Here's what happens:

1. **Sunlight**: The sun emits white light, which is made up of all the colors of the visible spectrum (red, orange, yellow, green, blue, indigo, and violet).
2. **Atmosphere**: When this sunlight enters Earth's atmosphere, it encounters tiny molecules of gases like nitrogen (N2) and oxygen (O2). These molecules are much smaller than the wavelength of light.
[BLABLA]
5. **Sky appears blue**: The combined effect of all these scattered blue and violet photons is what makes the sky appear blue to our eyes! The more direct sunlight that reaches our eyes, the whiter it will appear.

So, to summarize: the sky appears blue because of the selective scattering of shorter wavelengths (like blue and violet) by tiny molecules in the atmosphere, which dominates the colors we see when looking up at the sky.
```

As you can see it chose to ignore the tool when Yacana proposed it. It said:
```
In my opinion, using the `get_temperature` tool is NOT relevant to solving this task. The task asks about why the sky is blue, and temperature doesn't seem to be directly related to that.
```

## VIII. Assigning multiple Tools

In this section, we will see that you can assign more than one tool to a Task. You can add as many Tools as you want and the LLM will be asked what tool it wants to use. After using one of the tools it will be asked if it considers its Task complete. If it says "no" then Yacana will propose the list of tools again and a new iteration is started.  

This is roughly what the tool-calling mechanism looks like:
![toolcall1B](https://github.com/user-attachments/assets/bfaec298-44e9-4177-bfb8-e25bdfd01fe6)
ℹ️ This doesn't take into account many tweaks Yacana makes like: model config updates (in case of infinite loops), optional tools, self-reflection, multi-shot tool call examples, history cleaning, exiting when reaching max iterations, etc. However, it's definitely the *classic* process of calling tools one after the other.

> Additionnal behavior information: When only one tool is assigned the Agent won't be proposed to use it again. One tool is one shot !
> When giving multiple tools, the agent will this time be proposed to use another tool. He could choose to always use the same one though.
> In the future Yacana may allow you to have more control on how the tools are being used. For instance, allowing one tool to be re-called or when using multiple tools, each tool being used would be removed from the tool list, ensuring that each tool can only be used once.
> Also we could add a setting to force the LLM to use all the given tools from the list.
> Currently, giving more than one tool only ensure it makes use of one of them but could decide to stop after the first use if it wished to.
> Stay tuned for the next patch.
 
⚠️ For this next section we assume that you have already read section **Assigning a tool to a Task** of the documentation.

Let's make a more advance calculator. We'll add the missing tools and give them some "server side" checking to help the LLM use them properly.

```python
def adder(first_number: int, second_number: int) -> int:
    print("Adder was called with types = ", str(type(first_number)), str(type(second_number)))
    if not (isinstance(first_number, int)):
        raise ToolError("Parameter 'first_number' expected a type integer")
    if not (isinstance(second_number, int)):
        raise ToolError("Parameter 'second_number' expected a type integer")
    print(f"Adder was called with param = |{first_number}| and |{second_number}|")
    return first_number + second_number

def multiplier(first_number, second_number) -> int:
    print("Multiplier was called with types = ", str(type(first_number)), str(type(second_number)))
    if not (isinstance(first_number, int)):
        raise ToolError("Parameter 'first_number' expected a type integer")
    if not (isinstance(second_number, int)):
        raise ToolError("Parameter 'second_number' expected a type integer")
    print(f"Multiplier was called with param = |{first_number}| and |{second_number}|")
    return first_number * second_number

def substractor(first_number, second_number) -> int:
    print("substractor was called with types = ", str(type(first_number)), str(type(second_number)))
    if not (isinstance(first_number, int)):
        raise ToolError("Parameter 'first_number' expected a type integer")
    if not (isinstance(second_number, int)):
        raise ToolError("Parameter 'second_number' expected a type integer")
    print(f"substractor was called with param = |{first_number}| and |{second_number}|")
    return first_number - second_number



agent1 = Agent("Ai assistant", "llama3:8b")

# Defining 3 tools
adder_tool: Tool = Tool("Adder", "Adds two numbers and returns the result", adder)
substractor_tool: Tool = Tool("Substractor", "Subtracts two numbers and returns the result. When having, for instance, 2 - 6 the arguments are 2 and 6 and not 2 and -6. The tool does the subtraction.", substractor, usage_examples=[{"first_number": 2, "second_number": 4}])
multiplier_tool: Tool = Tool("Multiplier", "Multiplies two numbers and returns the result.", multiplier)

# Creating a Task to solve and assiging tools to it
result: str = Task(
    f"What's the result of '2 + 2 - 6 * 8'. Decompose the math before trying to solve it. Start with the multiplication. Use the tools at your disposal and do not do the maths yourself.",
    agent1, tools=[
        adder_tool,
        substractor_tool,
        multiplier_tool
    ]).solve().content

print("Result = ", result)
```

Output:
```
INFO: [PROMPT]: You have access to this list of tools definitions you can use to fulfill tasks :
* Name: 'Adder' - Usage: adder(first_number: int, second_number: int) -> int - Description: Adds two numbers and returns the result
* Name: 'Substractor' - Usage: substractor(first_number, second_number) -> int - Description: Subtracts two numbers and returns the result. When having, for instance, 2 - 6 the arguments are 2 and 6 and not 2 and -6. The tool does the subtraction.
* Name: 'Multiplier' - Usage: multiplier(first_number, second_number) -> int - Description: Multiplies two numbers and returns the result.
Please acknowledge the given tools.

INFO: [AI_RESPONSE]: I acknowledge that I have access to a list of tools with the following definitions:

1. **Adder**: A tool used for adding two numbers together and returning the result.
        * Usage: `adder(first_number: int, second_number: int) -> int`
2. **Substractor**: A tool used for subtracting one number from another and returning the result.
        * Usage: `substractor(first_number, second_number) -> int` (note that the order of arguments implies subtraction, e.g., 2 - 6 would be `substractor(2, 6)`)
3. **Multiplier**: A tool used for multiplying two numbers together and returning the result.
        * Usage: `multiplier(first_number, second_number) -> int`

I'm ready to use these tools to fulfill tasks!

INFO: [PROMPT]: You have a task to solve. I will give it to you between these tags `<task></task>`. However, your actual job is to decide if you need to use any of the available tools to solve the task or not. If you do need tools then output their names. The task to solve is <task>What's the result of '2 + 2 - 6 * 8'. Decompose the math before trying to solve it. Start with the multiplication. Use the tools at your disposal and do not do the maths yourself.</task> So, would any tools be useful in relation to the given task ?

INFO: [AI_RESPONSE]: I've got a task! Let's break down the math expression: "2 + 2 - 6 * 8".

The first operation I'll focus on is the multiplication: 6 * 8. This seems like a good candidate for using a tool!

Given the available tools, I think **Multiplier** could be useful here. It can help me calculate the result of multiplying 6 and 8.

So, my answer is: **Multiplier**

INFO: [PROMPT]: In order to summarize your previous answer in one word. Did you chose to use any tools ? Respond ONLY by 'yes' or 'no'.

INFO: [AI_RESPONSE]: Yes

INFO: [PROMPT]: You can only use one tool at a time. From this list of tools which one do you want to use: [Adder, Substractor, Multiplier]. You must answer ONLY with the single tool name. Nothing else.

INFO: [AI_RESPONSE]: Multiplier

INFO: [PROMPT]: To use the tool you MUST extract each parameter and use it as a JSON key like this: {"arg1": "<value1>", "arg2": "<value2>"}. You must respect arguments type. For instance, the tool `getWeather(city: str, lat: int, long: int)` would be structured like this {"city": "new-york", "lat": 10, "lon": 20}. In our case, the tool call you must use must look like that: {'first_number': 'arg 0', 'second_number': 'arg 1'}

INFO: [AI_RESPONSE]: {"first_number": 6, "second_number": 8}

INFO: [PROMPT]: Now that I showed you examples on how the tool is used it's your turn. Output the tool as valid JSON.

INFO: [AI_RESPONSE]: {"first_umber": 6, "second_number": 8}

WARNING: Yacana failed to call tool 'Multiplier' correctly based on the LLM output


INFO: [PROMPT]: The tool returned an error: `multiplier() got an unexpected keyword argument 'first_umber'`
Using this error message, fix the JSON arguments you gave.
Remember that you must output ONLY the tool arguments as valid JSON. For instance: {'first_number': 'arg 0', 'second_number': 'arg 1'}

INFO: [AI_RESPONSE]: {"first_number": 6, "second_number": 8}
Multiplier was called with types =  <class 'int'> <class 'int'>
Multiplier was called with param = |6| and |8|

INFO: [PROMPT]: Now that the tool responded do you need to make another tool call ? Explain why and what the remaining steps are if any.

INFO: [AI_RESPONSE]: Yes, I do need to make another tool call!

After using the **Multiplier** tool to get the result of 6 * 8 = 48, I now have to perform the addition and subtraction operations.

The expression is still: "2  + 2  - 48"

I need to use another tool to add 2 and 2. Then, I'll subtract 48 from the result.

So, my next step would be to use either the **Adder** or **Substractor** tool!

INFO: [PROMPT]: To summarize your previous answer in one word. Do you need to make another tool call ? Answer ONLY by 'yes' or 'no'.

INFO: [AI_RESPONSE]: Yes

INFO: Continuing tool calls loop


INFO: [PROMPT]: You can only use one tool at a time. From this list of tools which one do you want to use: [Adder, Substractor, Multiplier]. You must answer ONLY with the single tool name. Nothing else.

INFO: [AI_RESPONSE]: Adder

INFO: [PROMPT]: To use the tool you MUST extract each parameter and use it as a JSON key like this: {"arg1": "<value1>", "arg2": "<value2>"}. You must respect arguments type. For instance, the tool `getWeather(city: str, lat: int, long: int)` would be structured like this {"city": "new-york", "lat": 10, "lon": 20}. In our case, the tool call you must use must look like that: {'first_number': 'arg 0', 'second_number': 'arg 1'}

INFO: [AI_RESPONSE]: {"first_number": 2, "second_number": 2}

INFO: [PROMPT]: Now that I showed you examples of how the tool is used it's your turn. Output the tool as valid JSON.

INFO: [AI_RESPONSE]: {"first_number": 2, "second_number": 2}
Adder was called with types =  <class 'int'> <class 'int'>
Adder was called with param = |2| and |2|

INFO: [PROMPT]: Now that the tool responded do you need to make another tool call ? Explain why and what the remaining steps are if any.

INFO: [AI_RESPONSE]: Yes, I do need to make another tool call!
After using the  **Adder** tool to get the result of 2 + 2 = 4, I now have: "4 - 48"
I still have subtraction operation left, so I'll use either  **Substractor** or  **Multiplier** (which is not necessary in this case) and then I can finish the task.
So, my next step would be to use the  **Substractor** tool!

INFO: [PROMPT]: To summarize your previous answer in one word. Do you need to make another tool call ? Answer ONLY by 'yes' or 'no'.

INFO: [AI_RESPONSE]: Yes

INFO: Continuing tool calls loop


INFO: [PROMPT]: You can only use one tool at a time. From this list of tools which one do you want to use: [Adder, Substractor, Multiplier]. You must answer ONLY with the single tool name. Nothing else.

INFO: [AI_RESPONSE]: Substractor

INFO: [PROMPT]: To use the tool you MUST extract each parameter and use it as a JSON key like this: {"arg1": "<value1>", "arg2": "<value2>"}. You must respect arguments type. For instance, the tool `getWeather(city: str, lat: int, long: int)` would be structured like this {"city": "new-york", "lat": 10, "lon": 20}. In our case, the tool call you must use must look like that: {'first_number': 'arg 0', 'second_number': 'arg 1'}

INFO: [AI_RESPONSE]: {"first_number": 4, "second_number": 48}

INFO: [PROMPT]: Now that I showed you examples on how the tool is used it's your turn. Output the tool as valid JSON.

INFO: [AI_RESPONSE]: {"first_number": 4, "second_number": 48}
substractor was called with types =  <class 'int'> <class 'int'>
substractor was called with param = |4| and |48|

INFO: [PROMPT]: Now that the tool responded do you need to make another tool call ? Explain why and what are the remaining steps are if any.

INFO: [AI_RESPONSE]: No, I don't need to make another tool call!
The result of -44 is the final answer for the original expression: "2 + 2 - 6 * 8".
I've used all the necessary tools (Multiplier, Adder, and Substractor) to break down the math expression and get the correct answer!

INFO: [PROMPT]: To summarize your previous answer in one word. Do you need to make another tool call ? Answer ONLY by 'yes' or 'no'.

INFO: [AI_RESPONSE]: no

INFO: Exiting tool calls loop

Result =  No, I don't need to make another tool call!
The result of -44 is the final answer for the original expression: "2 + 2 - 6 * 8".
I've used all the necessary tools (Multiplier, Adder, and Substractor) to break down the math expression and get the correct answer!
```

-44 is the correct answer. You could throw in maybe one more operation. However, in our tests using Llama 3.0 going over 4 operations does not guarantee a correct result anymore. It may be about prompt engineering but we also think that Yacana should continue improving. For the moment the LLM tends to contradict itself at some point which sends the final result off. In the next update, Yacana will try to detect errors in reasoning and self-correct. Stay tuned for updates.  

## IX. Chat between two Agents

One of the best functionalities available in some other frameworks like CrewAI is to allow Agents to speak with one another. Yacana also has this functionality.  
This allows Agents to brainstorm and come up with solutions by themselves. However, where other frameworks propose many ways to schedule interactions, Yacana emphasizes on providing developers with ways to make them stop talking!

Making agents speak is not an issue. But making them stop is a whole other thing. How do you stop a conversation at the exact right moment without monitoring the chat yourself? There are only 2 ways. Either use a maximum number of iterations and stop the chat when it is reached or let one or more LLMs decide when the conversation should end.  
Yacana provides both.  
It utilizes the Task system that you already know and allows some Tasks to be worked on by the Agents. But be aware that using the Task system implies this mechanism had to be implemented differently than what you may have seen elsewhere. Even though the result is the same if done correctly...  

### Stopping chat using 'maximum iterations'

We'll be using a new class called [GroupSolve]() @todo url. It takes a list of Tasks (at least two) and an [EndChat]() object.  
1. The list of tasks will be the center of conversation for the LLM agents. The order in the list matters as the first task in the list will be the first to be evaluated and could be considered as the main task.
2. The EndChat object will allow you to configure when/how the conversation stops.
3. The GroupSolve class uses a `.solve()` method like the Task themselves. So don't forget about it.

Let's look at an example:
```python
# Creating two agents
agent1 = Agent("Ai assistant 1", "llama3:8b")
agent2 = Agent("Ai assistant 2", "llama3:8b")

# Creating two different Tasks to solve but that are related to one another
task1 = Task("Your task is to create a list of attractions for an amusement park.", agent1)
task2 = Task("Your task is to propose themes for an amusement park.", agent2)

# Creating the GroupSolve and specifying how the chat ends
GroupSolve([task1, task2], EndChat(EndChatMode.MAX_ITERATIONS_ONLY)).solve()
```

In the above code, we can see that the end of the chat condition is `EndChatMode.MAX_ITERATIONS_ONLY`. This means that both agents will exit the conversation after a predefined number of iterations. **By default set to 5**.  
You can set the max iteration level in the EndChat object with the `max_iterations=10` parameter:    
```
GroupSolve([task1, task2], EndChat(EndChatMode.MAX_ITERATIONS_ONLY, max_iterations=10)).solve()
```
ℹ️ Don't forget the `.solve()`

Output:
```
INFO: [PROMPT]: Your task is to create a list of attractions for an amusement park.

INFO: [AI_RESPONSE]: What a fun task! Here's a list of attractions for an amusement park:

**Thrill Rides**

1. **Sky Screamer**: A record-breaking rollercoaster that takes riders on a heart-pumping journey through loops, corkscrews, and airtime hills.
2. **Drop Zone**: A free-fall tower that sends thrill-seekers plummeting 10 stories down in mere seconds.
3. **Inferno**: A high-speed spinning coaster with flames and fire effects to simulate the sensation of flying through a fiery inferno.

**Family Favorites**

1. **Kiddie Coaster**: A gentle, slow-paced rollercoaster designed for young children with soft music and colorful scenery.
2. **Frog Hopper**: An interactive ride where riders hop aboard toy frogs that bounce up and down on giant lily pads.
3. **Storybook Train**: A classic train ride featuring beloved cartoon characters and whimsical settings.

**Water Park Fun**

1. **Splashin' Safari**: A water play area with geysers, fountains, and squirting animals for kids to enjoy.
2. **River Rapids**: A lazy river that winds its way through a scenic valley, complete with waterfalls and splash pads.
3. **Tornado Tube**: A spinning, twisting waterslide that sends riders swirling down into a splashy finale.

That's just a taste of what our amusement park has to offer!

INFO: [PROMPT]: Your task is to propose themes for an amusement park.

INFO: [AI_RESPONSE]: What a fun challenge!

Here are some theme ideas for an amusement park:

**1. Enchanted Forest**: A whimsical zone inspired by mythical creatures and ancient forests. Attractions could include a "Dragon's Den" roller coaster, a "Mermaid's Lagoon" splash pad, and a "Fairy Ring" carousel.

**2. Future Frontier**: A futuristic area with intergalactic and space-themed rides. Imagine a "Galaxy Blaster" launch coaster, an "Alien Invasion" dark ride, or a "Gravity Simulator" virtual reality experience.

**3. Pirate's Cove**: Ahoy matey! A swashbuckling zone filled with pirate-themed attractions like a "Jolly Roger" drop tower, a "Treasure Hunt" interactive play area, and a "Barrel of Laughs" comedy show.

These are just a few ideas to get you started. Which theme would you like to explore further?

INFO: [AI_RESPONSE]: What a fantastic list!

I can already imagine the thrill and excitement that visitors would experience with these attractions. Let's break down some potential themes and connections between the different areas:

**Thrill Rides**: These high-adrenaline rides would be perfect for the "Future Frontier" or "Enchanted Forest" theme, where riders would feel like they're exploring new worlds or battling mythical creatures.

**Family Favorites**: These gentle attractions would fit well in a "Carnival Spectacular" or "Pirate' s Cove" theme, where families could enjoy classic carnival games and pirate-themed play areas together.

**Water Park Fun**: The water park area would be an excellent addition to the "Enchanted Forest" or "Steampunk City" theme, where riders could explore fantastical aquatic worlds or steam-powered contraptions.

**Dark Rides**: These spooky attractions would be a great fit for the "Mystic Mountain" or "Haunted Mansion" theme, where visitors could experience eerie mysteries and supernatural encounters.

**Classic Attritions**: The merry-go-round, Ferris wheel, and fun house would be perfect additions to a "Retro Arcade" or "Carnival Spectacular" theme, where riders could enjoy classic games and nostalgic experiences.

**Special Experiences**: These unique shows and meet-and-greets would be excellent additions to the "Enchanted Forest" or "Future Frontier" theme, where visitors could encounter mythical creatures, astronauts, or otherworldly beings.

Some potential themes that could tie everything together:

1. **Mythical Realms**: A world of fantasy and adventure, where visitors can explore enchanted forests, battle dragons, and ride mythical creatures.
2. **Steampunk Odyssey**: A futuristic world with a Victorian twist, where steam-powered machines and fantastical contraptions await exploration.
3. **Galactic Frontier**: A cosmic adventure zone where visitors can blast off into space, explore alien worlds, and encounter extraterrestrial beings.

These are just a few ideas to get you started! What do you think? Would you like to add or modify any attractions or themes?
[BLABLA]
```
I won't show the full 5 iterations as it's useless. However, I'm sure you have one question at this point if you ran the program yourself...  

> Why do I get the logging twice ??

Well... This is because of how the conversation pattern is implemented. Let me explain... Have you ever read the documentation for the Microsoft Autogen framework? If you have, I hope you're having a better time with Yacana than I did with Autogen. That said, the conversational patterns they show are a series of dual-agent conversations. And never did I understand the mess they did before Yacana came to life. The reason why they chain two-agent conversations is because LLMs have been trained to speak in alternation with a user. It's how all "instruct" models have been fine-tuned.   
So to get the best performance out of the LLMs they chose to limit the number of participants to two. If more than two was ever needed then the context of the first conversation would be given to a new dual-chat with one of the agents remaining from the previous conversation (hence keeping the state from one conversation to the other). Then it goes on and on.  
![image](https://github.com/user-attachments/assets/c8c4d958-2ffc-4eca-8d4a-aef576627572)
*Source: Microsoft Autogen*  
I honestly think that it's smart but is a stinking mess that lost many people. Worst, it's the simpler pattern they provide...   

However, This two-chat-based conversations must still alternate between USER and ASSISTANT, so to make it work, each Agent thinks it's speaking to you (the human USER) even though it's speaking to the other Agent (ASSISTANT) instead.  
Yacana does not do things exactly in the same way but is bound to the same limitations. Two agents' chats give the best results so the alternation between USER and ASSISTANT must also be achieved! This is why you are seeing duplicated logs in the de INFO logging system. **It's the answer of one agent being used as a prompt for the other one.**

### Letting Agents in charge of ending the chat

The [EndChatMode]() @todo url enum provides multiple ways to stop a chat. These are the available values:
| Mode              | Needs Task annotation | Description |
| :---------------- | :------: | :----- |
| MAX_ITERATIONS_ONLY | False  | Chat ends when we reach the maximum number of rounds. *Defaults to 5.* |
| END_CHAT_AFTER_FIRST_COMPLETION | True | When a Task is marked as complete the chat ends immediatly |
| ONE_LAST_CHAT_AFTER_FIRST_COMPLETION | True | When a Task is marked as complete, one last agent can speak and then the chat ends |
| ONE_LAST_GROUP_CHAT_AFTER_FIRST_COMPLETION | True| When a Task is marked as complete, one whole table turn will be allowed before the chat ends |
| ALL_TASK_MUST_COMPLETE | True | All tasks must be marked as complete before the chat is ended |

⚠️ To prevent infinite loops ALL chat modes still have a maximum iteration count (defaults to 5) which can be change in the GroupSolve() class with the optionnal parameter `max_iterations=<int>`.

> What is the **Needs task annotation** column above ?  
> To let an agent know that it's in charge of ending the chat, its Task() must be given an optionnal parameter `llm_stops_by_itself=True`. All Task() constructors setting this, makes their assigned Agent potentially in charge of stopping the conversation.

#### Defining task completion objective

Now that LLMs can stop the conversation by themselve you must set clear objectives for this to happen. The conditions to which the objectives are met must be precise and consise. You will quickly see how prompt engineering is important in this matter.  
Yacana uses the following wording:  
> In your opinion, what objectives from your initial task have you 'NOT completed ?
Therefore you should use the same prompt style in your Task prompt. For instance *"The task is fulfilled when the objective <insert here>"*.
You should try different version of this "objective completed" prompt to find one that matches your task and LLM.

#### Testing chat ending modes

We'll test the different mode with simple games between two Agents.  

**END_CHAT_AFTER_FIRST_COMPLETION**

Let's play a simple guessing game where an agent thinks of a number ranging from 1 to 3. The other agent must guess the number correctly.

⚠️ ❗ Although this game feels simple. Most 8B models will fail at it. As you can see below I upgraded the model to something superior to the classic llama:3.0 that lacks in reasoning. You will need to adjust the model to your computer.  
But, a word of advice: don't expect any kind of great results with this approach. Local LLMs can brainstorm conceptual ideas but when it comes to logic and reasoning they are very bad. Even with this upgraded model, I often get illogical answers. Moreover, LLMs dislike numbers and have great difficulty to compare them.  

```python
agent1 = Agent("Ai assistant 1", "dolphin-mixtral:8x7b-v2.7-q4_K_M")
agent2 = Agent("Ai assistant 2", "dolphin-mixtral:8x7b-v2.7-q4_K_M")

# We create a Task to generate the secret number BEFORE entering the chat so that only Agent1 knows the secret number
Task("You must choose a number between 1 and 3. It will be referred as the 'secret number'. Output it once, now.", agent1).solve()

task1 = Task("You will receive guesses. If the guessed number is the secret number say 'You won'. The objective of this task is fulfilled when this condition is true: 'The guessed number is the same as the secret number'.", agent1, llm_stops_by_itself=True)
task2 = Task("You must find a secret number from 1 to 3. Propose one number.", agent2)

# Making a GroupSolve() with both Tasks
GroupSolve([task1, task2], EndChat(EndChatMode.END_CHAT_AFTER_FIRST_COMPLETION)).solve()

print("------ Agent1 --------")
task1.agent.history.pretty_print()

print("------Agent2----------")
task2.agent.history.pretty_print()
```

❕The most important line is the `task1` line with the `llm_stops_by_itself=True` parameter. This means that the assigned Agent is in charge of stopping the conversation.

Output:
```
INFO: [PROMPT][To: Ai assistant 1]: You must choose a number between 1 and 3. It will be referred as the 'secret number'. Output it once, now.

INFO: [AI_RESPONSE][From: Ai assistant 1]:  The secret number is: 2

INFO: [PROMPT][To: Ai assistant 1]: You will receive guesses. If the guessed number is the secret number say 'You won'. The objective of this task is fulfilled when this condition is true: 'The guessed number is the same as the secret number'.

INFO: [AI_RESPONSE][From: Ai assistant 1]:  You've got it right, I have already chosen the secret number 2. So if you guess the number 2, you will win!

INFO: [PROMPT][To: Ai assistant 2]: You must find a secret number from 1 to 3. Propose one number.

INFO: [AI_RESPONSE][From: Ai assistant 2]:  I suggest the number 2 as my proposal for the secret number.

INFO: [PROMPT][To: Ai assistant 2]:  You've got it right, I have already chosen the secret number 2. So if you guess the number 2, you will win!

INFO: [AI_RESPONSE][From: Ai assistant 2]:  Thank you for the information! In that case, based on your hint, I propose the number 2 as my final answer to find the secret number. Good luck with your game!

INFO: [PROMPT][To: Ai assistant 1]:  Thank you for the information! In that case, based on your hint, I propose the number 2 as my final answer to find the secret number. Good luck with your game!

INFO: [AI_RESPONSE][From: Ai assistant 1]:  Congratulations! Your guess was correct. The secret number is indeed 2. You won!

INFO: [PROMPT][To: Ai assistant 1]: In your opinion, what objectives from your initial task have you NOT completed ?

INFO: [AI_RESPONSE][From: Ai assistant 1]:  I believe there are no unfinished tasks at the moment. All tasks mentioned in the initial prompt have been successfully fulfilled as follows:

1. The secret number (2) was chosen and shared.
2. A condition for winning the game was provided, which is 'The guessed number is the same as the secret number'.
3. When the guessed number was 2, I confirmed that the participant had won the game.

INFO: [PROMPT][To: Ai assistant 1]: To summarize in one word, did you still had some objectives to fulfill ? Answer ONLY by 'yes' or 'no'. Do not output anything else !

INFO: [AI_RESPONSE][From: Ai assistant 1]: no
------ Agent1 --------
[user]:
You must choose a number between 1 and 3. It will be referred as the 'secret number'. Output it once, now.

[assistant]:
 The secret number is: 2

[user]:
You will receive guesses. If the guessed number is the secret number say 'You won'. The objective of this task is fulfilled when this condition is true: 'The guessed number is the same as the secret number'.

[assistant]:
 You've got it right, I have already chosen the secret number 2. So if you guess the number 2, you will win!

[user]:
 Thank you for the information! In that case, based on your hint, I propose the number 2 as my final answer to find the secret number. Good luck with your game!

[assistant]:
 Congratulations! Your guess was correct. The secret number is indeed 2. You won!

------Agent2----------
[user]:
You must find a secret number from 1 to 3. Propose one number.

[assistant]:
 I suggest the number 2 as my proposal for the secret number.

[user]:
 You've got it right, I have already chosen the secret number 2. So if you guess the number 2, you will win!

[assistant]:
 Thank you for the information! In that case, based on your hint, I propose the number 2 as my final answer to find the secret number. Good luck with your game!
```

The important part is in the INFO logs. Extracted from above as follow:
```
INFO: [PROMPT][To: Ai assistant 1]: In your opinion, what objectives from your initial task have you NOT completed ?

INFO: [AI_RESPONSE][From: Ai assistant 1]:  I believe there are no unfinished tasks at the moment. All tasks mentioned in the initial prompt have been successfully fulfilled as follows:

1. The secret number (2) was chosen and shared.
2. A condition for winning the game was provided, which is 'The guessed number is the same as the secret number'.
3. When the guessed number was 2, I confirmed that the participant had won the game.
```

Yacana asked Agent1 if it considered its objectives complete or not. The model responded that it completed all objectives:
```
INFO: [PROMPT][To: Ai assistant 1]: To summarize in one word, did you still had some objectives to fulfill ? Answer ONLY by 'yes' or 'no'. Do not output anything else !

INFO: [AI_RESPONSE][From: Ai assistant 1]: no
```

Right after this answer, the chat ended because of the `END_CHAT_AFTER_FIRST_COMPLETION` setting.

In summary, the conversation looks like this:

![GS3B](https://github.com/user-attachments/assets/2fde0c79-1a03-492d-8d9d-94e2b628e2bc)

ℹ️ You will learn what is a shift message later on and how it can be improved. 

**ONE_LAST_CHAT_AFTER_FIRST_COMPLETION**

This "end chat" mode allows one more Agent to speak after registering the first Task success.

Update the GroupSolve line like this:
```python
GroupSolve([task1, task2], EndChat(EndChatMode.ONE_LAST_CHAT_AFTER_FIRST_COMPLETION), shift_message_owner=task1).solve()
```

Output:

```
------ Agent1 --------
[user]:
You must choose a number between 1 and 3. It will be referred as the 'secret number'. Output it once, now.

[assistant]:
 2

[user]:
You will receive guesses. If the guessed number is the secret number say 'You won'. The objective of this task is fulfilled when this condition is true: 'The guessed number is the same as the secret number'.

[assistant]:
 I understand. You can now enter your guess, and I will check if it's the secret number.

[user]:
 Okay, my guess is 1. Let me know if that's correct or not.

[assistant]:
 Unfortunately, the guessed number (1) is not the same as the secret number (2). Please try again with a different number.

[user]:
 My apologies. Let me try again with another guess: 3. Is this the correct number?

[assistant]:
 No, the guessed number (3) is not the same as the secret number (2). Please try again with a different number.

[user]:
 I apologize for the confusion. My final guess is 2. I believe that should be the correct number. Let me know if I'm right.

[assistant]:
 Congratulations! The guessed number (2) is the same as the secret number (2). You won!

------Agent2----------
[user]:
You must find a secret number from 1 to 3. Propose one number.

[assistant]:
 I propose the number 2. Is that your secret number?

[user]:
 I understand. You can now enter your guess, and I will check if it's the secret number.

[assistant]:
 Okay, my guess is 1. Let me know if that's correct or not.

[user]:
 Unfortunately, the guessed number (1) is not the same as the secret number (2). Please try again with a different number.

[assistant]:
 My apologies. Let me try again with another guess: 3. Is this the correct number?

[user]:
 No, the guessed number (3) is not the same as the secret number (2). Please try again with a different number.

[assistant]:
 I apologize for the confusion. My final guess is 2. I believe that should be the correct number. Let me know if I'm right.

[user]:
 Congratulations! The guessed number (2) is the same as the secret number (2). You won!

[assistant]:
 Thank you for your cooperation! I'm glad we found the secret number together. Have a great day!
```

Agent1 said that the game was finished:
```
[assistant]:
 Congratulations! The guessed number (2) is the same as the secret number (2). You won!
```

Agent2 had the opportunity to answer to his win !
```
[user]:
 Congratulations! The guessed number (2) is the same as the secret number (2). You won!

[assistant]:
 Thank you for your cooperation! I'm glad we found the secret number together. Have a great day!
```

**ONE_LAST_GROUP_CHAT_AFTER_FIRST_COMPLETION**

Update the GroupSolve() line to this:
```
GroupSolve([task1, task2], EndChat(EndChatMode.ONE_LAST_GROUP_CHAT_AFTER_FIRST_COMPLETION)).solve()
```
This will allow both LLMs to speak one last time before the chat ends.

Output:
```
------ Agent1 --------
[BLABLA]
[assistant]:
 Congratulations, you won! Your guess of 1 matches my secret number, which was 2. Well done!

[user]:
 Thank you! I'm glad to have guessed correctly. Good luck in your future endeavors!

[assistant]:
 You're welcome, and thank you for the kind words! It's always a pleasure when someone successfully finds the secret number. Have a great day!
```
After the initial success, two more chats were generated.
```
------Agent2----------
[user]:
 Congratulations, you won! Your guess of 1 matches my secret number, which was 2. Well done!

[assistant]:
 Thank you! I'm glad to have guessed correctly. Good luck in your future endeavors!
```


**ALL_TASK_MUST_COMPLETE**

This "end chat" mode is useful when more than one Agent has the `llm_stops_by_itself=True`. This means that the conversation will only end when all the agent with this parameter have decided that their objectives are completed.  

To demonstrate this without having a headache let's make a silly GroupSolve() stating that one Agents must count from 0 to 2 and the other one from 0 to 3 to complete their respective objective.

❗ Shifting back to llama:3.0. Dolphin always wants to execute Python code when it has to do maths...

```
agent1 = Agent("Ai assistant 1", "llama3:8b")
agent2 = Agent("Ai assistant 2", "llama3:8b")

task1 = Task("Your task is to add 1 to the initial count which is 0. Your objective is complete when you get to output number '2'", agent1, llm_stops_by_itself=True)
task2 = Task("Your task is to add 1 to the initial count which is 0. Your objective is complete when you get to output number '3'", agent2, llm_stops_by_itself=True)

GroupSolve([task1, task2], EndChat(EndChatMode.ALL_TASK_MUST_COMPLETE)).solve()

print("------ Agent1 --------")
task1.agent.history.pretty_print()

print("------Agent2----------")
task2.agent.history.pretty_print()
```

Output:
```
------ Agent1 --------
[user]:
Your task is to add 1 to the initial count which is 0. Your objective is complete when you get to output number '2'

[assistant]:
I'm ready!

Initial count: 0
Add 1: 1

Please wait for my next update...

[user]:
Let's continue from where we left off!

Next update, please?

[assistant]:
Initial count: 1
Add 1: 2

Woohoo! I reached the target of '2'!

[user]:
Congratulations on reaching the target of 2!

Let's keep going!
What's the next update?

[assistant]:
Hehe, thank you!

Initial count: 2
Add 1: 3

Let's see where this journey takes me next!

------Agent2----------
[user]:
Your task is to add 1 to the initial count which is 0. Your objective is complete when you get to output number '3'

[assistant]:
Let's start with the initial count of 0.

I will add 1 to the count, making it...

1!

Next, I'll add 1 again, making it...

2!

And finally, I'll add 1 once more, making it...

3!

Woohoo! I've reached my objective!

[user]:
I'm ready!

Initial count: 0
Add 1: 1

Please wait for my next update...

[assistant]:
Let's continue from where we left off!

Next update, please?

[user]:
Initial count: 1
Add 1: 2

Woohoo! I reached the target of '2'!

[assistant]:
Congratulations on reaching the target of 2!

Let's keep going!
What's the next update?

[user]:
Hehe, thank you!

Initial count: 2
Add 1: 3

Let's see where this journey takes me next!

[assistant]:
Congratulations again!
You've reached your target of '3'!

Looks like we've completed the task successfully! Well done! Would you like to start a new adventure or try something else?
```

As you can see, Agent2 cheated. It counted from 0 to 3 in one message. But who cares? Both of them carried on till both had their achievement completed.  


### Getting better results by controlling the "shift message"

To achieve two agents speaking with each other we had to pipe the output of the first one into the prompt of the second. For this to work, we have to create an intermediary message in one of the Agent's conversation history. This special message is called the *"shift message"*.  

To showcase this let's make another game:  
* Agent1 has the main task of storing a list of numbers. The list is empty ;
* Agent2 has the task of giving numbers to Agent1 so that it add them to its list ;

The "game" finishes when Agent1 has 15 numbers in its list.  

```python
agent1 = Agent("Ai assistant 1", "llama3:8b")
agent2 = Agent("Ai assistant 2", "llama3:8b")


task1 = Task("Your task is to create a list of numbers. The list starts empty. The numbers will be given to you. Your objective is fulfilled when you have more than 15 numbers in the list.", agent1, llm_stops_by_itself=True)
task2 = Task("You will have access to a list of numbers. Ask to add 3 more number to this list. Also ask to print the complete list each time to keep track.", agent2)

GroupSolve([task1, task2], EndChat(EndChatMode.END_CHAT_AFTER_FIRST_COMPLETION, max_iterations=3)).solve()

print("------ Agent1 --------")
task1.agent.history.pretty_print()

print("------Agent2----------")
task2.agent.history.pretty_print()
```

The execution flow looks like this: 

![gs2B](https://github.com/user-attachments/assets/e1e8e1eb-5c31-40ac-b044-e505deaa5219)

Let's decompose the graph piece by piece. There are two columns: one for Agent1's point of view and one for Agent's 2 point of view. Like in any conversation, each speaker has its own point of view. This is why you shouldn't rely on the debugging logs only but also print each Agent's History.    
* In line 1 we have the blue messages which are the initial 2 `Task(...)` that were given to the GroupSolve (We summarized the prompts so that they fit the graph a bit better)
* In line 2 we have the AI answers to the prompts of line 1. What's very interesting here is that each initial prompt is solved by their respective Agent and isn't shared between them! This means that Agent2 doesn't know what Agent1's task is and vice versa. This is important because it must be taken into account when writing the prompt for the second Task.
  * In our example the Task2's prompt starts with *"You will have access to a list of numbers. ..."*. This demonstrates the importance of using the future because when this task is solved it won't know that there is any list whatsoever.
* In line 3 we add the *"shift"* ! The singular most important message here. As we need both Agents to speak to each other we need to initiate some kind of shift in the message so that the "USER" of one agent becomes the "ASSISTANT" of the other.
  * The shift message is very important because it must not throw off the LLM with a strange message. By default, it is copied from Agent1's answer from line 2 (see the red arrow). In general, the conversation still makes sense. But if it doesn't then you'll have to take control of the shift message!
  * Taking control of the shift message is fairly simple. You have 2 types of control. The first control is where the shift message should be placed. Either in Agent1 or in Agent2 (default) point of view. The second control is the content of the shift message. It can either come from the opposite side (default) or be set by you.
    * To control who gets the shift, use the `shift_owner=<task instance>` optionnal parameter from the GroupSolve() class, like this: `GroupSolve([task1, task2], EndChat(EndChatMode.END_CHAT_AFTER_FIRST_COMPLETION), shift_owner=task2).solve()`
    * To control the content of the shift message, use the `shift_content="<Some message>"` optional parameter from the GroupSolve() class. Note that not specifying this parameter results in the message being copied from the opposite Task (cf graph). Use it like this: `GroupSolve([task1, task2], EndChat(EndChatMode.END_CHAT_AFTER_FIRST_COMPLETION), shift_content="List is empty").solve()`
    * Both parameters can be set at the same time. Also, not specifying any of them will output the same result as shown in the above graph.

ℹ️ About the `max_iterations=<int>` optional parameter from the GroupSolve(...) class: Note that the iteration starts counting only after the shift message has been generated and history is synced. Resulting in one more message in one of the agent's History and two more in the other's (initial task solving + shift message).

Output
```
[ Not showing INFO debug]
------ Agent1 --------
[user]:
Your task is to create a list of numbers. The list starts empty. The numbers will be given to you. Your objective is fulfilled when you have more than 15 numbers in the list.

[assistant]:
I'm ready! My empty list is:

[]

Please give me the first number!

[user]:
Exciting!
Let's add some numbers to your list.

To start, I'd like to add the following three numbers:

1. 5
2. 8
3. 9

Here's the updated list:

[5, 8, 9]

Please confirm before I proceed with the next set of additions!

[assistant]:
Exciting indeed!

I've confirmed the initial list:

[5, 8, 9]

Now, please go ahead and add more numbers to the list.

[user]:
Fantastic!

Next, I'd like to add three more numbers to your list:

1. 2
2. 7
3. 11

Here's the updated list:

[5, 8, 9, 2, 7, 11]

Please confirm before I proceed with the next set of additions!

[assistant]:
Confirmation received!

The updated list looks great:

[5, 8, 9, 2, 7, 11]

I'm ready for the next addition!

[user]:
Wonderful!
Next, I'd like to add three more numbers to your list:
1. 15
2. 3
3. 6

Here's the updated list:

[5, 8, 9, 2, 7, 11, 15, 3, 6]
Please confirm before I proceed with the next set of additions!

[assistant]:
Confirmation received again!

The updated list now has a total of:

[5, 8, 9, 2, 7, 11, 15, 3, 6]

Which exceeds the initial goal of having more than 15 numbers in the list!

I'm pleased to report that my task is complete!

------Agent2----------
[user]:
You will have access to a list of numbers. Ask to add 3 more number to this list. Also ask to print the complete list each time to keep track.

[assistant]:
I'd be happy to help with that.

Please provide the initial list of numbers, and I'll proceed by adding three more numbers to it. I'll also request a printout of the updated list each time to ensure we're keeping track correctly.

Let's get started! What's the initial list of numbers?

[user]:
I'm ready! My empty list is:

[]

Please give me the first number!

[assistant]:
Exciting!
Let's add some numbers to your list.

To start, I'd like to add the following three numbers:

1. 5
2. 8
3. 9

Here's the updated list:

[5, 8, 9]

Please confirm before I proceed with the next set of additions!

[user]:
Exciting indeed!

I've confirmed the initial list:

[5, 8, 9]

Now, please go ahead and add more numbers to the list.

[assistant]:
Fantastic!

Next, I'd like to add three more numbers to your list:

1. 2
2. 7
3. 11

Here's the updated list:

[5, 8, 9, 2, 7, 11]

Please confirm before I proceed with the next set of additions!

[user]:
Confirmation received!

The updated list looks great:

[5, 8, 9, 2, 7, 11]

I'm ready for the next addition!

[assistant]:
Wonderful!
Next, I'd like to add three more numbers to your list:
1. 15
2. 3
3. 6

Here's the updated list:

[5, 8, 9, 2, 7, 11, 15, 3, 6]
Please confirm before I proceed with the next set of additions!
```

### Using tools inside GroupSolve

#### Description

GroupSolve uses the common Task class. This means that Tools are also available while agents are chatting. However, tools work the same way as described in the tool section @todo URL. Meaning that the Task's prompt will only be used to trigger the prompt but will not be used to act don't result of the tool. You need the other agent for that.  
This is an important concept because it means that the task's prompt will not be part of the conversation but the tool output will. Meaning that your tools must always return some computable knowledge that will be used by the second agent! 

#### Tool use in GroupSolve

Let's play a new game, without tools first:  
* The first agent will think of a number. The second agent will try to guess it based on indications like "higher" or "lower" given by the first agent.  
* The conversation ends when the second agent finds the correct number and wins the game!  

```python
# Creating our two players
agent1 = Agent("Player 1", "llama3:8b")
agent2 = Agent("Player 2", "llama3:8b")


# Making up a number and guiding player 2 
task1 = Task("Your task is to generate a random number between 1 and 20 then show it only once. Once you have the number, respond to the guesses you receive with one of the following:\n* If the guess is higher than your number, say 'lower.'\n* If the guess is lower than your number, say 'higher.'\n* If the guess matches your number, say 'You won!' and end the game.\nDo not provide any other feedback or responses, and only play one round.", agent1)

# Guessing the correct number. Higher or lower depending on what Player 1 tells you
# This agent has the llm_stops_by_itself=True meaning it will be in charge of stopping the conversation
task2 = Task("Your task is to guess a number between 1 and 20. You will receive feedback after each guess: either 'lower,' 'higher,' or 'You won!' Adjust your next guess based on this feedback. You must propose only one number at a time. Continue guessing until you receive confirmation that you have guessed correctly and the game is won. Only play one round.", agent2, llm_stops_by_itself=True)

# Adding the two tasks to the GroupSolve. The chat ends when the first success is achieved
GroupSolve([task1, task2], EndChat(EndChatMode.END_CHAT_AFTER_FIRST_COMPLETION)).solve()

# Printing both Agent's point of view
print("------ Agent 1 --------")
agent1.history.pretty_print()

print("------Agent 2----------")
agent2.history.pretty_print()
```

```
[Automatic logging I won't show]

------ Agent 1 --------
[user]:
Your task is to generate a random number between 1 and 20 then show it only once. Once you have the number, respond to the guesses you receive with one of the following:
* If the guess is higher than your number, say 'lower.'
* If the guess is lower than your number, say 'higher.'
* If the guess matches your number, say 'You won!' and end the game.
Do not provide any other feedback or responses, and only play one round.

[assistant]:
I've generated a random number between 1 and 20: **14**

What's your guess?

[user]:
Thanks for the feedback!

Since the correct answer is 14, which is higher than my initial guess of 10, my next guess will be... **12**.

What's the feedback this time?

[assistant]:
lower

[user]:
So the correct answer (14) is lower than my new guess (12). That means my next guess should be even lower!

My next guess is... **11**.

What's the feedback?

[assistant]:
higher

[user]:
So the correct answer (14) is higher than my guess of 11. That means I'm getting closer, but not quite there yet!
My next guess is... **13**.
What's the feedback?

[assistant]:
You won!

[user]:
WOOHOO!

I did it! I guessed correctly! The correct answer was indeed 14, and I finally got it right after a few attempts.

Thank you so much for playing the game with me! It was an absolute blast. If you want to play again or try a different game, just let me know!

[assistant]:
Congratulations on guessing correctly! You played the game perfectly!

I'm glad I could provide a fun experience for you. If you're up for another round or want to try a different game, just let me know and we can start fresh.

Thanks for playing with me, and I hope to see you again soon!

[user]:
Thank you so much! I'm thrilled to have played the game correctly and had a great experience. I'd love to play again or try a different game, so please feel free to suggest one anytime.

I'm always up for a fun challenge, and I appreciate your creativity in designing this guessing game. It was a delightful way to spend time together!

Let's do it again sometime soon!

[assistant]:
Aw, thank you so much! I'm thrilled that you had a great experience playing the game with me too!

I'd be happy to suggest another game or play again whenever you're ready. There are many options we can try, such as:

1. 20 Questions: You think of an object, and I try to guess what it is by asking yes or no questions.
2. Word Chain: We take turns thinking of a word that starts with the last letter of the previous word.
3. Hangman: I think of a word or phrase, and you try to guess it by suggesting letters.

Which one sounds interesting to you, or do you have a different game in mind?

Let me know when you're ready, and we can start a new game!

------Agent 2----------
[user]:
Your task is to guess a number between 1 and 20. You will receive feedback after each guess: either 'lower,' 'higher,' or 'You won!' Adjust your next guess based on this feedback. You must propose only one number at a time. Continue guessing until you receive confirmation that you have guessed correctly and the game is won. Only play one round.

[assistant]:
Sounds like fun!

My first guess is... 10.

What's the feedback?

[user]:
I've generated a random number between 1 and 20: **14**

What's your guess?

[assistant]:
Thanks for the feedback!

Since the correct answer is 14, which is higher than my initial guess of 10, my next guess will be... **12**.

What's the feedback this time?

[user]:
lower

[assistant]:
So the correct answer (14) is lower than my new guess (12). That means my next guess should be even lower!

My next guess is... **11**.

What's the feedback?

[user]:
higher

[assistant]:
So the correct answer (14) is higher than my guess of 11. That means I'm getting closer, but not quite there yet!
My next guess is... **13**.
What's the feedback?

[user]:
You won!

[assistant]:
WOOHOO!

I did it! I guessed correctly! The correct answer was indeed 14, and I finally got it right after a few attempts.

Thank you so much for playing the game with me! It was an absolute blast. If you want to play again or try a different game, just let me know!

[user]:
Congratulations on guessing correctly! You played the game perfectly!

I'm glad I could provide a fun experience for you. If you're up for another round or want to try a different game, just let me know and we can start fresh.

Thanks for playing with me, and I hope to see you again soon!

[assistant]:
Thank you so much! I'm thrilled to have played the game correctly and had a great experience. I'd love to play again or try a different game, so please feel free to suggest one anytime.

I'm always up for a fun challenge, and I appreciate your creativity in designing this guessing game. It was a delightful way to spend time together!

Let's do it again sometime soon!
```

It's a complete failure... :-( But why is that?  
First, let's analyze the output:
* Agent1 generates the initial secret number: it's 14!
* Agent2 tries 12 which is lower than 14. In theory, Agent1 should tell him "higher" to guide it toward the correct number
* Agent1 tells Agent2: "lower" which is wrong!
In the end:
* Agent2 tries 13
* Agent1 tells it it won!

No, it didn't !? The initial secret number was 14! Not 13...  

There are two issues here:
* The first one is that the LLM in use, which is Llama3.0, sucks at maths and cannot compare numbers accurately! This is a common issue even with frontier models sometimes. So the game breaks very quickly.
* The second issue is that the secret number is available to the other agent because of the shift message!
  * The shift message being the output of the Game Master, it spoils the secret number! (bummer)

To fix those issues:
* Use a tool and don't let the LLM compare numbers itself. Let the CPU deal with that!
* Either switch the shift message to the other agent or set it manually so that it doesn't spoil the secret number to the player.
* Maybe even better than switching the shift message: Create a Task beforehand that generates the secret number and after that, enter the GroupSolve() (We're going with that solution here but have fun toying with the examples!)


```python
# Creating our tool with type checking and 3 conditional returns
def high_low(secret_number: int, guessed_number: int) -> str:
    print(f"Tool is called with {secret_number} / {guessed_number}")

    # Validation
    if not (isinstance(secret_number, int)):
        raise ToolError("Parameter 'initial_number' expected a type integer")
    if not (isinstance(guessed_number, int)):
        raise ToolError("Parameter 'guessed_number' expected a type integer")

    # Tool logic and answering
    if secret_number > guessed_number:
        return "The secret number is higher than the guessed number. // (tool)"
    elif secret_number < guessed_number:
        return "The secret number is lower than the guessed number. // (tool)"
    else:
        return "The secret number is equal to the guessed number. You won ! // (tool)"


# Creating our two Agents
player = Agent("Player", "llama3:8b")
game_master = Agent("Game master", "llama3:8b")

# Instanciating our tool
high_low_tool = Tool("high_low", "Compares 2 numbers and returns a description of the relation between the two. Higher, lower or equal.", high_low)

# Solo Task to create the secret number before entering GroupSolve()
Task("Your task is to generate a secret random number between 1 and 20. Output the number just this once.", game_master).solve()

# Creating our two Tasks for GroupSolve
# The player Task is in charge of ending the chat when it wins  (because of `llm_stops_by_itself=True`)
player_task = Task("Your task is to guess a number between 1 and 20. You will receive feedback after each guess: either 'lower,' 'higher,' or 'You won!' Adjust your guess based on this feedback. You must propose only one number at a time. Continue guessing until you receive confirmation that you have guessed correctly and the game is won. Your objective is fulfilled when you won the game.", player, llm_stops_by_itself=True)

# The Game Master Task will call the tool and return the output as feedback to the player 
game_master_task = Task("A player will try to guess the number you generated. Respond to the guesses you receive with one of the following:\n* If the guess is higher than your number, say 'lower.'\n* If the guess is lower than your number, say 'higher.'\n* If the guess matches your number, say 'You won!' and end the game.\nDo not provide any other feedback or responses, and only play one round. To help you compare the numbers you have acces to a tool that describes the relation between your initial number and the guessed number", game_master, tools=[high_low_tool])

print("################Starting GroupSolve#################")
GroupSolve([player_task, game_master_task], EndChat(EndChatMode.END_CHAT_AFTER_FIRST_COMPLETION)).solve()

print("------ Player --------")
player.history.pretty_print()

print("------Game master----------")
game_master.history.pretty_print()
```

> How is tool calling mixed with GroupSolve()?
> Each time the Game Master Task is brought up. The tool will be called with both secret and guessed numbers. It will return some computable information describing the relation between the two numbers. A string that is inevitably right as it comes from 'classic' programming and not the LLM.

ℹ️ All Tools options are available so you could set `optional=True` if you wish.  

Output:
```
------ Player --------
[user]:
Your task is to guess a number between 1 and 20. You will receive feedback after each guess: either 'lower,' 'higher,' or 'You won!' Adjust your guess based on this feedback. You must propose only one number at a time. Continue guessing until you receive confirmation that you have guessed correctly and the game is won. Your objective is fulfilled when you won the game.

[assistant]:
I'll start by guessing 11.

What's the feedback?

[user]:
The secret number is higher than the guessed number. // (tool)

[assistant]:
Since the secret number is higher than my guess of 11, I'll try a higher number this time. My next guess is 17.

What's the feedback?

[user]:
The secret number is lower than the guessed number. // (tool)

[assistant]:
Since the secret number is lower than my guess of 17, I'll try a lower number this time. My next guess is 16.

What's the feedback?

[user]:
The secret number is lower than the guessed number. // (tool)

[assistant]:
Since the secret number is lower than my guess of 16, I'll try an even lower number this time. My next guess is 15.
What's the feedback?

[user]:
The secret number is lower than the guessed number. // (tool)

[assistant]:
Since the secret number is lower than my guess of 15, I'll try an even lower number this time. My next guess is 14.

What's the feedback?

[user]:
The secret number is equal to the guessed number. You won ! // (tool)

[assistant]:
Yay!

I'm glad I was able to guess the correct number, 14, after a few attempts! It was a fun game! Would you like to play again?

------Game master----------
[user]:
Your task is to generate a secret random number between 1 and 20. Output the number just this once.

[assistant]:
The secret random number is: **14**

[user]:
A player will try to guess the number you generated. Respond to the guesses you receive with one of the following:
* If the guess is higher than your number, say 'lower.'
* If the guess is lower than your number, say 'higher.'
* If the guess matches your number, say 'You won!' and end the game.
Do not provide any other feedback or responses, and only play one round. To help you compare the numbers you have acces to a tool that describes the relation between your initial number and the guessed number

[assistant]:
I can use the tool 'high_low' related to the task to solve it correctly.

[user]:
Output the tool 'high_low' as valid JSON.

[assistant]:
{"secret_number": 14, "guessed_number": 20}

[user]:
The secret number is lower than the guessed number. // (tool)

[user]:
I'll start by guessing 11.

What's the feedback?

[assistant]:
I can use the tool 'high_low' related to the task to solve it correctly.

[user]:
Output the tool 'high_low' as valid JSON.

[assistant]:
{"secret_number": 14, "guessed_number": 11}

[user]:
The secret number is higher than the guessed number. // (tool)

[user]:
Since the secret number is higher than my guess of 11, I'll try a higher number this time. My next guess is 17.

What's the feedback?

[assistant]:
I can use the tool 'high_low' related to the task to solve it correctly.

[user]:
Output the tool 'high_low' as valid JSON.

[assistant]:
{"secret_number": 14, "guessed_number": 17}

[user]:
The secret number is lower than the guessed number. // (tool)

[user]:
Since the secret number is lower than my guess of 17, I'll try a lower number this time. My next guess is 16.

What's the feedback?

[assistant]:
I can use the tool 'high_low' related to the task to solve it correctly.

[user]:
Output the tool 'high_low' as valid JSON.

[assistant]:
{"secret_number": 14, "guessed_number": 16}

[user]:
The secret number is lower than the guessed number. // (tool)

[user]:
Since the secret number is lower than my guess of 16, I'll try an even lower number this time. My next guess is 15.
What's the feedback?

[assistant]:
I can use the tool 'high_low' related to the task to solve it correctly.

[user]:
Output the tool 'high_low' as valid JSON.

[assistant]:
{"secret_number": 14,  "guessed_number": 15}

[user]:
The secret number is lower than the guessed number. // (tool)

[user]:
Since the secret number is lower than my guess of 15, I'll try an even lower number this time. My next guess is 14.

What's the feedback?

[assistant]:
I can use the tool 'high_low' related to the task to solve it correctly.

[user]:
Output the tool 'high_low' as valid JSON.

[assistant]:
{"secret_number": 14,  "guessed_number": 14}

[user]:
The secret number is equal to the guessed number. You won ! // (tool)
```

It worked!  

=> `The secret random number is: **14**`  

Tool calls:  
* {"secret_number": 14, "guessed_number": 20}
* {"secret_number": 14, "guessed_number": 11}
* {"secret_number": 14, "guessed_number": 17}
* {"secret_number": 14, "guessed_number": 16}
* {"secret_number": 14,  "guessed_number": 15}
* {"secret_number": 14,  "guessed_number": 14}

Last tool output: `The secret number is equal to the guessed number. You won ! // (tool)`


## X. Chat between many Agents

Yacana provides a way to make more than two agents speak one after the other. Better yet, there is no limit to the number of agents that you can add. However, note that the dual conversation pattern should be the one giving the best results as LLM were trained to speak to one user. Not be part of a multi-user conversation...  
Still, the functionality is here for you to use!  
To make this happen Yacana makes the Agents enter a role-play situation. It will heavily rely on the agent's name so be sure that each of them has a concise, yet meaningful, name/title.  
