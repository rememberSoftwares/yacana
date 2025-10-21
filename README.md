# Yacana

Task-driven multi-agents framework for developers to create open source LLM-powered apps with ease.  

<p align="center">
  <img src="https://github.com/user-attachments/assets/e59e056b-35c8-4077-a22a-3b6a72c9eb03">
</p>

---

<img src="http://ForTheBadge.com/images/badges/built-with-love.svg"/>  

<img src="https://img.shields.io/github/license/rememberSoftwares/yacana.svg"/> <img src="https://img.shields.io/badge/Maintained%3F-yes-green.svg">  [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/rememberSoftwares/yacana)




## What is Yacana

Yacana is designed for both *beginners* and *advanced* AI users.  

It features a simple OOP API with a **smooth learning curve**, while also offering advanced runtime LLM configurations when needed.  

The real strength of the framework lies in its ability to deliver impressive results with open-source models, even small ones, making **tool calling effortless with any LLM**.  

Yacana offers a **guided workflow** approach or **multi-turn chat** for production-grade capabilities, leveraging what we typically call '*Agents*'. However, Yacana takes a different approach to Agents compared to other frameworks, focusing more on chaining Tasks together rather than on the Agents themselves.

[Take this 1-minute survey to tell us what you think!](https://docs.google.com/forms/d/e/1FAIpQLSeaDaHgYagJWsQ97EkZTGRvpvq6tWzz7U9H7lojCsCI3fw9EQ/viewform?usp=dialog)

---

## Key Features

* 🔗 **Link** tasks together to create workflows
* 🧰 Enhanced **Tool Calling** for **every** local LLMs + **MCP** tools support
* 🤖 **Multi-agents** & **multi-turn** autonomous chat
* 🚄 **Streaming** for speed and **structured output** for reliability
* 🛠️ Same dev. API for both **Ollama** and **OpenAi** compatible servers
* 🚀 *Ready to use in minutes*

▶️▶️▶️ [Start by reading the documentation here](https://remembersoftwares.github.io/yacana/). ◀️◀️◀️    

*Yacana is free and open source under MIT license.*  
If you like Yacana consider giving a star to the repo! Opensource projects need your help! ⭐  

---

## Installation

```shell
pip install yacana
```

---

## Examples

🧪 Simple task solving and chaining (Ollama 🦙)
```python
from yacana import Task, OllamaAgent

ollama_agent = OllamaAgent("AI assistant", "llama3.1:8b", system_prompt="You are a helpful AI assistant")

# Create a task to tell a riddle
message = Task("Tell me a riddle. But do NOT tell me the answer.", ollama_agent).solve()

# "Why was the math book sad?"
print(message.content)

# Chain a second task to get the answer from the previous joke
message = Task("Now, give me the answer.", ollama_agent).solve()

# "Because it had too many problems!"
print(message.content)
```

<br />

🧪 Tool calling (Ollama 🦙)  
A calculator using tools called by the LLM to solve an equation. It makes use of the [Enhanced Tool Calling](https://remembersoftwares.github.io/yacana/pages/tool_calling.html) feature and [ModelSettings](https://remembersoftwares.github.io/yacana/pages/agents_and_tasks.html#configuring-llms-settings).
```python
from yacana import Task, OllamaAgent, ToolError, Tool, OllamaModelSettings

# Basic type validation to make sure the LLM gave correct values to the tools
def _validate_LLM_inputs(first_number, second_number):
    if not isinstance(first_number, int):
        raise ToolError("Parameter 'first_number' expected a type integer")
    if not isinstance(second_number, int):
        raise ToolError("Parameter 'second_number' expected a type integer")


# Simple math operations functions
def add(first_number: int, second_number: int) -> int:
    _validate_LLM_inputs(first_number, second_number)
    return first_number + second_number

def multiply(first_number: int, second_number: int) -> int:
    _validate_LLM_inputs(first_number, second_number)
    return first_number * second_number

def subtract(first_number: int, second_number: int) -> int:
    _validate_LLM_inputs(first_number, second_number)
    return first_number - second_number


# Defining tools
addition = Tool("Addition", "Add two integer numbers and returns the result.", add)
multiplication = Tool("Multiplication", "Multiplies two integer numbers and returns the result.", multiply)
subtraction = Tool("Subtraction", "Subtracts two integer numbers and returns the result.", subtract)

# Defining model settings like 'temperature' to get better results
model_settings = OllamaModelSettings(temperature=0.7)

ollama_agent = OllamaAgent("AI assistant", "llama3.1:8b", model_settings=model_settings)

# Solving an equation using ONLY the tools and not the underlying model knowledge
message1 = Task("Do 2+4-(6*7) by following PEMDAS. Make a plan first.", ollama_agent, tools=[addition, subtraction, multiplication]).solve()

print("\nAgent's history:\n")
ollama_agent.history.pretty_print()
# Will print the whole history of the agent, including the tools used and their parameters
#Result is: -36
```

<br />

🎉 New feature! 👋  
🧪 **Structured output** allows you to get valid JSON from an LLM and access it through a real object and not just a dict! (Ollama 🦙)  
```python
from pydantic import BaseModel
from yacana import GenericMessage, Task, OllamaAgent

class CountryFact(BaseModel):
    name: str
    fact: str

class Facts(BaseModel):
    countryFacts: list[CountryFact]

ollama_agent = OllamaAgent("AI assistant", "llama3.1:8b", system_prompt="You are a helpful AI assistant")

message: GenericMessage = Task("Tell me 3 facts about Canada.", ollama_agent, structured_output=Facts).solve()

# Print the content of the message as a JSON string
print(message.content)
# Print the structured output as a real class instance
print("Name = ", message.structured_output.countryFacts[0].name)
print("Fact = ", message.structured_output.countryFacts[0].fact)
```

<br />

🎉 New feature! 👋  
🧪 **Streaming** allows you to get tokens as they are generated by the LLM. (Ollama 🦙)  
```python
from yacana import Task, OllamaAgent

# A streaming callback that can do things with tokens
def streaming(chunk: str):
    print(f"chunk = |{chunk}|")

ollama_agent = OllamaAgent("AI assistant", "llama3.1:8b")

# Solving a task and streaming the result
message = Task("Count from 1 to 100 very quickly!", ollama_agent, streaming_callback=streaming).solve()
"""
INFO: [PROMPT][To: AI assistant]: Count from 1 to 100 very quickly!
chunk = |Here|
chunk = | we|
chunk = | go|
chunk = |:

|
chunk = |1|
chunk = |,|
chunk = | |
chunk = |2|
chunk = |,|
chunk = | |
chunk = |3|
chunk = |,|
chunk = | |
chunk = |4|
chunk = |,|
chunk = | |
chunk = |5|
chunk = |,|
chunk = | |
chunk = |6|
"""
```

<br />

🦙🤖💬  
* *All of this works with Ollama and **any OpenAI-compatible servers!** (ChatGPT, VLLM, etc)  
Just replace the OllamaAgent by an [OpenAiAgent](https://remembersoftwares.github.io/yacana/pages/other_inference_servers.html) and you're good to go!*  
* Many other features to discover like medias (images, sound, ...) or managing the history by tags, etc!  

## Quick demo

Let's make an application that looks for PDF invoices inside a folder then checks if you have enough money to pay them and finaly rename them so things don't get messy!  
Order of operation:   
1. Check if it is an invoice. If not it will skip to the next one.
2. Deduct the money on the invoice from the bank account (`@checking_account_limit` variable) and tell you if you don't have enough money to pay for everything!
3. Rename the invoice file to match `<category><total price>.pdf` so that it's clean.

*We'll test with these 3 PDFs. Two invoices and one random text:*

<p align="center">
  <img src="https://github.com/user-attachments/assets/9a58b927-0017-4214-b1ef-331a7c0fafaf">
</p>

### Demo setup

```shell
pip install yacana

# Only for parsing the PDFs in this demo
pip install pypdf

git clone https://github.com/rememberSoftwares/yacana.git
cd yacana/examples/invoices_checker/ollama
python3 quick_demo.py
```

⚠️ **Requirements:**  
* Before running the script make sure that you installed Ollama on your computer  
* The Agents are using `llama3.1:8b`. If you are using another LLM model, update the 3 agents' declaration in the script to match the one you installed:  

➡️
```python
agent1 = OllamaAgent("Expert banker", "llama3.1:8b", model_settings=ms)
agent2 = OllamaAgent("Naming expert", "llama3.1:8b")
agent3 = OllamaAgent("File-system helper", "llama3.1:8b", model_settings=ms)
```
* If you are not using Ollama please refer to: [Using other inference servers](https://remembersoftwares.github.io/yacana/pages/other_inference_servers.html) and use one of the corresponding [demos](https://github.com/rememberSoftwares/yacana/tree/main/examples/invoices_checker).

**Script:**

```python
from yacana import OllamaAgent, Task, Tool, GroupSolve, EndChat, EndChatMode, LoggerManager, ToolError, OllamaModelSettings
import os
from typing import List
from pypdf import PdfReader

# How much money you have on your bank account
checking_account_limit: int = 3000
# Path where to find the invoices
invoices_folder_path = "../assets/invoices/"
# Uncomment to hide info logs.
# LoggerManager.set_log_level(None)


#################
#    HELPERS    #
#################
def list_invoices() -> List[str]:
    """
    Not a 'tool' ; List all files in the folder
    :return:
    """
    return [f for f in os.listdir(invoices_folder_path) if os.path.isfile(os.path.join(invoices_folder_path, f))]


def read_pdf(file_name: str) -> str:
    """
    Not a tool ; Returns the content of a PDF file
    :param file_name:
    :return:
    """
    # creating a pdf reader object
    reader = PdfReader(file_name)
    # extracting text from all pages
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text





###############
#    TOOLS    #
###############
def invoice_expense_tracker(invoice_total: float) -> str:
    """
    Deducts an amount of money from the bank account and returns data on the current balance
    :param invoice_total:
    :return:
    """
    global checking_account_limit
    if not isinstance(invoice_total, int) and not isinstance(invoice_total, float):
        raise ToolError("Invoice total must be a number (float or integer)")
    checking_account_limit -= invoice_total
    tool_deduction: str = f"After deducing {invoice_total}$ from the checking account. The current balance is now at {checking_account_limit}"
    print("[Tool]: ", tool_deduction)
    return tool_deduction

def check_file_existence(file_name: str) -> str:
    """
    Checks if a file exists with the given name
    :param file_name:
    :return:
    """
    print("[Tool]: Checking file existence of ", file_name)
    if os.path.exists(invoices_folder_path + file_name) is True:
        answer: str = "This file name is already taken. Find something else."
    else:
        answer: str = "File name is available."
        print("[Tool]: ", answer)
    return answer





###############
#    Logic    #
###############

# Lowering temperature so the LLM doesn't get too creative
ms = OllamaModelSettings(temperature=0.4)

# Creating 3 agents
agent1 = OllamaAgent("Expert banker", "llama3.1:8b", model_settings=ms)
agent2 = OllamaAgent("File-system helper", "llama3.1:8b", model_settings=ms)
agent3 = OllamaAgent("Naming expert", "llama3.1:8b")


# Registering 2 tools
expense_tracker_tool: Tool = Tool("Expense tracker", "Takes as input a price from an invoice and deducts it from the user's account. Returns the new account balance.", invoice_expense_tracker)
check_file_existence_tool = Tool("File existence checker", "Takes as input a file name and tells if the name in already taken", check_file_existence)

# Making a checkpoint, so we can go back in time later
checkpoint_ag1: str = agent1.history.create_check_point()
checkpoint_ag2: str = agent2.history.create_check_point()
checkpoint_ag3: str = agent3.history.create_check_point()

# Listing PDF to read
files: List[str] = list_invoices()

# Looping on each PDF
for invoice_file in files:

    # Getting PDF content
    invoice_content: str = read_pdf(invoices_folder_path + invoice_file)

    Task(f"You will get the content of a pdf. Determine if the file is an invoice or not. The pdf content is the following: {invoice_content}", agent1).solve()

    # Yes/no router
    router: str = Task(f"Is the file an invoice ? If it is, answer ONLY by 'yes' else answer ONLY by 'no'.", agent1).solve().content
    if "yes" in router.lower():
        Task(f"Extract the total price from the invoice.", agent1).solve()
        # Calling tool
        Task("We must register this new price into an invoice tracker", agent1, tools=[expense_tracker_tool]).solve()
        # Yes/no router
        router = Task("Is the current account balance still positive ? Answer ONLY by 'yes' or 'no'.", agent1, forget=True).solve().content
        # !! Reversed condition !! ; looking for 'yes' or its absence is safer than looking for 'no'
        if "yes" not in router.lower():
            print("WARNING ! You are spending to much !!")

        # Multi-agent chat to determine a new name for the PDF
        GroupSolve(
            [
                Task("You must find a name for the invoice file. It must follow this pattern: '<category>_<total_price>.pdf'", agent1),
                Task("Check that the proposed file name is not already taken.", agent2, tools=[check_file_existence_tool]),
                Task("If the file name is already taken, add an incrementation to the end of the name. Your objective is complete as soon as a correct file name is found. No need to research further.", agent3, llm_stops_by_itself=True)
            ],
            EndChat(EndChatMode.END_CHAT_AFTER_FIRST_COMPLETION, max_iterations=3)
        ).solve()

        new_file_name = Task("Output ONLY the chosen file name and nothing else", agent1).solve().content
        print(f"File {invoice_file} will be renamed to '{new_file_name}'")
        # Renaming PDF file
        os.rename(invoices_folder_path + invoice_file, invoices_folder_path + new_file_name)

    else:
        print(f"File {invoice_file} is not an invoice. Skipping...")

    # Loading checkpoint to reset all agents to a previous state
    agent1.history.load_check_point(checkpoint_ag1)
    agent2.history.load_check_point(checkpoint_ag2)
    agent3.history.load_check_point(checkpoint_ag3)
```

### Call graph

![invoice_demo](https://github.com/user-attachments/assets/7cf6fd5f-325f-4868-b4c9-0667a30543fd)

To look at more complete examples try the [examples folder](https://github.com/rememberSoftwares/yacana/tree/main/examples).

---

## Releases

**0.3.3**  
+ Minor doc fixes.

**0.3.2**  
+ Fixed streaming error.

**0.3.1**  
+ Fixed major error in requirements asking for Numpy.

**0.3.0**  
+ MCP tools support in Alpha.
+ Mixing Yacana and OpenAi tool calling styles with both OllamaAgent and OpenAiAgent.
+ Internal use of structured output can be shutdown if your LLM does not support it.
+ Tools without arguments now have their own workflow when using Yacana tool calling style.
+ Removed Herobrine

**0.2.1**  
+ Ollama Enhanced Tool Calling is now better (70% accuracy upgrade!).
+ More history features.
+ Support for thinking models like deepseek-R1

**0.2.0**  
+ Streaming.
+ Structured output.
+ OpenAI compatible agents.
+ Tags for history messages.

**0.1.3**  
+ The initial release of Yacana.

## Roadmap

❗ Highest priority  
* Adding native RAG capabilities
* More MCP resources support

❕ Lower priority  
* Adding memory to agents
* Simplify shift message and maybe rework GroupChat itself a bit.  
* Keeping working on the documentation.  
* Adding a section about code generation.  

More to come

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more details.


