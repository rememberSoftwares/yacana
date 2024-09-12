# Yacana

Task-driven multi-agents framework for developers to create open source LLM-powered apps with ease.  

<p align="center">
  <img src="https://github.com/user-attachments/assets/c54b2e47-8a76-444d-9510-339acc1d16a8">
</p>

## What is Yacana

Yacana is designed for both *beginners* and *advanced* AI users.  

It features a simple OOP API with a **smooth learning curve**, while also offering advanced runtime LLM configurations when needed.  

The real strength of the framework lies in its ability to deliver impressive results with open-source models, even small ones, making **tool calling effortless with any LLM**.  

For production-grade capabilities, Yacana offers a **guided workflow** approach or **multi-turn chat**, both leveraging what we typically call '*Agents*'. However, Yacana takes a different approach to Agents compared to other frameworks, focusing more on chaining Tasks together rather than on the Agents themselves.

## Most trending features

* Link tasks together to create workflows
* Tool calling for every LLMs
* Multi-agents autonomous chat
* Ready to use in minutes

*Yacana is free and open source under MIT license.*  

Be ready in minutes!  

▶️▶️▶️ [Start by reading the documentation here](https://remembersoftwares.github.io/yacana/). ◀️◀️◀️    

## Installation
```shell
pip install yacana
```

## Quick demo

Let's make an application that looks for PDF invoices inside a folder. For each one it will:  
* Check if it is an invoice. If not it will skip to the next one.
* Deduct the money on the invoice from the bank account (`@checking_account_limit`) and tell you if you don't have enough money to pay for everything!
* Rename the invoice file name to match `<category><total price>.pdf`.

### Demo setup

```shell
pip install yacana

# Only for parsing the PDFs in this demo
pip install pypdf

git clone https://github.com/rememberSoftwares/yacana.git
cd yacana/examples/invoices_checker
python quick_demo.py
```

⚠️ Requirements:  
* Before running the script make sure that you installed Ollama on your computer
* The Agents are using `llama3.1:8b`. If you are using another LLM model, update the 3 agent declarations in the script to match the one you installed:
➡️
```
agent1 = Agent("Expert banker", "llama3.1:8b", model_settings=ms)
agent2 = Agent("Naming expert", "llama3.1:8b")
agent3 = Agent("File-system helper", "llama3.1:8b", model_settings=ms)
```

---

```python
from yacana import Agent, Task, Tool, GroupSolve, EndChat, EndChatMode, ModelSettings, LoggerManager, ToolError
import os
from typing import List
from pypdf import PdfReader

# How much money you have on your bank account
checking_account_limit: int = 3000
# Path where to find the invoices
invoices_folder_path = "../assets/invoices/"
# Uncomment to hide info logs.
# LoggerManager.set_log_level(None)


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
ms = ModelSettings(temperature=0.4)

# Creating 3 agents
agent1 = Agent("Expert banker", "llama3.1:8b", model_settings=ms)
agent2 = Agent("File-system helper", "llama3.1:8b", model_settings=ms)
agent3 = Agent("Naming expert", "llama3.1:8b")


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

## Roadmap

❗ Highest priority
* Compatibility with inference servers other than Ollama, like vllm, etc.

❕ Lower priority
* Simplifying shift message system and maybe reworking GroupChat itself a bit.
* Keep working on the documentation.
* Add a section on code generation.
