## Software Architecture :
The high-level structure of a system showing components and how they interact.

## My Project Module contains:
-Input handling

-Intent understanding

-AI processing

-Task modules

-External APIs

## Layered Architecture Style
Layered architecture means the system is divided into layers where:

-Each layer has a specific responsibility
-Layers communicate in order
-One layer does not directly control everythin

## Level 1 (Presentation Layer)
In your code:

input("You: ")
print("AI:", reply)

Accept user messages
Display output
Provide interface to system

## Layer 2 (Intent Classifier)
It classifies into:

CHAT,EMAIL,CALENDAR,NEWS


If this layer didn’t exist then
Every module would run unnecessarily
System would be slow and messy

## Layer 3 Application 

This layer contains main logic modules :
email_tool()
calendar_tool()
news_tool()
normal_chat()

Each module:
Receives processed input
Performs logic
Returns result

Example:
Calendar tool:
Extract details
Validate date/time
Create event

## Layer 4 (AI and API)
This is where intelligence happens.

Components:
HuggingFace model
LangChain chains
Prompt templates

And API connect our LLM to the outside world
like gmail api,  duckduckgo

## Data Flows

Step-by-step flow:

1.User enters message
2.Intent classifier analyzes it
3.System selects module
4.Extraction chain processes details
5.Business logic executes
6.API call happens
7.Response returned to user

Request → Processing → Response 

## Maintainable
This is "Maintainable" because i explicitly used and create my own
tool for gmail, calender and news so that i can customize it and 
if in future gmail_tool break then i can fix this seperatly witjout
affecting other functions

## Scalability
I can scale it as per user need without affecting or changing other 
module.

## Modularity
Each tool is independent

## Reusability
Intent classifier can be reused.
Email module can be reused.

## Testability
You can test each tool separately

## Componenets
1. Intent Handler - Read user msgs
2. Intent Classifier - Decide action
3. Extract Modeules and tools - Extract structured data from natural language
4. Task Modules - Email Module, Calender Module, News Module, Chat Module
5. External APIs


