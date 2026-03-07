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

## Component : Intent Classifier and Email_Tool
Purpose

The intent classifier determines what the user wants to do.

Example intents: Chat, Email, Calender, News

## Working
Steps performed:

1.Accept user message
2.Analyze text using rules or AI model
3.Identify user intent
4.Forward request to the correct module

### Example Code

```python
def classify_intent(message):
    if "mail" in message:
        return "EMAIL"
    elif "meeting" in message:
        return "CALENDAR"
    elif "news" in message:
        return "NEWS"
    else:
        return "CHAT"
```
Here it take input as message from intent generation chain 
and through this function it find out if there is keyword like "mail" and matches the keyword and perform the action accordingly.

## Email Tool Function

The `email_tool()` function is responsible for sending emails based on a user message.



# Application Components

## Components Present in the System

The software system consists of the following application components:

---

## 1. Frontend (User Interface)

The Frontend is the part of the system that interacts directly with the user.

**Responsibilities:**
- Accepts user input (messages or commands)
- Displays responses and results
- Sends requests to the backend through APIs

**Examples:**
- Web interface
- Chat interface

---

## 2. Backend 

The Backend acts as the central controller of the system.

**Responsibilities:**
- Receives requests from the frontend
- Processes user input
- Routes the request to the correct service
- Sends results back to the frontend

---

## 3. Intent Classifier

This component determines what the user wants to do.

**Responsibilities:**
- Analyzes user messages
- Classifies intent such as:
  - EMAIL
  - CALENDAR
  - NEWS
  - CHAT

**Example Code:**

```python
def classify_intent(message):
    if "mail" in message:
        return "EMAIL"
    elif "meeting" in message:
        return "CALENDAR"
    elif "news" in message:
        return "NEWS"
    else:
        return "CHAT"
```
### 4.Email Service

Handles email-related tasks.
Extracts recipient, subject, and body.
Sends email using an email API.

## 5. Calendar Service

Handles meeting scheduling tasks.
Extracts date and time.
Creates calendar events

## 6. News Service
Handles news-related requests.
Fetches news from an API.
Returns summarized results

## 7. Chat Service
Handles general conversation.
Provides responses when the request is not related to email, calendar, or news.

## 8. External APIs / Services
These are third-party services used by the system.
Email API
News API
Calendar API










# Assignment — Software Architecture

**Project:** Intelligent Task Automation Platform

---

## I. Chosen Architecture Style: **Microservices Architecture**

The system is designed as a **Microservices** architecture with a central **Orchestrator** (API Gateway + coordinator). Multiple small, independently deployable services communicate over HTTP/APIs and share minimal state (session in Redis, logs in MongoDB).

---

### A. Justification: How the Software Falls in This Category (Component Granularity)

- **Granularity:** Each deployable unit is a **separate process** with a **single, well-defined responsibility** and its own runtime/stack:
  - **Frontend** — Single React SPA; presentation only; talks to Orchestrator via REST.
  - **Orchestrator** — Single Node/Express service; API gateway, auth, session, workflow coordination (NLU → RAG → Planner → Task Execution); does not implement NLU/RAG logic itself.
  - **NLU Service** — Separate FastAPI (Python) process; only intent + entity extraction; exposed as an HTTP API.
  - **RAG Service** — Separate Python process + Vector DB; only retrieval/augmentation; exposed as an HTTP API.
  - **Connectors** — Implemented inside the Orchestrator process (Calendar, Email, Todo); can be split into separate services later without changing the overall style.
- **Communication:** Services interact via **synchronous HTTP/API calls** (Orchestrator → NLU, RAG, LLM). No shared in-process memory; only shared data stores are Redis (session) and MongoDB (logs), used by the Orchestrator.
- **Deployment:** Frontend, Orchestrator, NLU, and RAG can be **deployed and scaled independently** (e.g. multiple NLU or RAG instances behind a load balancer).

**Diagram — Component granularity and communication:**

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           MICROSERVICES BOUNDARIES                           │
├─────────────────┬──────────────────┬─────────────────┬───────────────────────┤
│   Frontend      │   Orchestrator   │   NLU Service   │   RAG Service         │
│   (React SPA)   │   (Node/Express) │   (FastAPI)     │   (Python + Vector)   │
├─────────────────┼──────────────────┼─────────────────┼───────────────────────┤
│ • UI            │ • API Gateway    │ • /parse        │ • /query              │
│ • Auth flows    │ • Auth middleware│ • Intent        │ • Retrieval           │
│ • Chat / Plan   │ • Session (Redis)│ • Entities      │ • Vector DB           │
│                 │ • Call NLU/RAG   │                 │                       │
│                 │ • Planner/LLM    │                 │                       │
│                 │ • Task Exec      │                 │                       │
│                 │ • Connectors     │                 │                       │
│                 │ • Logs (MongoDB) │                 │                       │
└────────┬────────┴─────────┬────────┴──────────┬──────┴────────────┬──────────┘
         │                  │                   │                   │
         │    HTTP/REST     │     HTTP          │      HTTP         │
         └──────────────────┴───────────────────┴───────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
                 Redis           MongoDB         LLM 
               (session)         (data)       (optional)
```

---

### B. Why This Architecture Is the Best Choice

- **Scalability**
  - **NLU** and **RAG** can be scaled independently (e.g. more replicas under load) without scaling the Orchestrator or Frontend.
  - Orchestrator can be scaled horizontally behind a load balancer; Redis/MongoDB can be scaled per product needs.
- **Maintainability**
  - Clear **bounded contexts**: NLU team can change models/frameworks without touching RAG or Orchestrator; same for RAG and connectors.
  - **Small codebases per service** (frontend, orchestrator, nlu, rag) reduce cognitive load and ease onboarding.
- **Performance**
  - **Right tool per job:** Python/FastAPI for NLU/RAG (ML/data stacks); Node for Orchestrator (I/O-bound, many external calls). No single monolith bottleneck.
  - **Caching and isolation:** Session in Redis keeps auth/session off the main DB; heavy RAG/NLU work stays in their own processes.
- **Other requirements**
  - **Technology diversity:** Microservices allow different languages (TypeScript, Python) and runtimes (Node, FastAPI, Vector DB) in one system.
  - **Evolvability:** New connectors or AI services can be added as new endpoints or new microservices without rewriting existing ones.
  - **Fault isolation:** A failure in NLU or RAG can be contained (timeouts, fallbacks) without bringing down the whole application.

**Trade-off acknowledged:** Operational complexity (more services to deploy and monitor) is accepted in return for scalability, maintainability, and the ability to use the best stack per component.

---

## II. Application Components (Present in the Project)

| # | Component | Responsibility | Technology |
|---|-----------|----------------|------------|
| 1 | **Frontend** | User interface: login/register, landing, dashboard, chat, plan approval, sessions list | React, TypeScript, Vite |
| 2 | **API Gateway / Orchestrator** | Single entry for API; routing, auth middleware, request/response handling | Node.js, Express (inside Orchestrator) |
| 3 | **Orchestrator Service** | Coordinates workflow: call NLU → RAG → Planner (LLM) → Task Execution; manages session and connectors | Node.js, Express |
| 4 | **Session Manager** | Store and retrieve user session state (e.g. conversation context) | Redis |
| 5 | **NLU Service** | Parse natural language; output intent and entities (e.g. `schedule_meeting`, `date: tomorrow`) | FastAPI (Python) |
| 6 | **RAG Service** | Query domain knowledge; retrieve relevant context for planning/execution | Python, Vector DB |
| 7 | **Planner / LLM** | Generate step-by-step plan from NLU output + RAG context | LLM (hosted API or local, e.g. Ollama) |
| 8 | **Connector Manager** | Execute approved actions: calendar, email, todo | Node (inside Orchestrator); adapters for GCal, Gmail/SMTP, Todo |
| 9 | **Raw Logs Store** | Persist request/execution logs for auditing and debugging | MongoDB |

**Diagram — Application components and data flow:**

```
                    ┌──────────────┐
                    │   User       │
                    └──────┬───────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 1. Frontend (React)                                                       │
│    • Landing, Login, Register, Dashboard, Chat, Sessions                  │
└──────┬───────────────────────────────────────────────────────────────────┘
       │ HTTP
       ▼
┌──────────────────────────────────────────────────────────────────────────┐
│ 2. API Gateway / 3. Orchestrator Service (Node/Express)                  │
│    • Auth, route requests, coordinate NLU → RAG → Planner → Task Exec     │
└──┬───────┬───────────┬────────────┬─────────────┬───────────────────────┘
   │       │           │            │             │
   ▼       ▼           ▼            ▼             ▼
┌─────┐ ┌─────┐   ┌────────┐  ┌─────────┐  ┌──────────────┐
│ 4.  │ │ 9.  │   │ 5. NLU │  │ 6. RAG  │  │ 7. Planner   │
│Redis│ │Mongo│   │Service │  │ Service │  │ / LLM        │
│     │ │ DB  │   │(FastAPI)│  │(Python) │  │(external)   │
└─────┘ └─────┘   └────────┘  └─────────┘  └──────────────┘
                                               │
                                               ▼
                                        ┌──────────────┐
                                        │ 8. Connector │
                                        │ Manager      │
                                        │ Calendar /   │
                                        │ Email / Todo  │
                                        └──────────────┘
```

---

## Summary

- **Architecture style:** Microservices (with central Orchestrator).
- **Granularity:** Frontend, Orchestrator, NLU, RAG (and Connectors within Orchestrator) are separate deployable units with single responsibilities and HTTP boundaries.
- **Rationale:** Best fit for scalability, maintainability, performance, and technology diversity for this AI task-automation project.
- **Application components:** Nine components listed above (Frontend, Gateway, Orchestrator, Session/Redis, NLU, RAG, Planner/LLM, Connector Manager, Logs/MongoDB).