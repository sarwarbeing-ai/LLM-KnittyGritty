🔍 Building a Binary QA Model on Contextual Documents

From RAG to Fine-Tuning for Better Binary QA 🚀

Traditionally, without fine-tuning:  
We used a **RAG (Retrieval-Augmented Generation)** system where, for a given question, the top-*k* relevant documents were retrieved. These document chunks were passed as **context** to an LLM, which would then generate a **binary (Yes/No)** answer based on the provided information.

The LLM also needed to identify the specific chunk that justified its answer.

👉 But here's the catch: while this works, the approach depends heavily on zero-shot or few-shot performance of the base model—and accuracy can vary.



💡 So, why not fine-tune?

Fine-tuning has become dramatically easier and more accessible today, especially with smaller open-source models like **Gemma3, llama3, etc**. It offers better **consistency**, **latency**, and **accuracy** for specific downstream tasks like **binary classification with context grounding**.



🔧 What I did:

I **supervised fine-tuned the Gemma3 1B model** on a structured dataset built for binary QA tasks.  

Each data sample includes:  
- A **question**  
- Multiple **document chunks + metadata**  
- A **label** ("Yes"/"No")  
- The specific **chunk** that supports or contradicts the answer





🔎 Dataset Sample

🧠 System Prompt:

**Task:**  
You are given a **question** along with multiple **contexts** and their associated **metadata**.

**Your goal is to:**

1. **If the question is answered with "Yes" by any context:**  
   - Return the **context** that supports a "Yes" answer, along with its **metadata** and the answer `"Yes"`.

2. **If no context supports a "Yes" answer:**  
   - Return a **context that contradicts** the question (i.e., implies the answer is "No"), along with its **metadata** and the answer `"No"`.
   


👤 User Prompt (sample):


**Question:** Does the report mention sustainability initiatives?

**Context 1:**  
_"In 2022, EcoTech hosted a Sustainability Summit in Northern Cascadia, bringing together 14 suppliers to explore climate-positive manufacturing. Workshops included carbon offsetting, waste reduction, and community-led reforestation."_  
**Metadata:** Company: EcoTech Industries | Page: 47

**Context 2:**  
_"Our logistics operations focus on cost and delivery optimization, without involving environmental programs."_  
**Metadata:** Company: ShipXpress Corp | Page: 12

**Context 3:**  
_"We are currently evaluating possible improvements in vendor performance for FY2023, with no mention of environmental or social governance."_  
**Metadata:** Company: ByteMark Solutions | Page: 30

**Context 4:**
_"SolarNova has implemented a zero-landfill policy across all production units and aims to become carbon-neutral by 2027. As of this year, 70% of its energy usage comes from renewable sources."_
**Metadata:** Company: SolarNova Technologies | Page: 5

**Context 5:**
_"The marketing strategy this year is centered around expanding influencer partnerships and entering new demographics, particularly Gen Z consumers."_
**Metadata:** Company: TrendSpire Media | Page: 91


🤖 Assistant Response:

**Answer:** Yes  
**Chunk:** _"In 2022, EcoTech hosted a Sustainability Summit in Northern Cascadia, bringing together 14 suppliers to explore climate-positive manufacturing. Workshops included carbon offsetting, waste reduction, and community-led reforestation."_  
**Company:** EcoTech Industries

**Page:** 47
