<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Building Agentic RAG with Self-Reflection using Google ADK: A Complete Tutorial

Based on the article "How Agents Think," we'll explore how to implement two powerful agentic design patterns: **Agentic RAG** and **Self-Reflection** using Google's Agent Development Kit (ADK). These patterns represent sophisticated thinking loops that enable agents to reason, retrieve information, and improve their responses iteratively.

## Understanding Agent Thinking Patterns

Before diving into implementation, let's understand how these patterns work according to the **thought-action-observation cycle**[^1]:

### Agentic RAG Pattern

Unlike traditional RAG that simply retrieves and generates, Agentic RAG adds **planning, reasoning, and decision-making** to the retrieval process[^2]. The agent actively decides:

- What information to search for
- Which tools to use for retrieval
- Whether retrieved information is sufficient
- How to synthesize multiple sources


### Self-Reflection Pattern

Self-reflection implements a **critique-and-improve loop** where agents evaluate their own outputs and iteratively refine them[^2]. This follows the pattern:

1. **Generate** initial response
2. **Critique** the response quality
3. **Refine** based on feedback
4. **Repeat** until satisfactory

## Prerequisites and Setup

### Installation

```bash
# Install Google ADK
pip install google-adk

# Install additional dependencies for RAG
pip install chromadb sentence-transformers requests beautifulsoup4
```


### Basic Project Structure

```
agentic_rag_project/
├── agent.py
├── tools/
│   ├── __init__.py
│   ├── rag_tools.py
│   └── reflection_tools.py
├── data/
│   └── knowledge_base/
└── requirements.txt
```


## Step 1: Building the RAG Foundation

### Create Vector Database and Search Tools

```python
# tools/rag_tools.py
import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import requests
from bs4 import BeautifulSoup

class RAGTools:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("knowledge_base")
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to the vector database"""
        embeddings = self.encoder.encode(documents)
        ids = [f"doc_{i}" for i in range(len(documents))]
        
        self.collection.add(
            embeddings=embeddings.tolist(),
            documents=documents,
            metadatas=metadata or [{}] * len(documents),
            ids=ids
        )
    
    def vector_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search the vector database for relevant documents"""
        query_embedding = self.encoder.encode([query])
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        return [
            {
                "content": doc,
                "metadata": meta,
                "score": 1 - distance  # Convert distance to similarity
            }
            for doc, meta, distance in zip(
                results['documents'][^0],
                results['metadatas'][^0],
                results['distances'][^0]
            )
        ]
    
    def web_search(self, query: str, num_results: int = 3) -> List[Dict]:
        """Perform web search for recent information"""
        # This is a simplified example - in production, use proper search APIs
        search_url = f"https://www.google.com/search?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; research bot)'}
        
        try:
            response = requests.get(search_url, headers=headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract search results (simplified)
            results = []
            for i, result in enumerate(soup.find_all('div', class_='g')[:num_results]):
                title_elem = result.find('h3')
                snippet_elem = result.find('span')
                
                if title_elem and snippet_elem:
                    results.append({
                        "title": title_elem.get_text(),
                        "content": snippet_elem.get_text(),
                        "source": "web_search"
                    })
            
            return results
        except Exception as e:
            return [{"content": f"Web search failed: {str(e)}", "source": "error"}]

# Initialize RAG tools
rag_tools = RAGTools()
```


## Step 2: Implementing Self-Reflection Tools

```python
# tools/reflection_tools.py
from google.adk.agents import LlmAgent

class ReflectionTools:
    def __init__(self):
        # Critic agent for evaluating responses
        self.critic_agent = LlmAgent(
            name="content_critic",
            model="gemini-2.0-flash",
            instruction="""You are a critical evaluator of AI responses.
            
            Analyze the given response for:
            - Accuracy and factual correctness
            - Completeness and thoroughness
            - Clarity and coherence
            - Relevance to the original question
            - Use of sources and citations
            
            Provide specific, actionable feedback for improvement.
            Rate the response quality from 1-10 and explain your reasoning."""
        )
        
        # Refiner agent for improving responses
        self.refiner_agent = LlmAgent(
            name="content_refiner",
            model="gemini-2.0-pro",
            instruction="""You are an expert content refiner.
            
            Given an original response and critical feedback, create an improved version that:
            - Addresses all identified issues
            - Maintains the original intent
            - Improves clarity and accuracy
            - Better integrates source information
            
            Return only the refined response."""
        )
    
    async def critique_response(self, original_query: str, response: str) -> Dict:
        """Critique a response and provide feedback"""
        critique_prompt = f"""
        Original Query: {original_query}
        
        Response to Evaluate: {response}
        
        Please provide your critical analysis and improvement suggestions.
        """
        
        critique_result = await self.critic_agent.run(critique_prompt)
        return {
            "critique": critique_result,
            "needs_improvement": "score: [1-6]" in critique_result.lower()
        }
    
    async def refine_response(self, original_query: str, response: str, critique: str) -> str:
        """Refine a response based on critique"""
        refine_prompt = f"""
        Original Query: {original_query}
        
        Original Response: {response}
        
        Critical Feedback: {critique}
        
        Please provide an improved response that addresses the feedback.
        """
        
        refined_response = await self.refiner_agent.run(refine_prompt)
        return refined_response

# Initialize reflection tools
reflection_tools = ReflectionTools()
```


## Step 3: Creating the Main Agentic RAG Agent

```python
# agent.py
from google.adk.agents import LlmAgent
from tools.rag_tools import rag_tools
from tools.reflection_tools import reflection_tools
import asyncio

class AgenticRAGAgent:
    def __init__(self):
        # Main reasoning agent
        self.main_agent = LlmAgent(
            name="agentic_rag_researcher",
            model="gemini-2.0-pro",
            instruction="""You are an advanced research assistant using Agentic RAG with self-reflection.
            
            For each query, follow this process:
            1. PLAN: Analyze what information you need
            2. SEARCH: Use vector search for existing knowledge and web search for recent info
            3. EVALUATE: Determine if you have sufficient information
            4. SYNTHESIZE: Create a comprehensive response with citations
            5. REFLECT: Self-evaluate and improve if needed
            
            Always cite your sources and be transparent about your reasoning process."""
        )
        
        self.max_reflection_iterations = 2
    
    async def search_and_retrieve(self, query: str) -> Dict:
        """Perform comprehensive information retrieval"""
        # Vector database search
        vector_results = rag_tools.vector_search(query, top_k=5)
        
        # Web search for recent information
        web_results = rag_tools.web_search(query, num_results=3)
        
        return {
            "vector_results": vector_results,
            "web_results": web_results,
            "total_sources": len(vector_results) + len(web_results)
        }
    
    async def synthesize_response(self, query: str, search_results: Dict) -> str:
        """Create initial response from retrieved information"""
        context = self._format_search_results(search_results)
        
        synthesis_prompt = f"""
        Query: {query}
        
        Available Information:
        {context}
        
        Based on the retrieved information, provide a comprehensive answer that:
        - Directly addresses the query
        - Integrates information from multiple sources
        - Includes proper citations
        - Acknowledges any limitations or gaps
        """
        
        response = await self.main_agent.run(synthesis_prompt)
        return response
    
    def _format_search_results(self, results: Dict) -> str:
        """Format search results for context"""
        context = "Vector Database Results:\n"
        for i, result in enumerate(results["vector_results"]):
            context += f"[{i+1}] {result['content']} (Score: {result['score']:.2f})\n"
        
        context += "\nWeb Search Results:\n"
        for i, result in enumerate(results["web_results"]):
            context += f"[{len(results['vector_results'])+i+1}] {result['content']} (Source: {result.get('title', 'Web')})\n"
        
        return context
    
    async def run_with_reflection(self, query: str) -> Dict:
        """Main execution with self-reflection loop"""
        print(f"🔍 Processing query: {query}")
        
        # Step 1: Search and retrieve information
        print("📚 Searching for relevant information...")
        search_results = await self.search_and_retrieve(query)
        print(f"Found {search_results['total_sources']} sources")
        
        # Step 2: Initial synthesis
        print("🧠 Synthesizing initial response...")
        current_response = await self.synthesize_response(query, search_results)
        
        # Step 3: Self-reflection loop
        iteration = 0
        reflection_history = []
        
        while iteration < self.max_reflection_iterations:
            print(f"🔄 Reflection iteration {iteration + 1}")
            
            # Critique the current response
            critique_result = await reflection_tools.critique_response(query, current_response)
            reflection_history.append(critique_result)
            
            # If response is good enough, break
            if not critique_result["needs_improvement"]:
                print("✅ Response quality is satisfactory")
                break
            
            # Refine the response
            print("🔧 Refining response based on critique...")
            current_response = await reflection_tools.refine_response(
                query, current_response, critique_result["critique"]
            )
            
            iteration += 1
        
        return {
            "final_response": current_response,
            "search_results": search_results,
            "reflection_history": reflection_history,
            "iterations": iteration + 1
        }

# Initialize the agent
agentic_rag_agent = AgenticRAGAgent()
```


## Step 4: Advanced Features and Workflow Management

### Adding Parallel Processing for Multiple Queries

```python
# Enhanced agent with parallel processing
from google.adk.agents import ParallelAgent

class AdvancedAgenticRAG(AgenticRAGAgent):
    def __init__(self):
        super().__init__()
        
        # Specialized sub-agents for different types of queries
        self.factual_agent = LlmAgent(
            name="factual_researcher",
            model="gemini-2.0-flash",
            instruction="Specialize in factual, data-driven research with emphasis on accuracy."
        )
        
        self.analytical_agent = LlmAgent(
            name="analytical_researcher", 
            model="gemini-2.0-pro",
            instruction="Specialize in analytical thinking, comparisons, and strategic insights."
        )
    
    async def route_query(self, query: str) -> str:
        """Determine which specialized agent should handle the query"""
        routing_prompt = f"""
        Analyze this query and determine if it's primarily:
        - FACTUAL: Seeking specific facts, data, or information
        - ANALYTICAL: Requiring analysis, comparison, or strategic thinking
        
        Query: {query}
        
        Respond with either 'FACTUAL' or 'ANALYTICAL'
        """
        
        route_decision = await self.main_agent.run(routing_prompt)
        return "factual" if "FACTUAL" in route_decision.upper() else "analytical"
    
    async def run_specialized_research(self, query: str) -> Dict:
        """Run research with specialized agent routing"""
        # Determine routing
        agent_type = await self.route_query(query)
        selected_agent = self.factual_agent if agent_type == "factual" else self.analytical_agent
        
        print(f"🎯 Routing to {agent_type} specialist")
        
        # Continue with normal agentic RAG process using specialized agent
        search_results = await self.search_and_retrieve(query)
        
        # Use specialized agent for synthesis
        context = self._format_search_results(search_results)
        synthesis_prompt = f"""
        Query: {query}
        Available Information: {context}
        
        Provide a specialized {agent_type} response addressing this query.
        """
        
        response = await selected_agent.run(synthesis_prompt)
        
        # Apply reflection loop
        critique_result = await reflection_tools.critique_response(query, response)
        
        if critique_result["needs_improvement"]:
            response = await reflection_tools.refine_response(
                query, response, critique_result["critique"]
            )
        
        return {
            "final_response": response,
            "agent_type": agent_type,
            "search_results": search_results,
            "reflection_applied": critique_result["needs_improvement"]
        }
```


## Step 5: Usage Examples and Testing

### Basic Usage Example

```python
# example_usage.py
import asyncio

async def main():
    # Initialize knowledge base with sample documents
    sample_docs = [
        "Artificial Intelligence agents use reasoning loops to make decisions.",
        "RAG systems combine retrieval with generation for better responses.",
        "Self-reflection in AI involves critiquing and improving outputs.",
        "Google ADK provides tools for building multi-agent systems."
    ]
    
    metadata = [
        {"source": "AI textbook", "topic": "agents"},
        {"source": "RAG paper", "topic": "retrieval"},
        {"source": "Reflection study", "topic": "self-improvement"},
        {"source": "ADK docs", "topic": "development"}
    ]
    
    rag_tools.add_documents(sample_docs, metadata)
    
    # Test queries
    test_queries = [
        "How do AI agents use reasoning loops?",
        "What are the benefits of combining RAG with self-reflection?",
        "Compare traditional RAG with Agentic RAG approaches"
    ]
    
    for query in test_queries:
        print(f"\n{'='*60}")
        result = await agentic_rag_agent.run_with_reflection(query)
        
        print(f"Query: {query}")
        print(f"Final Response: {result['final_response']}")
        print(f"Reflection Iterations: {result['iterations']}")
        print(f"Sources Used: {result['search_results']['total_sources']}")

if __name__ == "__main__":
    asyncio.run(main())
```


### Advanced Multi-Agent Example

```python
# multi_agent_example.py
from google.adk.agents import LlmAgent

async def create_research_team():
    """Create a multi-agent research team with specialized roles"""
    
    # Orchestrator agent
    orchestrator = LlmAgent(
        name="research_orchestrator",
        model="gemini-2.0-pro",
        instruction="""You coordinate a research team with specialized agents.
        
        Available team members:
        - agentic_rag_researcher: Handles information retrieval and synthesis
        - content_critic: Evaluates response quality
        - content_refiner: Improves responses based on feedback
        
        Delegate tasks appropriately and ensure high-quality outputs.""",
        sub_agents=[
            agentic_rag_agent.main_agent,
            reflection_tools.critic_agent,
            reflection_tools.refiner_agent
        ]
    )
    
    return orchestrator

async def run_team_research(query: str):
    """Run research using the full team"""
    orchestrator = await create_research_team()
    
    team_prompt = f"""
    Research Query: {query}
    
    Please coordinate with your team to:
    1. Conduct thorough research using agentic RAG
    2. Have the critic evaluate the initial response
    3. Refine the response if needed
    4. Provide a final, high-quality answer
    
    Ensure the final response includes citations and reasoning transparency.
    """
    
    result = await orchestrator.run(team_prompt)
    return result
```


## Step 6: Monitoring and Optimization

### Performance Tracking

```python
# monitoring.py
import time
from typing import Dict, List

class AgentPerformanceMonitor:
    def __init__(self):
        self.metrics = {
            "response_times": [],
            "reflection_iterations": [],
            "source_counts": [],
            "quality_scores": []
        }
    
    def track_execution(self, result: Dict, execution_time: float):
        """Track execution metrics"""
        self.metrics["response_times"].append(execution_time)
        self.metrics["reflection_iterations"].append(result.get("iterations", 0))
        self.metrics["source_counts"].append(
            result.get("search_results", {}).get("total_sources", 0)
        )
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary statistics"""
        return {
            "avg_response_time": sum(self.metrics["response_times"]) / len(self.metrics["response_times"]),
            "avg_reflections": sum(self.metrics["reflection_iterations"]) / len(self.metrics["reflection_iterations"]),
            "avg_sources": sum(self.metrics["source_counts"]) / len(self.metrics["source_counts"]),
            "total_queries": len(self.metrics["response_times"])
        }

# Usage with monitoring
monitor = AgentPerformanceMonitor()

async def monitored_research(query: str):
    start_time = time.time()
    result = await agentic_rag_agent.run_with_reflection(query)
    execution_time = time.time() - start_time
    
    monitor.track_execution(result, execution_time)
    return result
```


## Key Benefits and Applications

### Advantages of This Implementation

1. **Intelligent Information Retrieval**: The agent actively decides what to search for and evaluates information quality[^2]
2. **Self-Improving Responses**: Reflection loops ensure higher quality outputs through iterative refinement[^2]
3. **Transparent Reasoning**: Each step of the thinking process is visible and auditable[^1]
4. **Scalable Architecture**: Multi-agent design allows for specialized expertise and parallel processing[^3]
5. **Flexible Tool Integration**: Easy to add new search tools and knowledge sources[^2]

### Real-World Applications

- **Research Assistants**: Comprehensive literature reviews with quality assurance
- **Customer Support**: Self-improving responses with knowledge base integration
- **Content Creation**: Fact-checked, well-researched content with multiple source validation
- **Decision Support**: Analysis combining internal knowledge with external research


## Conclusion

This implementation demonstrates how to combine **Agentic RAG** and **Self-Reflection** patterns using Google ADK to create sophisticated AI agents that think, reason, and improve their responses. The system follows the core principles from "How Agents Think" by implementing proper **thought-action-observation cycles**[^1] while leveraging Google ADK's powerful multi-agent capabilities[^3].

The resulting agent can handle complex research tasks, evaluate its own performance, and continuously improve its responses - representing a significant advancement over simple chatbot interactions toward true agentic intelligence.

<div style="text-align: center">⁂</div>

[^1]: https://boringbot.substack.com/p/day-2-how-agents-think

[^2]: https://saptak.in/writing/2025/04/26/powerful-agentic-design-patterns-for-building-ai-agents-with-google-adk

[^3]: https://developers.googleblog.com/en/agent-development-kit-easy-to-build-multi-agent-applications/

[^4]: https://www.youtube.com/watch?v=VtveulQzByo

[^5]: https://www.youtube.com/watch?v=TvW4A0a75mw

[^6]: https://github.com/google/adk-samples/tree/main/agents/RAG

[^7]: https://gaodalie.substack.com/p/google-adk-mcp-rag-ollama-the-key

[^8]: https://www.youtube.com/watch?v=v5ymBTXNqtk

[^9]: https://www.youtube.com/watch?v=4QZNNJEpG-k

[^10]: https://cloud.google.com/blog/products/ai-machine-learning/build-multi-agentic-systems-using-google-adk

[^11]: https://www.youtube.com/watch?v=wPMWiiubNew

[^12]: https://www.reddit.com/r/PromptEngineering/comments/1kj5lva/implementing_multiple_agent_samples_using_google/

[^13]: https://dev.to/mayank_laddha_ml/simple-code-to-understand-self-reflection-agentic-design-pattern-58o3

[^14]: https://www.reddit.com/r/Rag/comments/1k3sf39/google_adk_agent_development_kit_rag/

[^15]: https://docs.truefoundry.com/docs/tracing/tracing-in-google-adk

[^16]: https://google.github.io/adk-docs/agents/multi-agents/

[^17]: https://www.youtube.com/watch?v=H9omrb4lkM0

[^18]: https://github.com/google/adk-samples

[^19]: https://codelabs.developers.google.com/instavibe-adk-multi-agents/instructions

[^20]: https://google.github.io/adk-docs/tutorials/

[^21]: https://www.leewayhertz.com/agentic-rag/

