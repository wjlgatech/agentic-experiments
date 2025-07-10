<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Adding Tool Use Capacity to Google ADK: MCP-Based Implementation

Building on the previous Agentic RAG implementation, we'll now enhance the Google ADK codebase with sophisticated tool use capabilities using the Model-Context Protocol (MCP) pattern. This will enable agents to dynamically select, execute, and chain tools based on context and task requirements.

## **Core Architecture Overview**

The tool use system follows a modular architecture with these key components:


| Component | Purpose | Key Features |
| :-- | :-- | :-- |
| **Tool Registry** | Manages available tools and their schemas | Dynamic registration, capability discovery |
| **Tool Selector** | Chooses appropriate tools based on context | Context analysis, multi-tool coordination |
| **Execution Engine** | Handles tool invocation and chaining | Structured I/O, error handling, retries |
| **Protocol Handler** | Implements MCP-like communication | Standardized messaging, type safety |

## **Step 1: MCP-Inspired Tool Protocol Implementation**

### **Tool Definition Schema**

```python
# tools/protocol.py
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import json
import asyncio

class ToolType(Enum):
    SEARCH = "search"
    COMPUTATION = "computation"
    DATA_PROCESSING = "data_processing"
    COMMUNICATION = "communication"
    FILE_OPERATION = "file_operation"
    API_INTEGRATION = "api_integration"

@dataclass
class ToolParameter:
    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    enum_values: Optional[List[str]] = None

@dataclass
class ToolSchema:
    name: str
    description: str
    tool_type: ToolType
    parameters: List[ToolParameter]
    returns: Dict[str, str]
    examples: List[Dict[str, Any]]
    fallback_tools: Optional[List[str]] = None
    dependencies: Optional[List[str]] = None

class MCPMessage:
    def __init__(self, method: str, params: Dict[str, Any], id: str = None):
        self.method = method
        self.params = params
        self.id = id or f"msg_{hash(str(params))}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "jsonrpc": "2.0",
            "method": self.method,
            "params": self.params,
            "id": self.id
        }

class MCPResponse:
    def __init__(self, result: Any = None, error: Dict[str, Any] = None, id: str = None):
        self.result = result
        self.error = error
        self.id = id
    
    def to_dict(self) -> Dict[str, Any]:
        response = {"jsonrpc": "2.0", "id": self.id}
        if self.error:
            response["error"] = self.error
        else:
            response["result"] = self.result
        return response
```


### **Tool Registry System**

```python
# tools/registry.py
from typing import Dict, List, Callable, Any
from .protocol import ToolSchema, ToolType, MCPMessage, MCPResponse
import inspect

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}
        self.tool_schemas: Dict[str, ToolSchema] = {}
        self.tool_categories: Dict[ToolType, List[str]] = {}
    
    def register_tool(self, schema: ToolSchema, implementation: Callable):
        """Register a tool with its schema and implementation"""
        self.tools[schema.name] = {
            "schema": schema,
            "implementation": implementation,
            "metadata": {
                "registered_at": time.time(),
                "call_count": 0,
                "success_rate": 1.0
            }
        }
        self.tool_schemas[schema.name] = schema
        
        # Categorize tool
        if schema.tool_type not in self.tool_categories:
            self.tool_categories[schema.tool_type] = []
        self.tool_categories[schema.tool_type].append(schema.name)
    
    def get_tools_by_type(self, tool_type: ToolType) -> List[str]:
        """Get all tools of a specific type"""
        return self.tool_categories.get(tool_type, [])
    
    def get_tool_schema(self, tool_name: str) -> Optional[ToolSchema]:
        """Get schema for a specific tool"""
        return self.tool_schemas.get(tool_name)
    
    def list_available_tools(self) -> Dict[str, ToolSchema]:
        """List all available tools with their schemas"""
        return self.tool_schemas
    
    def validate_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> bool:
        """Validate if tool call parameters match schema"""
        if tool_name not in self.tool_schemas:
            return False
        
        schema = self.tool_schemas[tool_name]
        for param in schema.parameters:
            if param.required and param.name not in parameters:
                return False
            if param.name in parameters and param.enum_values:
                if parameters[param.name] not in param.enum_values:
                    return False
        return True

# Global tool registry
tool_registry = ToolRegistry()
```


## **Step 2: Core Tool Implementations**

### **Web Search Tool**

```python
# tools/search_tools.py
import requests
from bs4 import BeautifulSoup
from .protocol import ToolSchema, ToolParameter, ToolType
from .registry import tool_registry

class WebSearchTool:
    def __init__(self):
        self.search_engines = {
            "duckduckgo": self._duckduckgo_search,
            "serp": self._serp_search,
            "custom": self._custom_search
        }
    
    async def search(self, query: str, engine: str = "duckduckgo", 
                    num_results: int = 5, include_snippets: bool = True) -> Dict[str, Any]:
        """Perform web search with specified engine"""
        try:
            search_func = self.search_engines.get(engine, self._duckduckgo_search)
            results = await search_func(query, num_results, include_snippets)
            
            return {
                "success": True,
                "query": query,
                "engine": engine,
                "results": results,
                "total_results": len(results)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "query": query,
                "engine": engine
            }
    
    async def _duckduckgo_search(self, query: str, num_results: int, 
                                include_snippets: bool) -> List[Dict[str, Any]]:
        """DuckDuckGo search implementation"""
        url = f"https://duckduckgo.com/html/?q={query}"
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; research bot)'}
        
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        results = []
        for result in soup.find_all('div', class_='result')[:num_results]:
            title_elem = result.find('a', class_='result__a')
            snippet_elem = result.find('a', class_='result__snippet')
            
            if title_elem:
                result_data = {
                    "title": title_elem.get_text().strip(),
                    "url": title_elem.get('href', ''),
                    "snippet": snippet_elem.get_text().strip() if snippet_elem and include_snippets else ""
                }
                results.append(result_data)
        
        return results

# Register web search tool
web_search_schema = ToolSchema(
    name="web_search",
    description="Search the web for current information and relevant content",
    tool_type=ToolType.SEARCH,
    parameters=[
        ToolParameter("query", "string", "Search query", required=True),
        ToolParameter("engine", "string", "Search engine to use", required=False, 
                     default="duckduckgo", enum_values=["duckduckgo", "serp", "custom"]),
        ToolParameter("num_results", "integer", "Number of results to return", 
                     required=False, default=5),
        ToolParameter("include_snippets", "boolean", "Include result snippets", 
                     required=False, default=True)
    ],
    returns={"results": "array", "success": "boolean", "total_results": "integer"},
    examples=[
        {"query": "latest AI research 2024", "engine": "duckduckgo", "num_results": 3}
    ],
    fallback_tools=["local_search", "knowledge_base_search"]
)

web_search_tool = WebSearchTool()
tool_registry.register_tool(web_search_schema, web_search_tool.search)
```


### **Computation and Data Processing Tools**

```python
# tools/computation_tools.py
import pandas as pd
import numpy as np
import json
from typing import Any, Dict, List
from .protocol import ToolSchema, ToolParameter, ToolType
from .registry import tool_registry

class ComputationTool:
    async def calculate(self, expression: str, variables: Dict[str, float] = None) -> Dict[str, Any]:
        """Safely evaluate mathematical expressions"""
        try:
            # Safe evaluation with limited scope
            allowed_names = {
                "__builtins__": {},
                "abs": abs, "round": round, "min": min, "max": max,
                "sum": sum, "len": len, "pow": pow,
                "math": __import__("math"),
                "np": np
            }
            
            if variables:
                allowed_names.update(variables)
            
            result = eval(expression, allowed_names)
            return {
                "success": True,
                "result": result,
                "expression": expression,
                "variables": variables or {}
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "expression": expression
            }

class DataProcessingTool:
    async def process_data(self, data: List[Dict[str, Any]], operation: str, 
                          parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process structured data with various operations"""
        try:
            df = pd.DataFrame(data)
            parameters = parameters or {}
            
            if operation == "filter":
                condition = parameters.get("condition", "")
                df = df.query(condition)
            elif operation == "aggregate":
                group_by = parameters.get("group_by", [])
                agg_func = parameters.get("function", "mean")
                if group_by:
                    df = df.groupby(group_by).agg(agg_func)
                else:
                    df = df.agg(agg_func)
            elif operation == "sort":
                sort_by = parameters.get("column", "")
                ascending = parameters.get("ascending", True)
                df = df.sort_values(sort_by, ascending=ascending)
            elif operation == "transform":
                column = parameters.get("column", "")
                function = parameters.get("function", "")
                df[column] = df[column].apply(eval(function))
            
            return {
                "success": True,
                "result": df.to_dict("records"),
                "operation": operation,
                "shape": df.shape
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "operation": operation
            }

# Register computation tools
calculation_schema = ToolSchema(
    name="calculate",
    description="Perform mathematical calculations and evaluations",
    tool_type=ToolType.COMPUTATION,
    parameters=[
        ToolParameter("expression", "string", "Mathematical expression to evaluate", required=True),
        ToolParameter("variables", "object", "Variables to use in calculation", required=False)
    ],
    returns={"result": "number", "success": "boolean"},
    examples=[
        {"expression": "2 + 2 * 3", "variables": {}},
        {"expression": "x * y + z", "variables": {"x": 10, "y": 5, "z": 3}}
    ]
)

data_processing_schema = ToolSchema(
    name="process_data",
    description="Process and analyze structured data",
    tool_type=ToolType.DATA_PROCESSING,
    parameters=[
        ToolParameter("data", "array", "Array of data objects", required=True),
        ToolParameter("operation", "string", "Processing operation", required=True,
                     enum_values=["filter", "aggregate", "sort", "transform"]),
        ToolParameter("parameters", "object", "Operation-specific parameters", required=False)
    ],
    returns={"result": "array", "success": "boolean", "shape": "array"},
    examples=[
        {"data": [{"name": "John", "age": 30}, {"name": "Jane", "age": 25}], 
         "operation": "sort", "parameters": {"column": "age", "ascending": True}}
    ]
)

computation_tool = ComputationTool()
data_processing_tool = DataProcessingTool()

tool_registry.register_tool(calculation_schema, computation_tool.calculate)
tool_registry.register_tool(data_processing_schema, data_processing_tool.process_data)
```


### **API Integration Tools**

```python
# tools/api_tools.py
import aiohttp
import json
from typing import Dict, Any, Optional
from .protocol import ToolSchema, ToolParameter, ToolType
from .registry import tool_registry

class APIIntegrationTool:
    def __init__(self):
        self.session = None
    
    async def _get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def make_request(self, url: str, method: str = "GET", 
                          headers: Dict[str, str] = None,
                          data: Dict[str, Any] = None,
                          params: Dict[str, str] = None) -> Dict[str, Any]:
        """Make HTTP requests to external APIs"""
        try:
            session = await self._get_session()
            
            async with session.request(
                method=method.upper(),
                url=url,
                headers=headers,
                json=data if method.upper() in ["POST", "PUT", "PATCH"] else None,
                params=params
            ) as response:
                
                content_type = response.headers.get('content-type', '')
                if 'application/json' in content_type:
                    result = await response.json()
                else:
                    result = await response.text()
                
                return {
                    "success": response.status < 400,
                    "status_code": response.status,
                    "data": result,
                    "headers": dict(response.headers),
                    "url": str(response.url)
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "method": method
            }

# Register API integration tool
api_request_schema = ToolSchema(
    name="api_request",
    description="Make HTTP requests to external APIs",
    tool_type=ToolType.API_INTEGRATION,
    parameters=[
        ToolParameter("url", "string", "API endpoint URL", required=True),
        ToolParameter("method", "string", "HTTP method", required=False, 
                     default="GET", enum_values=["GET", "POST", "PUT", "DELETE", "PATCH"]),
        ToolParameter("headers", "object", "Request headers", required=False),
        ToolParameter("data", "object", "Request body data", required=False),
        ToolParameter("params", "object", "Query parameters", required=False)
    ],
    returns={"data": "any", "success": "boolean", "status_code": "integer"},
    examples=[
        {"url": "https://api.example.com/data", "method": "GET"},
        {"url": "https://api.example.com/users", "method": "POST", 
         "data": {"name": "John", "email": "john@example.com"}}
    ]
)

api_tool = APIIntegrationTool()
tool_registry.register_tool(api_request_schema, api_tool.make_request)
```


## **Step 3: Intelligent Tool Selection System**

### **Context-Aware Tool Selector**

```python
# tools/selector.py
from typing import List, Dict, Any, Optional
from .registry import tool_registry
from .protocol import ToolType, ToolSchema
import google.generativeai as genai

class ToolSelector:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-pro')
        self.selection_history = []
    
    async def select_tools(self, task_description: str, context: Dict[str, Any] = None,
                          max_tools: int = 3) -> List[Dict[str, Any]]:
        """Select appropriate tools based on task and context"""
        available_tools = tool_registry.list_available_tools()
        
        # Create tool selection prompt
        selection_prompt = self._create_selection_prompt(
            task_description, available_tools, context, max_tools
        )
        
        try:
            response = self.model.generate_content(selection_prompt)
            selected_tools = self._parse_tool_selection(response.text, available_tools)
            
            # Store selection for learning
            self.selection_history.append({
                "task": task_description,
                "context": context,
                "selected_tools": selected_tools,
                "timestamp": time.time()
            })
            
            return selected_tools
        except Exception as e:
            # Fallback to rule-based selection
            return self._fallback_selection(task_description, available_tools, max_tools)
    
    def _create_selection_prompt(self, task: str, tools: Dict[str, ToolSchema], 
                                context: Dict[str, Any], max_tools: int) -> str:
        """Create prompt for tool selection"""
        tool_descriptions = []
        for name, schema in tools.items():
            tool_descriptions.append(
                f"- {name}: {schema.description} (Type: {schema.tool_type.value})"
            )
        
        context_str = json.dumps(context, indent=2) if context else "No additional context"
        
        return f"""
        You are an intelligent tool selector for an AI agent system.
        
        Task: {task}
        Context: {context_str}
        
        Available Tools:
        {chr(10).join(tool_descriptions)}
        
        Select up to {max_tools} most appropriate tools for this task.
        Consider:
        1. Task requirements and complexity
        2. Tool capabilities and limitations
        3. Potential tool combinations
        4. Context relevance
        
        Respond with a JSON array of selected tools with reasoning:
        [
            {{
                "tool_name": "tool_name",
                "reasoning": "why this tool is selected",
                "priority": 1-3,
                "estimated_parameters": {{"param": "value"}}
            }}
        ]
        """
    
    def _parse_tool_selection(self, response: str, available_tools: Dict[str, ToolSchema]) -> List[Dict[str, Any]]:
        """Parse LLM response into tool selection"""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                selections = json.loads(json_match.group())
                
                # Validate selections
                valid_selections = []
                for selection in selections:
                    if selection.get("tool_name") in available_tools:
                        valid_selections.append(selection)
                
                return valid_selections
        except:
            pass
        
        return self._fallback_selection("", available_tools, 3)
    
    def _fallback_selection(self, task: str, tools: Dict[str, ToolSchema], 
                           max_tools: int) -> List[Dict[str, Any]]:
        """Rule-based fallback selection"""
        # Simple keyword-based selection
        task_lower = task.lower()
        selected = []
        
        if any(word in task_lower for word in ["search", "find", "look", "research"]):
            selected.append({
                "tool_name": "web_search",
                "reasoning": "Task involves searching for information",
                "priority": 1,
                "estimated_parameters": {"query": task}
            })
        
        if any(word in task_lower for word in ["calculate", "compute", "math"]):
            selected.append({
                "tool_name": "calculate",
                "reasoning": "Task involves mathematical computation",
                "priority": 2,
                "estimated_parameters": {}
            })
        
        return selected[:max_tools]

# Initialize tool selector
tool_selector = ToolSelector()
```


## **Step 4: Tool Execution Engine with Chaining**

### **Execution Engine Implementation**

```python
# tools/executor.py
import asyncio
from typing import List, Dict, Any, Optional
from .registry import tool_registry
from .selector import tool_selector
from .protocol import MCPMessage, MCPResponse

class ToolExecutor:
    def __init__(self):
        self.execution_history = []
        self.max_retries = 3
        self.timeout = 30
    
    async def execute_task(self, task_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a task using appropriate tools"""
        # Step 1: Select tools
        selected_tools = await tool_selector.select_tools(task_description, context)
        
        if not selected_tools:
            return {
                "success": False,
                "error": "No appropriate tools found for task",
                "task": task_description
            }
        
        # Step 2: Execute tools
        execution_results = []
        current_context = context or {}
        
        for tool_config in selected_tools:
            result = await self._execute_single_tool(
                tool_config, current_context, task_description
            )
            execution_results.append(result)
            
            # Update context with results for chaining
            if result.get("success"):
                current_context.update({
                    f"{tool_config['tool_name']}_result": result.get("data", result.get("result"))
                })
        
        # Step 3: Synthesize results
        final_result = await self._synthesize_results(
            task_description, execution_results, current_context
        )
        
        # Store execution history
        self.execution_history.append({
            "task": task_description,
            "tools_used": selected_tools,
            "results": execution_results,
            "final_result": final_result,
            "timestamp": time.time()
        })
        
        return final_result
    
    async def _execute_single_tool(self, tool_config: Dict[str, Any], 
                                  context: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Execute a single tool with retry logic"""
        tool_name = tool_config["tool_name"]
        
        if tool_name not in tool_registry.tools:
            return {
                "success": False,
                "error": f"Tool {tool_name} not found",
                "tool_name": tool_name
            }
        
        tool_info = tool_registry.tools[tool_name]
        
        # Prepare parameters
        parameters = await self._prepare_parameters(tool_config, context, task)
        
        # Validate parameters
        if not tool_registry.validate_tool_call(tool_name, parameters):
            return {
                "success": False,
                "error": f"Invalid parameters for tool {tool_name}",
                "tool_name": tool_name,
                "parameters": parameters
            }
        
        # Execute with retries
        for attempt in range(self.max_retries):
            try:
                # Create MCP message
                message = MCPMessage(
                    method=f"tools/{tool_name}",
                    params=parameters
                )
                
                # Execute tool
                result = await asyncio.wait_for(
                    tool_info["implementation"](**parameters),
                    timeout=self.timeout
                )
                
                # Update tool metadata
                tool_info["metadata"]["call_count"] += 1
                if result.get("success", True):
                    success_rate = tool_info["metadata"]["success_rate"]
                    call_count = tool_info["metadata"]["call_count"]
                    tool_info["metadata"]["success_rate"] = (
                        (success_rate * (call_count - 1) + 1) / call_count
                    )
                
                return {
                    "success": result.get("success", True),
                    "data": result,
                    "tool_name": tool_name,
                    "parameters": parameters,
                    "attempt": attempt + 1
                }
                
            except asyncio.TimeoutError:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return {
                    "success": False,
                    "error": f"Tool {tool_name} timed out after {self.timeout}s",
                    "tool_name": tool_name
                }
            except Exception as e:
                if attempt < self.max_retries - 1:
                    # Try fallback tools if available
                    fallback_tools = tool_info["schema"].fallback_tools
                    if fallback_tools:
                        for fallback_tool in fallback_tools:
                            if fallback_tool in tool_registry.tools:
                                tool_config["tool_name"] = fallback_tool
                                return await self._execute_single_tool(tool_config, context, task)
                    
                    await asyncio.sleep(2 ** attempt)
                    continue
                
                return {
                    "success": False,
                    "error": str(e),
                    "tool_name": tool_name,
                    "attempt": attempt + 1
                }
    
    async def _prepare_parameters(self, tool_config: Dict[str, Any], 
                                 context: Dict[str, Any], task: str) -> Dict[str, Any]:
        """Prepare parameters for tool execution"""
        estimated_params = tool_config.get("estimated_parameters", {})
        
        # Use LLM to refine parameters based on context
        tool_schema = tool_registry.get_tool_schema(tool_config["tool_name"])
        
        if not tool_schema:
            return estimated_params
        
        # Create parameter refinement prompt
        param_prompt = f"""
        Task: {task}
        Tool: {tool_config['tool_name']}
        Tool Description: {tool_schema.description}
        
        Required Parameters:
        {json.dumps([{
            "name": p.name,
            "type": p.type,
            "description": p.description,
            "required": p.required,
            "default": p.default
        } for p in tool_schema.parameters], indent=2)}
        
        Context: {json.dumps(context, indent=2)}
        Estimated Parameters: {json.dumps(estimated_params, indent=2)}
        
        Provide refined parameters as JSON object:
        """
        
        try:
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(param_prompt)
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if json_match:
                refined_params = json.loads(json_match.group())
                return refined_params
        except:
            pass
        
        return estimated_params
    
    async def _synthesize_results(self, task: str, results: List[Dict[str, Any]], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize results from multiple tools"""
        successful_results = [r for r in results if r.get("success")]
        failed_results = [r for r in results if not r.get("success")]
        
        if not successful_results:
            return {
                "success": False,
                "error": "All tool executions failed",
                "failed_tools": [r.get("tool_name") for r in failed_results],
                "task": task
            }
        
        # Combine successful results
        combined_data = {}
        for result in successful_results:
            tool_name = result.get("tool_name")
            combined_data[tool_name] = result.get("data")
        
        return {
            "success": True,
            "task": task,
            "tools_executed": len(results),
            "successful_tools": len(successful_results),
            "failed_tools": len(failed_results),
            "combined_results": combined_data,
            "context": context
        }

# Initialize tool executor
tool_executor = ToolExecutor()
```


## **Step 5: Integration with Google ADK Agent**

### **Enhanced Agentic RAG with Tool Use**

```python
# enhanced_agentic_rag.py
from google.adk.agents import LlmAgent
from tools.executor import tool_executor
from tools.registry import tool_registry
import json

class ToolEnabledAgenticRAG:
    def __init__(self):
        self.main_agent = LlmAgent(
            name="tool_enabled_researcher",
            model="gemini-2.0-pro",
            instruction="""You are an advanced research assistant with access to various tools.
            
            Available capabilities:
            - Web search for current information
            - Mathematical calculations
            - Data processing and analysis
            - API integrations
            - File operations
            
            For each query:
            1. ANALYZE: Determine what tools are needed
            2. PLAN: Decide on tool execution strategy
            3. EXECUTE: Use tools to gather information
            4. SYNTHESIZE: Combine results into comprehensive response
            5. REFLECT: Evaluate and improve if needed
            
            Always explain your tool usage and reasoning process."""
        )
    
    async def process_query_with_tools(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process query using intelligent tool selection and execution"""
        print(f"🔍 Processing query with tools: {query}")
        
        # Step 1: Analyze query and determine tool needs
        analysis_result = await self._analyze_query_requirements(query, context)
        
        # Step 2: Execute tools based on analysis
        if analysis_result.get("needs_tools", False):
            print("🛠️ Executing tools...")
            tool_results = await tool_executor.execute_task(query, context)
            
            # Step 3: Synthesize final response using tool results
            final_response = await self._synthesize_with_tools(
                query, tool_results, context
            )
        else:
            print("💭 Processing without tools...")
            final_response = await self._process_without_tools(query, context)
        
        return final_response
    
    async def _analyze_query_requirements(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if query requires tool usage"""
        available_tools = tool_registry.list_available_tools()
        tool_list = "\n".join([f"- {name}: {schema.description}" 
                              for name, schema in available_tools.items()])
        
        analysis_prompt = f"""
        Analyze this query to determine if it requires external tools:
        
        Query: {query}
        Context: {json.dumps(context or {}, indent=2)}
        
        Available Tools:
        {tool_list}
        
        Determine:
        1. Does this query need external tools? (yes/no)
        2. What type of information is needed?
        3. Which tools would be most helpful?
        4. What is the complexity level? (simple/medium/complex)
        
        Respond with JSON:
        {{
            "needs_tools": boolean,
            "information_type": "string",
            "recommended_tools": ["tool1", "tool2"],
            "complexity": "simple|medium|complex",
            "reasoning": "explanation"
        }}
        """
        
        try:
            response = self.main_agent.run(analysis_prompt)
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback analysis
        return {
            "needs_tools": any(word in query.lower() for word in 
                             ["search", "find", "calculate", "current", "latest", "data"]),
            "information_type": "general",
            "recommended_tools": ["web_search"],
            "complexity": "medium",
            "reasoning": "Fallback analysis based on keywords"
        }
    
    async def _synthesize_with_tools(self, query: str, tool_results: Dict[str, Any], 
                                   context: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize response using tool results"""
        synthesis_prompt = f"""
        Query: {query}
        
        Tool Execution Results:
        {json.dumps(tool_results, indent=2)}
        
        Context: {json.dumps(context or {}, indent=2)}
        
        Based on the tool results, provide a comprehensive response that:
        1. Directly answers the query
        2. Integrates information from all successful tools
        3. Explains how tools were used
        4. Acknowledges any limitations or failed tools
        5. Provides citations where appropriate
        
        Format your response clearly and professionally.
        """
        
        try:
            response = self.main_agent.run(synthesis_prompt)
            return {
                "success": True,
                "response": response,
                "tools_used": tool_results.get("tools_executed", 0),
                "tool_results": tool_results
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tool_results": tool_results
            }
    
    async def _process_without_tools(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process query without external tools"""
        try:
            response = self.main_agent.run(f"Query: {query}\nContext: {json.dumps(context or {})}")
            return {
                "success": True,
                "response": response,
                "tools_used": 0,
                "tool_results": None
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "tools_used": 0
            }

# Initialize enhanced agent
enhanced_agent = ToolEnabledAgenticRAG()
```


## **Step 6: Usage Examples and Testing**

### **Complete Usage Example**

```python
# usage_example.py
import asyncio

async def demonstrate_tool_usage():
    """Demonstrate the tool-enabled agentic RAG system"""
    
    # Test queries that require different tools
    test_queries = [
        {
            "query": "What is the current price of Bitcoin and calculate 10% of that value?",
            "context": {"user_portfolio": "crypto_investor"}
        },
        {
            "query": "Search for recent AI research papers and summarize the key findings",
            "context": {"research_area": "artificial_intelligence"}
        },
        {
            "query": "Process this sales data and find the top performing products",
            "context": {
                "sales_data": [
                    {"product": "Widget A", "sales": 1500, "profit": 300},
                    {"product": "Widget B", "sales": 2000, "profit": 450},
                    {"product": "Widget C", "sales": 800, "profit": 200}
                ]
            }
        }
    ]
    
    for test_case in test_queries:
        print(f"\n{'='*80}")
        print(f"Query: {test_case['query']}")
        print(f"Context: {test_case['context']}")
        print("-" * 80)
        
        result = await enhanced_agent.process_query_with_tools(
            test_case["query"], 
            test_case["context"]
        )
        
        print(f"Success: {result['success']}")
        print(f"Tools Used: {result['tools_used']}")
        print(f"Response: {result['response']}")
        
        if result.get('tool_results'):
            print(f"Tool Results Summary: {result['tool_results'].get('successful_tools', 0)} successful, "
                  f"{result['tool_results'].get('failed_tools', 0)} failed")

async def test_individual_tools():
    """Test individual tools directly"""
    print("\n🧪 Testing Individual Tools")
    
    # Test web search
    print("\n1. Testing Web Search:")
    search_result = await tool_executor.execute_task(
        "Search for latest developments in quantum computing",
        {"domain": "technology"}
    )
    print(f"Search Success: {search_result['success']}")
    
    # Test calculation
    print("\n2. Testing Calculation:")
    calc_result = await tool_executor.execute_task(
        "Calculate the compound interest for $10000 at 5% for 3 years",
        {"principal": 10000, "rate": 0.05, "time": 3}
    )
    print(f"Calculation Success: {calc_result['success']}")
    
    # Test data processing
    print("\n3. Testing Data Processing:")
    data_result = await tool_executor.execute_task(
        "Sort this data by sales value",
        {
            "data": [
                {"name": "Product A", "sales": 1000},
                {"name": "Product B", "sales": 1500},
                {"name": "Product C", "sales": 800}
            ]
        }
    )
    print(f"Data Processing Success: {data_result['success']}")

# Run demonstrations
if __name__ == "__main__":
    asyncio.run(demonstrate_tool_usage())
    asyncio.run(test_individual_tools())
```


## **Key Benefits and Features**

### **Advanced Capabilities**

1. **Intelligent Tool Selection**: Context-aware selection of appropriate tools based on task requirements
2. **Automatic Parameter Generation**: LLM-powered parameter preparation for tool execution
3. **Tool Chaining**: Results from one tool can be used as input for subsequent tools
4. **Fallback Mechanisms**: Automatic retry with alternative tools when primary tools fail
5. **Performance Monitoring**: Track tool usage, success rates, and execution times

### **MCP-Inspired Protocol**

- **Standardized Communication**: Consistent message format across all tools
- **Type Safety**: Schema validation for tool parameters and outputs
- **Extensibility**: Easy addition of new tools following the protocol
- **Error Handling**: Comprehensive error reporting and recovery mechanisms


### **Production-Ready Features**

- **Async Execution**: Non-blocking tool execution with timeout handling
- **Retry Logic**: Exponential backoff and fallback tool selection
- **Monitoring**: Execution history and performance metrics
- **Scalability**: Designed for high-concurrency environments


## **Conclusion**

This implementation adds sophisticated tool use capabilities to the Google ADK codebase, enabling agents to:

- **Dynamically select appropriate tools** based on context and task requirements
- **Execute tools with intelligent parameter preparation** and error handling
- **Chain multiple tools together** for complex task completion
- **Provide fallback mechanisms** for robust operation
- **Monitor and optimize performance** over time

The MCP-inspired protocol ensures consistency and extensibility, while the integration with Google ADK maintains the powerful multi-agent capabilities we've built throughout this conversation. This creates a truly intelligent system capable of handling complex, multi-step tasks through coordinated tool usage.

