"""Q&A Engine using LangChain agent"""
import os
from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.logging_config import get_logger
logger = get_logger(__name__)

# Import LangChain components
# Use bind_tools for tool calling (works with LangChain 1.x)
try:
    # Try to use agent framework (LangChain 0.3.x)
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    USE_AGENT = True
except ImportError:
    # Fallback: Use LLM with tools directly (LangChain 1.x compatible)
    USE_AGENT = False
    logger.warning("AgentExecutor not available. Using direct LLM with tools.")

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .tools import get_tools

class MedicalQAEngine:
    """Medical Q&A engine using LangChain agent"""
    
    def __init__(self):
        """Initialize the Q&A engine"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.openai_model,
            api_key=self.openai_api_key,
            temperature=0.7
        )
        
        # Get tools
        self.tools = get_tools()
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are a medical question answering expert. The user will ask you a question/query about medical topics.
                 
                 Answer the question to the best of your ability using the following guidelines:
                 - First, use the medical_knowledge_search_tool to find answers in medical documents
                 - If you cannot find a satisfactory answer in the medical knowledge base, use the internet_search_tool
                 - Provide clear, accurate, and helpful medical information
                 - If information is not available, be honest about it
                 - Always cite your sources when possible
                 """),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        
        # Create agent executor if available, otherwise use LLM with tools directly
        if USE_AGENT:
            self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=False,
                handle_parsing_errors=True
            )
            self.use_agent_executor = True
        else:
            # Use LLM with tools bound directly (LangChain 1.x approach)
            self.llm_with_tools = self.llm.bind_tools(self.tools)
            self.use_agent_executor = False
        
        logger.info(f"Medical Q&A engine initialized (use_agent={USE_AGENT})")
    
    def process_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process a medical query and return answer
        
        Args:
            query: The user's question
            conversation_history: Optional conversation history for context
            
        Returns:
            Dict with answer, sources, and metadata
        """
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Prepare input for agent
            if conversation_history:
                # Format conversation history
                messages = []
                for msg in conversation_history:
                    if isinstance(msg, dict):
                        if "input" in msg:
                            messages.append(("human", msg["input"]))
                        if "output" in msg:
                            messages.append(("ai", msg["output"]))
                    else:
                        messages.append(msg)
                messages.append(("human", query))
                agent_input = {"input": messages}
            else:
                agent_input = {"input": query}
            
            # Invoke agent or LLM with tools
            if self.use_agent_executor:
                response = self.agent_executor.invoke(agent_input)
                answer = response.get("output", "I couldn't generate a response.")
            else:
                # Direct LLM invocation with tools (LangChain 1.x)
                # Need to handle tool calls properly - execute tools when LLM requests them
                from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
                
                messages = [HumanMessage(content=query)]
                max_iterations = 5  # Prevent infinite loops
                iteration = 0
                sources_used = []
                
                while iteration < max_iterations:
                    # Invoke LLM with current messages
                    response = self.llm_with_tools.invoke(messages)
                    messages.append(response)
                    
                    # Check if LLM wants to call tools
                    # In LangChain 1.x, tool_calls might be in response.additional_kwargs or directly in response
                    tool_calls = None
                    if hasattr(response, 'tool_calls') and response.tool_calls:
                        tool_calls = response.tool_calls
                    elif hasattr(response, 'additional_kwargs') and 'tool_calls' in response.additional_kwargs:
                        tool_calls = response.additional_kwargs['tool_calls']
                    
                    if tool_calls:
                        logger.info(f"LLM requested {len(tool_calls)} tool call(s)")
                        
                        # Execute each tool call
                        for tool_call in tool_calls:
                            # Handle different tool_call formats
                            if isinstance(tool_call, dict):
                                tool_name = tool_call.get('name', tool_call.get('function', {}).get('name', ''))
                                tool_input = tool_call.get('args', tool_call.get('function', {}).get('arguments', {}))
                                tool_id = tool_call.get('id', tool_call.get('function', {}).get('id', ''))
                                
                                # Parse tool_input if it's a string (JSON)
                                if isinstance(tool_input, str):
                                    import json
                                    try:
                                        tool_input = json.loads(tool_input)
                                    except:
                                        tool_input = {"query": tool_input}
                            else:
                                # Tool call might be an object
                                tool_name = getattr(tool_call, 'name', '')
                                tool_input = getattr(tool_call, 'args', {})
                                tool_id = getattr(tool_call, 'id', '')
                            
                            logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
                            
                            # Find and execute the tool
                            tool_result = None
                            for tool in self.tools:
                                if tool.name == tool_name:
                                    try:
                                        # Extract query from tool_input (could be dict or string)
                                        if isinstance(tool_input, dict):
                                            query_param = tool_input.get('query', '')
                                        else:
                                            query_param = str(tool_input)
                                        
                                        logger.info(f"[TOOL] Invoking tool {tool_name} with query: '{query_param}'")
                                        # Invoke tool with proper format
                                        tool_result = tool.invoke({"query": query_param})
                                        sources_used.append(tool_name)
                                        logger.info(f"[TOOL] Tool {tool_name} executed successfully. Result length: {len(str(tool_result))}")
                                        logger.info(f"[TOOL] Tool result preview (first 300 chars): {str(tool_result)[:300]}...")
                                    except Exception as e:
                                        tool_result = f"Error executing tool {tool_name}: {str(e)}"
                                        logger.error(f"[TOOL] ERROR: Tool execution error: {str(e)}", exc_info=True)
                                    break
                            
                            # Add tool result to messages so LLM can process it
                            if tool_result is not None:
                                messages.append(ToolMessage(
                                    content=str(tool_result),
                                    tool_call_id=tool_id
                                ))
                        
                        iteration += 1
                        continue  # Loop back to let LLM process tool results
                    else:
                        # No tool calls, we have the final answer
                        answer = response.content if hasattr(response, 'content') else str(response)
                        logger.info("LLM provided final answer without tool calls")
                        break
                else:
                    # Max iterations reached
                    answer = "I'm having trouble processing your query. Please try again."
                    logger.warning("Max iterations reached in tool calling loop")
                
                # Use sources from tool calls
                if not sources_used:
                    sources_used = ["Medical Knowledge Base"]  # Default
            
            logger.info(f"Query processed successfully. Answer length: {len(answer)}")
            
            # Extract sources
            if self.use_agent_executor:
                sources = self._extract_sources(response)
            else:
                # For LangChain 1.x, use sources from tool calls
                sources = sources_used if 'sources_used' in locals() else ["Medical Knowledge Base"]
            
            return {
                "answer": answer,
                "query": query,
                "sources": sources,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "answer": f"I encountered an error while processing your query: {str(e)}",
                "query": query,
                "sources": [],
                "status": "error",
                "error": str(e)
            }
    
    def _extract_sources(self, response: Dict[str, Any]) -> List[str]:
        """Extract source information from agent response"""
        # The agent response may contain intermediate steps with tool calls
        # Extract sources from tool results if available
        sources = []
        
        if "intermediate_steps" in response:
            for step in response["intermediate_steps"]:
                if len(step) > 1:
                    tool_name = step[0].tool if hasattr(step[0], 'tool') else None
                    if tool_name == "medical_knowledge_search_tool":
                        sources.append("Medical Knowledge Base")
                    elif tool_name == "internet_search_tool":
                        sources.append("Web Search")
        
        # If no sources found, add default
        if not sources:
            sources.append("Medical Knowledge Base")
        
        return sources


from typing import Dict, Any, Optional, List
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.logging_config import get_logger
logger = get_logger(__name__)

# Import LangChain components
# Use bind_tools for tool calling (works with LangChain 1.x)
try:
    # Try to use agent framework (LangChain 0.3.x)
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    USE_AGENT = True
except ImportError:
    # Fallback: Use LLM with tools directly (LangChain 1.x compatible)
    USE_AGENT = False
    logger.warning("AgentExecutor not available. Using direct LLM with tools.")

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from .tools import get_tools

class MedicalQAEngine:
    """Medical Q&A engine using LangChain agent"""
    
    def __init__(self):
        """Initialize the Q&A engine"""
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.openai_model,
            api_key=self.openai_api_key,
            temperature=0.7
        )
        
        # Get tools
        self.tools = get_tools()
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", """You are a medical question answering expert. The user will ask you a question/query about medical topics.
                 
                 Answer the question to the best of your ability using the following guidelines:
                 - First, use the medical_knowledge_search_tool to find answers in medical documents
                 - If you cannot find a satisfactory answer in the medical knowledge base, use the internet_search_tool
                 - Provide clear, accurate, and helpful medical information
                 - If information is not available, be honest about it
                 - Always cite your sources when possible
                 """),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        
        # Create agent executor if available, otherwise use LLM with tools directly
        if USE_AGENT:
            self.agent = create_tool_calling_agent(self.llm, self.tools, self.prompt)
            self.agent_executor = AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=False,
                handle_parsing_errors=True
            )
            self.use_agent_executor = True
        else:
            # Use LLM with tools bound directly (LangChain 1.x approach)
            self.llm_with_tools = self.llm.bind_tools(self.tools)
            self.use_agent_executor = False
        
        logger.info(f"Medical Q&A engine initialized (use_agent={USE_AGENT})")
    
    def process_query(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Process a medical query and return answer
        
        Args:
            query: The user's question
            conversation_history: Optional conversation history for context
            
        Returns:
            Dict with answer, sources, and metadata
        """
        try:
            logger.info(f"Processing query: {query[:100]}...")
            
            # Prepare input for agent
            if conversation_history:
                # Format conversation history
                messages = []
                for msg in conversation_history:
                    if isinstance(msg, dict):
                        if "input" in msg:
                            messages.append(("human", msg["input"]))
                        if "output" in msg:
                            messages.append(("ai", msg["output"]))
                    else:
                        messages.append(msg)
                messages.append(("human", query))
                agent_input = {"input": messages}
            else:
                agent_input = {"input": query}
            
            # Invoke agent or LLM with tools
            if self.use_agent_executor:
                response = self.agent_executor.invoke(agent_input)
                answer = response.get("output", "I couldn't generate a response.")
            else:
                # Direct LLM invocation with tools (LangChain 1.x)
                # Need to handle tool calls properly - execute tools when LLM requests them
                from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
                
                messages = [HumanMessage(content=query)]
                max_iterations = 5  # Prevent infinite loops
                iteration = 0
                sources_used = []
                
                while iteration < max_iterations:
                    # Invoke LLM with current messages
                    response = self.llm_with_tools.invoke(messages)
                    messages.append(response)
                    
                    # Check if LLM wants to call tools
                    # In LangChain 1.x, tool_calls might be in response.additional_kwargs or directly in response
                    tool_calls = None
                    if hasattr(response, 'tool_calls') and response.tool_calls:
                        tool_calls = response.tool_calls
                    elif hasattr(response, 'additional_kwargs') and 'tool_calls' in response.additional_kwargs:
                        tool_calls = response.additional_kwargs['tool_calls']
                    
                    if tool_calls:
                        logger.info(f"LLM requested {len(tool_calls)} tool call(s)")
                        
                        # Execute each tool call
                        for tool_call in tool_calls:
                            # Handle different tool_call formats
                            if isinstance(tool_call, dict):
                                tool_name = tool_call.get('name', tool_call.get('function', {}).get('name', ''))
                                tool_input = tool_call.get('args', tool_call.get('function', {}).get('arguments', {}))
                                tool_id = tool_call.get('id', tool_call.get('function', {}).get('id', ''))
                                
                                # Parse tool_input if it's a string (JSON)
                                if isinstance(tool_input, str):
                                    import json
                                    try:
                                        tool_input = json.loads(tool_input)
                                    except:
                                        tool_input = {"query": tool_input}
                            else:
                                # Tool call might be an object
                                tool_name = getattr(tool_call, 'name', '')
                                tool_input = getattr(tool_call, 'args', {})
                                tool_id = getattr(tool_call, 'id', '')
                            
                            logger.info(f"Executing tool: {tool_name} with input: {tool_input}")
                            
                            # Find and execute the tool
                            tool_result = None
                            for tool in self.tools:
                                if tool.name == tool_name:
                                    try:
                                        # Extract query from tool_input (could be dict or string)
                                        if isinstance(tool_input, dict):
                                            query_param = tool_input.get('query', '')
                                        else:
                                            query_param = str(tool_input)
                                        
                                        logger.info(f"[TOOL] Invoking tool {tool_name} with query: '{query_param}'")
                                        # Invoke tool with proper format
                                        tool_result = tool.invoke({"query": query_param})
                                        sources_used.append(tool_name)
                                        logger.info(f"[TOOL] Tool {tool_name} executed successfully. Result length: {len(str(tool_result))}")
                                        logger.info(f"[TOOL] Tool result preview (first 300 chars): {str(tool_result)[:300]}...")
                                    except Exception as e:
                                        tool_result = f"Error executing tool {tool_name}: {str(e)}"
                                        logger.error(f"[TOOL] ERROR: Tool execution error: {str(e)}", exc_info=True)
                                    break
                            
                            # Add tool result to messages so LLM can process it
                            if tool_result is not None:
                                messages.append(ToolMessage(
                                    content=str(tool_result),
                                    tool_call_id=tool_id
                                ))
                        
                        iteration += 1
                        continue  # Loop back to let LLM process tool results
                    else:
                        # No tool calls, we have the final answer
                        answer = response.content if hasattr(response, 'content') else str(response)
                        logger.info("LLM provided final answer without tool calls")
                        break
                else:
                    # Max iterations reached
                    answer = "I'm having trouble processing your query. Please try again."
                    logger.warning("Max iterations reached in tool calling loop")
                
                # Use sources from tool calls
                if not sources_used:
                    sources_used = ["Medical Knowledge Base"]  # Default
            
            logger.info(f"Query processed successfully. Answer length: {len(answer)}")
            
            # Extract sources
            if self.use_agent_executor:
                sources = self._extract_sources(response)
            else:
                # For LangChain 1.x, use sources from tool calls
                sources = sources_used if 'sources_used' in locals() else ["Medical Knowledge Base"]
            
            return {
                "answer": answer,
                "query": query,
                "sources": sources,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "answer": f"I encountered an error while processing your query: {str(e)}",
                "query": query,
                "sources": [],
                "status": "error",
                "error": str(e)
            }
    
    def _extract_sources(self, response: Dict[str, Any]) -> List[str]:
        """Extract source information from agent response"""
        # The agent response may contain intermediate steps with tool calls
        # Extract sources from tool results if available
        sources = []
        
        if "intermediate_steps" in response:
            for step in response["intermediate_steps"]:
                if len(step) > 1:
                    tool_name = step[0].tool if hasattr(step[0], 'tool') else None
                    if tool_name == "medical_knowledge_search_tool":
                        sources.append("Medical Knowledge Base")
                    elif tool_name == "internet_search_tool":
                        sources.append("Web Search")
        
        # If no sources found, add default
        if not sources:
            sources.append("Medical Knowledge Base")
        
        return sources

