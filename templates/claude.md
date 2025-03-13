# Claude Agent Template

This document provides a boilerplate template for implementing an agentic system using Claude 3.7 with recursive reasoning and tool-based abilities.

## Overview

This template demonstrates how to create an agent that:
- Leverages Claude 3.7's advanced reasoning capabilities
- Uses tool calling for extensible functionality
- Maintains conversation history
- Processes queries recursively through a thinking loop

## Code Template

```python
"""
Agent Boilerplate Template

This template demonstrates how to implement an agent using Claude 3.7 with:
- Tool-based abilities
- Recursive reasoning
- Conversation history management
"""

import os
import json
import anthropic
from typing import Dict, List, Any, Optional, Tuple
from rich.console import Console

# Initialize console for rich output
console = Console()

# Define your tools
def get_tools():
    """
    Define the tools available to the agent using the Anthropic API format.
    """
    return [
        {
            "name": "tool_name",
            "description": "Description of what this tool does",
            "input_schema": {
                "type": "object",
                "properties": {
                    "param1": {
                        "type": "string",
                        "description": "Description of parameter 1"
                    },
                    "param2": {
                        "type": "number", 
                        "description": "Description of parameter 2"
                    }
                },
                "required": ["param1"]
            }
        }
        # Add more tools as needed
    ]

# Define the function implementations that correspond to the tools
def get_tool_mapping():
    """
    Maps tool names to their implementation functions.
    """
    return {
        "tool_name": execute_tool_function,
        # Add more mappings as needed
    }

def execute_tool_function(params: Dict) -> Dict:
    """
    Example implementation of a tool function.
    
    Args:
        params: Dictionary of parameters from the tool call
        
    Returns:
        Dictionary with result or error
    """
    try:
        # Implement the tool functionality
        param1 = params.get("param1")
        
        # Process the parameters and generate a result
        result = f"Processed {param1}"
        
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

def update_conversation_history(conversation_history, question=None, response=None):
    """
    Helper function to update conversation history.
    
    Args:
        conversation_history: The existing conversation history
        question: Optional question to add as a user message
        response: Optional response to add as an assistant message
        
    Returns:
        Updated conversation history
    """
    if conversation_history is None:
        conversation_history = []
        
    if question is not None:
        conversation_history.append({
            "role": "user",
            "content": [{"type": "text", "text": question}]
        })
        
    if response is not None:
        conversation_history.append({
            "role": "assistant", 
            "content": [{"type": "text", "text": response}]
        })
        
    return conversation_history

def run_agent(query: str, context_data: Any = None, max_tokens: int = 4000):
    """
    Run the agent to process a natural language query.
    
    Args:
        query: The natural language query to process
        context_data: Optional context data specific to your application
        max_tokens: Maximum number of tokens in the response
        
    Returns:
        The agent's response as a string and count of tool calls made
    """
    # Get API key from environment variable
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]Error: ANTHROPIC_API_KEY environment variable is not set.[/red]")
        return "Error: ANTHROPIC_API_KEY environment variable is not set.", 0
    
    # Define your system prompt
    system_prompt = """
    <purpose>
        You are an expert assistant that uses tools to accomplish tasks.
    </purpose>
    
    <instructions>
        <operations>
            Use the available tools to help answer questions and perform tasks.
            Think step by step to solve complex problems.
            Always explain your reasoning clearly.
        </operations>
        
        <tool_selection>
            Choose the appropriate tool based on the query type.
            For complex tasks, break them down into multiple tool calls.
        </tool_selection>
        
        <requirements>
            Provide complete answers.
            When uncertain, gather more information before providing a final answer.
            Format results in a clear, readable way.
        </requirements>
    </instructions>
    """
    
    # Prepare initial context
    initial_message = f"Query: {query}"
    
    # Add any application-specific context
    if context_data:
        initial_message += f"\nContext: {json.dumps(context_data)}"
    
    # Format as per Claude's content blocks requirements
    messages = [{"role": "user", "content": [{"type": "text", "text": initial_message}]}]
    
    # Map tool names to their corresponding functions
    tool_functions = get_tool_mapping()
    
    # Get tools for the agent
    tools = get_tools()
    
    # Initialize client
    client = anthropic.Anthropic(api_key=api_key)
    
    # Track tool calls
    total_tool_calls = 0
    
    # Track the final response from Claude
    final_response_text = ""
    
    # Process messages recursively until Claude provides a complete response
    while True:
        console.print("[yellow]Making API call to Claude[/yellow]")
        
        try:
            # Make the API call with thinking=true for recursive reasoning
            response = client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=max_tokens,
                messages=messages,
                system=system_prompt,
                tools=tools,
                temperature=0.0,  # Adjust as needed
            )
            console.print("[green]API call successful[/green]")
        except Exception as e:
            console.print(f"[red]Error in API call: {str(e)}[/red]")
            return f"Error in API call: {str(e)}", total_tool_calls
        
        # Extract text from response
        response_text = ""
        for block in response.content:
            if block.type == "text":
                console.print(f"[cyan]Claude:[/cyan] {block.text}")
                response_text += block.text
        
        # Update the final response text
        if response_text:
            final_response_text = response_text
        
        # Process any tool calls
        tool_calls = [block for block in response.content if block.type == "tool_use"]
        total_tool_calls += len(tool_calls)
        
        # If no tool calls, we're done!
        if not tool_calls:
            break
        
        # Add Claude's response to the messages
        messages.append({"role": "assistant", "content": response.content})
        
        # Process tool calls one at a time
        for tool in tool_calls:
            console.print(f"[blue]Tool Call:[/blue] {tool.name}({json.dumps(tool.input, indent=2)})")
            
            func = tool_functions.get(tool.name)
            if func:
                # Execute the tool
                output = func(tool.input)
                result_text = output.get("error") or output.get("result", "")
                console.print(f"[green]Tool Result:[/green] {result_text}")
                
                # Add the tool result to the messages
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tool.id,
                        "content": result_text
                    }]
                })
    
    # Return the final response text and tool call count
    return final_response_text, total_tool_calls

def main():
    """
    Example usage of the agent.
    """
    # Ensure API key is set
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        console.print("[red]Error: ANTHROPIC_API_KEY environment variable is not set.[/red]")
        return
    
    # Run the agent with a sample query
    query = "What can you help me with?"
    response, tool_calls = run_agent(query)
    
    console.print(f"\n[bold]Final Response:[/bold]")
    console.print(response)
    console.print(f"\n[bold]Tool calls made:[/bold] {tool_calls}")

if __name__ == "__main__":
    main()
```

## Key Components

### 1. Tool Definition

Tools define the capabilities available to the agent. Each tool specifies:
- A name for identification
- A description explaining its purpose
- An input schema with parameter definitions

### 2. Tool Implementation

The actual functionality of each tool is implemented as a Python function that:
- Accepts parameters as a dictionary
- Returns results or errors in a standardized format
- Handles exceptions gracefully

### 3. System Prompt

The system prompt defines the agent's:
- Purpose and role
- Operating instructions
- Tool selection guidance
- Response requirements

Use structured XML-like tags for clarity and organization.

### 4. Recursive Processing Loop

The agent processes queries through a recursive loop that:
1. Sends the current conversation state to Claude
2. Processes any tool calls made by Claude
3. Adds tool results back to the conversation
4. Continues until Claude provides a complete response

### 5. Conversation History Management

The agent manages conversation history by:
- Maintaining a list of message objects
- Adding user queries and assistant responses
- Formatting them according to Claude's API requirements

## Customization

Adapt this template for your specific use case by:

1. Defining domain-specific tools relevant to your application
2. Tailoring the system prompt for your specific domain
3. Adding specialized context processing for your data
4. Implementing custom tool functions for your needs
5. Adjusting the response formatting to match your requirements

## Best Practices

1. **Tool Design**
   - Keep tools focused on specific, discrete tasks
   - Use clear, descriptive names and parameters
   - Provide detailed error messages

2. **System Prompt**
   - Structure with XML-like tags for clarity
   - Be specific about when to use each tool
   - Include examples of proper tool usage

3. **Error Handling**
   - Gracefully handle tool execution errors
   - Provide clear error messages back to Claude
   - Implement retry logic for API calls

4. **Testing**
   - Test tools individually before integration
   - Create sample queries covering various scenarios
   - Monitor tool call patterns to optimize performance