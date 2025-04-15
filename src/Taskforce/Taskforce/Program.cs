using Microsoft.Extensions.Logging;
using Taskforce.Configuration;
using Taskforce.Core.Agents;
using Taskforce.Core.Entities;
using Taskforce.Core.Tools;
using Taskforce.Infrastructure.LLM;

internal class Program
{
    private static readonly ILoggerFactory LoggerFactory = Microsoft.Extensions.Logging.LoggerFactory.Create(builder =>
    {
        builder.AddConsole(); // Adds the console logger
    });

    static async Task Main(string[] args)
    {
        var logger = LoggerFactory.CreateLogger("Program");
        logger.LogInformation("Application starting...");
        
        // Create LLM client
        var llmClient = new OpenAIChatClient();
        
        // Create tools for the agent
        var tools = CreateTools();
        
        // Create a web search agent
        var webSearchAgent = new ToolCallingAgent(
            tools: tools,
            model: llmClient,
            systemPrompt: GetWebSearchSystemPrompt(),
            logger: LoggerFactory.CreateLogger("WebSearchAgent")
        );
        
        // Execute the agent with a search query
        logger.LogInformation("Running Web Search Agent...");
        var searchQuery = "quantum%20computing?";
        logger.LogInformation($"Search query: {searchQuery}");
        
        var searchResponse = await webSearchAgent.Run(searchQuery);
        await Console.Out.WriteLineAsync("\nSearch response:\n" + searchResponse);
    }

    static List<Tool> CreateTools()
    {
        // Create tools that the agent can use
        return new List<Tool>
        {
            new WebSearchTool(
                logger: LoggerFactory.CreateLogger("WebSearchTool"),
                maxResults: 5
            ),
            new Tool(
                name: "final_answer",
                description: "Provide the final answer to the user",
                argumentType: typeof(string),
                execute: async (args) =>
                {
                    // This tool is used by the agent to provide the final answer
                    return args.ToString();
                }
            )
        };
    }
    
    static string GetWebSearchSystemPrompt() => @"
You are a helpful AI assistant that can search the web for information.
When a user asks a question, use the web_search tool to find relevant information.
After searching, analyze the results and provide a comprehensive answer.

Available tools:
1. web_search: Search the web for information about a given query
   - Parameters:
     - query (string): The search query to execute

2. final_answer: Provide the final answer to the user
   - Parameters:
     - text (string): The final answer to provide to the user

IMPORTANT: When using a tool, you MUST format your response as a JSON object with the following structure:
{
  ""name"": ""tool_name"",
  ""arguments"": { ""parameter_name"": ""parameter_value"" },
  ""id"": ""unique_id""
}

For example, to search for quantum computing:
{
  ""name"": ""web_search"",
  ""arguments"": { ""query"": ""latest developments in quantum computing 2023"" },
  ""id"": ""search-1""
}

To provide a final answer:
{
  ""name"": ""final_answer"",
  ""arguments"": ""Your comprehensive answer here based on the search results"",
  ""id"": ""answer-1""
}
";
}
