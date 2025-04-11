using Microsoft.Extensions.Logging;
using Taskforce.Application;
using Taskforce.Configuration;
using Taskforce.Core.Agents;
using Taskforce.Core.Entities;
using Taskforce.Core.Interfaces;
using Taskforce.Core.Services;
using Taskforce.Domain.Entities;
using Taskforce.Domain.Interfaces;
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

        // Example receipt image path
        var receipts = new List<string> { @"C:\Users\rudi\Documents\Arbeit\CSS\297657.png" };
        List<byte[]> receipts_bytes = receipts.Select(File.ReadAllBytes).ToList();
        
        // Load configuration
        var config = TaskforceConfig.Create("C:/Users/rudi/source/repos/taskforce/src/Taskforce/Taskforce/Configuration/sample/taskforce_receipt.yaml");
        
        // Create LLM client
        var llmClient = new OpenAIChatClient();
        
        // Create tools for the agent
        var tools = CreateTools();
        
        // Create both agents from the configuration
        var markdownAgent = new ToolCallingAgent(
            tools: tools,
            model: llmClient,
            systemPrompt: config.AgentConfigs[0].Mission, // Markdown agent
            logger: LoggerFactory.CreateLogger("MarkdownAgent")
        );

        var extractionAgent = new ToolCallingAgent(
            tools: tools,
            model: llmClient,
            systemPrompt: config.AgentConfigs[1].Mission, // Agent Smith
            logger: LoggerFactory.CreateLogger("ExtractionAgent"),
            isExtractionAgent: true
        );
        
        // Execute both agents
        logger.LogInformation("Running Markdown Agent...");
        var markdownResponse = await markdownAgent.Run("Convert the uploaded invoice image into markdown format", receipts_bytes);
        await Console.Out.WriteLineAsync("Markdown response:\n" + markdownResponse);

        logger.LogInformation("Running Extraction Agent...");
        var extractionResponse = await extractionAgent.Run($"Extract all relevant receipt details from the following markdown:\n\n{markdownResponse}");
        await Console.Out.WriteLineAsync("\nExtraction response:\n" + extractionResponse);
    }

    static List<Tool> CreateTools()
    {
        // Create tools that the agent can use
        return new List<Tool>
        {
            new Tool(
                name: "extract_receipt_details",
                description: "Extract details from a receipt image",
                argumentType: typeof(string),
                execute: async (args) =>
                {
                    // This is a placeholder implementation
                    // In a real application, this would call an OCR service or similar
                    return "Receipt details extracted successfully";
                }
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
}
