using System.Text.Json;
using Microsoft.Extensions.Logging;
using Taskforce.Core.Entities;
using Taskforce.Core.Exceptions;
using Taskforce.Core.Interfaces;
using Taskforce.Core.Services;
using Taskforce.Domain.Interfaces;

namespace Taskforce.Core.Agents
{
    /// <summary>
    /// A specialized agent that implements tool-calling capabilities on top of the base MultiStepAgent.
    /// This agent can interpret LLM responses as tool calls, execute tools, and handle both structured (JSON)
    /// and unstructured responses. It also supports image processing capabilities.
    /// </summary>
    public class ToolCallingAgent : MultiStepAgent
    {
        /// <summary>
        /// Parser responsible for converting text responses into structured tool calls.
        /// </summary>
        private readonly IToolParser _toolParser;

        /// <summary>
        /// Collection of images that can be processed during the agent's execution.
        /// Used when the agent needs to analyze or reference images as part of its task.
        /// </summary>
        private List<byte[]> _images;

        /// <summary>
        /// Indicates whether this agent is specifically for data extraction tasks.
        /// When true, JSON responses are automatically wrapped in a final_answer tool call.
        /// </summary>
        private readonly bool _isExtractionAgent;

        /// <summary>
        /// Initializes a new instance of the ToolCallingAgent with specified capabilities.
        /// </summary>
        /// <param name="tools">List of tools available to the agent</param>
        /// <param name="model">Language model instance for reasoning</param>
        /// <param name="systemPrompt">Optional custom system prompt (uses default if null)</param>
        /// <param name="toolParser">Optional custom tool parser (uses DefaultToolParser if null)</param>
        /// <param name="maxSteps">Maximum allowed steps (defaults to 6)</param>
        /// <param name="logger">Optional logger instance</param>
        /// <param name="isExtractionAgent">Whether this agent is for data extraction (defaults to false)</param>
        public ToolCallingAgent(
            List<Tool> tools,
            LLMBase model,
            string? systemPrompt = null,
            IToolParser? toolParser = null,
            int maxSteps = 6,
            ILogger? logger = null,
            bool isExtractionAgent = false)
            : base(tools, model, systemPrompt ?? GetDefaultToolCallingSystemPrompt(), maxSteps, logger)
        {
            _toolParser = toolParser ?? new DefaultToolParser(logger);
            _images = new List<byte[]>();
            _isExtractionAgent = isExtractionAgent;
        }

        /// <summary>
        /// Provides the default system prompt for tool-calling functionality.
        /// Includes placeholders for tool descriptions that are populated at runtime.
        /// </summary>
        /// <returns>Default system prompt template for tool calling</returns>
        private static string GetDefaultToolCallingSystemPrompt() => @"
You are a helpful AI assistant that can use tools to accomplish tasks.
When you need to use a tool, specify the tool name and arguments in JSON format.
Available tools:
{{tool_descriptions}}

To provide a final answer, use the 'final_answer' tool.
";

        /// <summary>
        /// Overridden Run method that supports processing images alongside the text prompt.
        /// </summary>
        /// <param name="prompt">The initial prompt/task for the agent to process</param>
        /// <param name="images">Optional list of images to process</param>
        /// <returns>The final result as a string</returns>
        public async Task<string> Run(string prompt, List<byte[]>? images = null)
        {
            _images = images ?? new List<byte[]>();
            return await base.Run(prompt);
        }

        /// <summary>
        /// Implements the core step logic for tool calling. This method:
        /// 1. Prepares the context from memory
        /// 2. Gets a response from the language model
        /// 3. Parses the response into a tool call
        /// 4. Executes the appropriate tool
        /// 5. Handles the result
        /// </summary>
        /// <param name="log">Log object for recording step details</param>
        /// <returns>Final answer if task is complete, null to continue to next step</returns>
        /// <exception cref="AgentGenerationError">Thrown when model response is invalid</exception>
        /// <exception cref="AgentExecutionError">Thrown when tool execution fails</exception>
        protected override async Task<object?> Step(ActionStep log)
        {
            // Get agent's memory state
            var agentMemory = WriteInnerMemoryFromLogs();
            log.AgentMemory = agentMemory;

            try
            {
                // Format system prompt with tool descriptions
                var toolDescriptions = string.Join("\n", Tools.Select(t => t.ToToolDefinition()));
                var formattedSystemPrompt = SystemPrompt.Replace("{{tool_descriptions}}", toolDescriptions);

                // Format user prompt from memory
                var userPrompt = string.Join("\n", agentMemory.Select(m => $"{m["role"]}: {m["content"]}"));

                // Get model's response, handling both image and text-only cases
                object? modelResponse;
                if (_images.Count > 0)
                {
                    modelResponse = await Model.SendMessageAsync(formattedSystemPrompt, userPrompt, _images);
                }
                else
                {
                    modelResponse = await Model.SendMessageAsync(formattedSystemPrompt, userPrompt);
                }

                // Validate model response
                if (modelResponse == null)
                {
                    throw new AgentGenerationError("Model did not return a response", Logger);
                }

                var responseStr = modelResponse.ToString();
                if (string.IsNullOrEmpty(responseStr))
                {
                    throw new AgentGenerationError("Model returned an empty response", Logger);
                }

                // Determine if response is already in JSON format
                bool isJson = false;
                try
                {
                    JsonDocument.Parse(responseStr);
                    isJson = true;
                }
                catch (JsonException)
                {
                    // Not JSON, will handle below
                }

                // Parse or construct tool call based on response format and agent type
                ToolCall toolCall;
                if (isJson)
                {
                    if (_isExtractionAgent)
                    {
                        // For extraction agent, wrap the JSON in a final_answer tool call
                        var jsonResponse = JsonSerializer.Serialize(new
                        {
                            name = "final_answer",
                            arguments = responseStr,
                            id = Guid.NewGuid().ToString()
                        });
                        
                        toolCall = _toolParser.ParseToolCall(jsonResponse);
                    }
                    else
                    {
                        // Parse as normal tool call
                        toolCall = _toolParser.ParseToolCall(responseStr);
                    }
                }
                else
                {
                    // Wrap non-JSON response as final_answer
                    var jsonResponse = JsonSerializer.Serialize(new
                    {
                        name = "final_answer",
                        arguments = responseStr,
                        id = Guid.NewGuid().ToString()
                    });
                    
                    toolCall = _toolParser.ParseToolCall(jsonResponse);
                }
                
                log.ToolCalls = new List<ToolCall> { toolCall };

                // Execute tool call and handle result
                Logger.LogInformation($"Executing tool: {toolCall.Name} with arguments: {JsonSerializer.Serialize(toolCall.Arguments)}");

                if (toolCall.Name == "final_answer")
                {
                    var finalAnswer = ExtractFinalAnswer(toolCall.Arguments);
                    log.ActionOutput = finalAnswer;
                    return finalAnswer;
                }

                var tool = Tools.FirstOrDefault(t => t.Name == toolCall.Name) 
                    ?? throw new AgentExecutionError($"Unknown tool: {toolCall.Name}", Logger);

                var observation = await tool.Execute(toolCall.Arguments);
                log.Observations = observation?.ToString();

                return null; // Continue to next step
            }
            catch (Exception ex)
            {
                throw new AgentExecutionError($"Error in step execution: {ex.Message}", Logger, ex);
            }
        }

        /// <summary>
        /// Extracts the final answer from tool arguments, handling various response formats.
        /// Supports string responses, JSON elements, and objects with 'answer' properties.
        /// </summary>
        /// <param name="arguments">The arguments object to extract the answer from</param>
        /// <returns>The extracted answer, or null if extraction fails</returns>
        private object? ExtractFinalAnswer(object? arguments)
        {
            if (arguments == null)
            {
                return null;
            }

            if (arguments is string strArg)
            {
                return strArg;
            }

            if (arguments is JsonElement jsonElement)
            {
                try
                {
                    return jsonElement.GetProperty("answer").GetString() ?? jsonElement.ToString();
                }
                catch
                {
                    return jsonElement.ToString();
                }
            }

            return arguments.ToString();
        }
    }
} 