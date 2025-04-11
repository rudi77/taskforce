using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Taskforce.Core.Entities;
using Taskforce.Core.Exceptions;
using Taskforce.Core.Interfaces;
using Taskforce.Core.Services;
using Taskforce.Domain.Interfaces;
using Taskforce.Domain.Services;

namespace Taskforce.Core.Agents
{
    public class ToolCallingAgent : MultiStepAgent
    {
        private readonly IToolParser _toolParser;
        private List<byte[]> _images;
        private readonly bool _isExtractionAgent;

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

        private static string GetDefaultToolCallingSystemPrompt() => @"
You are a helpful AI assistant that can use tools to accomplish tasks.
When you need to use a tool, specify the tool name and arguments in JSON format.
Available tools:
{{tool_descriptions}}

To provide a final answer, use the 'final_answer' tool.
";

        public async Task<string> Run(string prompt, List<byte[]>? images = null)
        {
            _images = images ?? new List<byte[]>();
            return await base.Run(prompt);
        }

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

                // Get model's response
                object? modelResponse;
                if (_images.Count > 0)
                {
                    modelResponse = await Model.SendMessageAsync(formattedSystemPrompt, userPrompt, _images);
                }
                else
                {
                    modelResponse = await Model.SendMessageAsync(formattedSystemPrompt, userPrompt);
                }

                if (modelResponse == null)
                {
                    throw new AgentGenerationError("Model did not return a response", Logger);
                }

                var responseStr = modelResponse.ToString();
                if (string.IsNullOrEmpty(responseStr))
                {
                    throw new AgentGenerationError("Model returned an empty response", Logger);
                }

                // Check if the response is already in JSON format
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
                        // Parse as normal
                        toolCall = _toolParser.ParseToolCall(responseStr);
                    }
                }
                else
                {
                    // Wrap the response in a JSON format for the final_answer tool
                    var jsonResponse = JsonSerializer.Serialize(new
                    {
                        name = "final_answer",
                        arguments = responseStr,
                        id = Guid.NewGuid().ToString()
                    });
                    
                    toolCall = _toolParser.ParseToolCall(jsonResponse);
                }
                
                log.ToolCalls = new List<ToolCall> { toolCall };

                // Execute tool call
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