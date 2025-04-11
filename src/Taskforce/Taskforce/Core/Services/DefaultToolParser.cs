using System;
using System.Collections.Generic;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Taskforce.Core.Entities;
using Taskforce.Core.Interfaces;
using Taskforce.Core.Exceptions;

namespace Taskforce.Core.Services
{
    public class DefaultToolParser : IToolParser
    {
        private readonly ILogger _logger;

        public DefaultToolParser(ILogger? logger = null)
        {
            _logger = logger ?? NullLogger.Instance;
        }

        public ToolCall ParseToolCall(string content)
        {
            if (string.IsNullOrEmpty(content))
            {
                throw new AgentParsingError("Content cannot be null or empty", _logger);
            }

            try
            {
                var jsonDoc = JsonDocument.Parse(content);
                var root = jsonDoc.RootElement;

                var name = root.GetProperty("name").GetString();
                var arguments = root.GetProperty("arguments").GetRawText();
                var id = root.GetProperty("id").GetString();

                if (string.IsNullOrEmpty(name) || string.IsNullOrEmpty(arguments) || string.IsNullOrEmpty(id))
                {
                    throw new AgentParsingError("Tool call properties cannot be null or empty", _logger);
                }

                return new ToolCall
                {
                    Name = name,
                    Arguments = arguments,
                    Id = id
                };
            }
            catch (JsonException ex)
            {
                throw new AgentParsingError($"Failed to parse tool call: {ex.Message}", _logger, ex);
            }
        }

        public List<ToolCall> ParseToolCalls(string content)
        {
            if (string.IsNullOrEmpty(content))
            {
                throw new AgentParsingError("Content cannot be null or empty", _logger);
            }

            try
            {
                var jsonDoc = JsonDocument.Parse(content);
                var root = jsonDoc.RootElement;

                if (root.ValueKind != JsonValueKind.Array)
                {
                    throw new AgentParsingError("Tool calls must be an array", _logger);
                }

                var toolCalls = new List<ToolCall>();
                foreach (var element in root.EnumerateArray())
                {
                    var name = element.GetProperty("name").GetString();
                    var arguments = element.GetProperty("arguments").GetRawText();
                    var id = element.GetProperty("id").GetString();

                    if (string.IsNullOrEmpty(name) || string.IsNullOrEmpty(arguments) || string.IsNullOrEmpty(id))
                    {
                        throw new AgentParsingError("Tool call properties cannot be null or empty", _logger);
                    }

                    toolCalls.Add(new ToolCall
                    {
                        Name = name,
                        Arguments = arguments,
                        Id = id
                    });
                }

                return toolCalls;
            }
            catch (JsonException ex)
            {
                throw new AgentParsingError($"Failed to parse tool calls: {ex.Message}", _logger, ex);
            }
        }
    }
} 