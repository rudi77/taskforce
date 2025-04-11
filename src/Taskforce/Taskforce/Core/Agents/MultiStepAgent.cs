using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Taskforce.Core.Entities;
using Taskforce.Core.Exceptions;
using Taskforce.Domain.Interfaces;
using Taskforce.Domain.Services;

namespace Taskforce.Core.Agents
{
    public abstract class MultiStepAgent
    {
        protected readonly List<Tool> Tools;
        protected readonly LLMBase Model;
        protected readonly string SystemPrompt;
        protected readonly int MaxSteps;
        protected readonly ILogger Logger;
        protected readonly List<AgentStepLog> Logs;
        protected readonly MemoryManager Memory;
        protected readonly Dictionary<string, object> State;
        protected int StepNumber;

        protected MultiStepAgent(
            List<Tool> tools,
            LLMBase model,
            string systemPrompt = null,
            int maxSteps = 6,
            ILogger logger = null)
        {
            Tools = tools ?? throw new ArgumentNullException(nameof(tools));
            Model = model ?? throw new ArgumentNullException(nameof(model));
            SystemPrompt = systemPrompt ?? DefaultSystemPrompt;
            MaxSteps = maxSteps;
            Logger = logger ?? NullLogger.Instance;
            Logs = new List<AgentStepLog>();
            Memory = new MemoryManager(new ShortTermMemory());
            State = new Dictionary<string, object>();
            StepNumber = 0;
        }

        protected virtual string DefaultSystemPrompt => "You are a helpful AI assistant.";

        public virtual async Task<string> Run(string prompt)
        {
            var result = await ExecuteSteps(prompt);
            return result?.ToString() ?? string.Empty;
        }

        protected async Task<object?> ExecuteSteps(string prompt)
        {
            StepNumber = 0;
            Logs.Clear();
            Memory.Clear();

            // Add initial prompt to memory
            Memory.Store(prompt);

            while (StepNumber < MaxSteps)
            {
                var log = new ActionStep
                {
                    StepNumber = StepNumber,
                    StartTime = DateTime.UtcNow
                };

                try
                {
                    var result = await Step(log);
                    if (result != null)
                    {
                        log.EndTime = DateTime.UtcNow;
                        Logs.Add(log);
                        return result;
                    }
                }
                catch (Exception ex)
                {
                    log.Error = new AgentExecutionError($"Error in step {StepNumber}: {ex.Message}", Logger, ex);
                    Logs.Add(log);
                    throw;
                }

                log.EndTime = DateTime.UtcNow;
                Logs.Add(log);
                StepNumber++;
            }

            throw new AgentMaxStepsError($"Agent exceeded maximum steps ({MaxSteps})", Logger);
        }

        protected abstract Task<object?> Step(ActionStep log);

        protected List<Dictionary<string, string>> WriteInnerMemoryFromLogs()
        {
            var messages = new List<Dictionary<string, string>>();

            // Add system prompt
            messages.Add(new Dictionary<string, string>
            {
                ["role"] = "system",
                ["content"] = SystemPrompt
            });

            // Add memory messages
            var memoryContent = Memory.Retrieve();
            if (!string.IsNullOrEmpty(memoryContent))
            {
                messages.Add(new Dictionary<string, string>
                {
                    ["role"] = "user",
                    ["content"] = memoryContent
                });
            }

            return messages;
        }
    }
} 