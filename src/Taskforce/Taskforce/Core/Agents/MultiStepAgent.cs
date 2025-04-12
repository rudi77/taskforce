using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Taskforce.Core.Entities;
using Taskforce.Core.Exceptions;
using Taskforce.Domain.Interfaces;
using Taskforce.Domain.Services;

namespace Taskforce.Core.Agents
{
    /// <summary>
    /// A multi-step agent that implements the ReAct (Reason + Act) pattern for executing complex tasks.
    /// This abstract class provides the core infrastructure for step-based task execution with memory
    /// management, tool integration, and logging capabilities.
    /// </summary>
    public abstract class MultiStepAgent
    {
        /// <summary>
        /// Collection of tools available to the agent for performing actions.
        /// Each tool represents a specific capability the agent can use.
        /// </summary>
        protected readonly List<Tool> Tools;

        /// <summary>
        /// The language model instance used by the agent for reasoning and decision making.
        /// </summary>
        protected readonly LLMBase Model;

        /// <summary>
        /// The system prompt that defines the agent's behavior and capabilities.
        /// This is sent at the start of each conversation with the LLM.
        /// </summary>
        protected readonly string SystemPrompt;

        /// <summary>
        /// Maximum number of steps the agent can take before throwing an exception.
        /// Prevents infinite loops and ensures task completion within bounds.
        /// </summary>
        protected readonly int MaxSteps;

        /// <summary>
        /// Logger instance for recording agent activities and debugging information.
        /// </summary>
        protected readonly ILogger Logger;

        /// <summary>
        /// Chronological record of all steps taken by the agent during execution.
        /// Used for debugging, monitoring, and maintaining conversation context.
        /// </summary>
        protected readonly List<AgentStepLog> Logs;

        /// <summary>
        /// Manager for handling the agent's memory, including storing and retrieving
        /// context from previous steps.
        /// </summary>
        protected readonly MemoryManager Memory;

        /// <summary>
        /// Dictionary for maintaining arbitrary state data between steps.
        /// Allows the agent to persist information across iterations.
        /// </summary>
        protected readonly Dictionary<string, object> State;

        /// <summary>
        /// Current step number in the execution sequence.
        /// Used for tracking progress and enforcing MaxSteps limit.
        /// </summary>
        protected int StepNumber;

        /// <summary>
        /// Initializes a new instance of the MultiStepAgent with specified tools and capabilities.
        /// </summary>
        /// <param name="tools">List of tools available to the agent</param>
        /// <param name="model">Language model instance for reasoning</param>
        /// <param name="systemPrompt">Optional custom system prompt (uses default if null)</param>
        /// <param name="maxSteps">Maximum allowed steps (defaults to 6)</param>
        /// <param name="logger">Optional logger instance</param>
        /// <exception cref="ArgumentNullException">Thrown if tools or model is null</exception>
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

        /// <summary>
        /// Default system prompt used when no custom prompt is provided.
        /// Can be overridden by derived classes to provide specialized behavior.
        /// </summary>
        protected virtual string DefaultSystemPrompt => "You are a helpful AI assistant.";

        /// <summary>
        /// Primary entry point for running the agent with a given prompt.
        /// </summary>
        /// <param name="prompt">The initial prompt/task for the agent to process</param>
        /// <returns>The final result as a string, or empty string if no result</returns>
        public virtual async Task<string> Run(string prompt)
        {
            var result = await ExecuteSteps(prompt);
            return result?.ToString() ?? string.Empty;
        }

        /// <summary>
        /// Core execution loop implementing the ReAct pattern. Manages the step-by-step
        /// execution process, including memory management, logging, and error handling.
        /// </summary>
        /// <param name="prompt">Initial prompt to begin execution with</param>
        /// <returns>The final result object, or null if no result was produced</returns>
        /// <exception cref="AgentMaxStepsError">Thrown when maximum steps are exceeded</exception>
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

        /// <summary>
        /// Abstract method that defines the actual reasoning and action logic for each step.
        /// Must be implemented by derived classes to define specific agent behavior.
        /// </summary>
        /// <param name="log">Log object for recording step details</param>
        /// <returns>Result object if task is complete, null to continue to next step</returns>
        protected abstract Task<object?> Step(ActionStep log);

        /// <summary>
        /// Constructs the conversation history for the LLM, including system prompt
        /// and relevant memory contents. Used to maintain context between steps.
        /// </summary>
        /// <returns>List of message dictionaries representing the conversation history</returns>
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