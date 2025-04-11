using System;
using Taskforce.Core.Exceptions;
using Taskforce.Domain.Interfaces;

namespace Taskforce.Core.Entities
{
    public abstract class AgentStepLog
    {
        public DateTime StartTime { get; set; }
        public DateTime EndTime { get; set; }
        public TimeSpan Duration => EndTime - StartTime;
    }

    public class ActionStep : AgentStepLog
    {
        public List<Dictionary<string, string>>? AgentMemory { get; set; }
        public List<ToolCall>? ToolCalls { get; set; }
        public int StepNumber { get; set; }
        public AgentError? Error { get; set; }
        public string? LlmOutput { get; set; }
        public string? Observations { get; set; }
        public List<string>? ObservationImages { get; set; }
        public object? ActionOutput { get; set; }
    }

    public class PlanningStep : AgentStepLog
    {
        public required string Plan { get; set; }
        public required string Facts { get; set; }
    }

    public class TaskStep : AgentStepLog
    {
        public required string Task { get; set; }
        public List<string>? TaskImages { get; set; }
    }

    public class SystemPromptStep : AgentStepLog
    {
        public required string SystemPrompt { get; set; }
    }
} 