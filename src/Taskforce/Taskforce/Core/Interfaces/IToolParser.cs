using System.Collections.Generic;
using Taskforce.Core.Entities;

namespace Taskforce.Core.Interfaces
{
    public interface IToolParser
    {
        ToolCall ParseToolCall(string content);
        List<ToolCall> ParseToolCalls(string content);
    }
} 