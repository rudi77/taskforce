using System;
using Microsoft.Extensions.Logging;

namespace Taskforce.Core.Exceptions
{
    public abstract class AgentError : Exception
    {
        protected readonly ILogger Logger;

        protected AgentError(string message, ILogger logger, Exception innerException = null)
            : base(message, innerException)
        {
            Logger = logger;
            Logger.LogError(message);
        }
    }

    public class AgentExecutionError : AgentError
    {
        public AgentExecutionError(string message, ILogger logger, Exception innerException = null)
            : base(message, logger, innerException)
        {
        }
    }

    public class AgentGenerationError : AgentError
    {
        public AgentGenerationError(string message, ILogger logger, Exception innerException = null)
            : base(message, logger, innerException)
        {
        }
    }

    public class AgentMaxStepsError : AgentError
    {
        public AgentMaxStepsError(string message, ILogger logger)
            : base(message, logger)
        {
        }
    }

    public class AgentParsingError : AgentError
    {
        public AgentParsingError(string message, ILogger logger, Exception innerException = null)
            : base(message, logger, innerException)
        {
        }
    }
} 